
# & $env:MAMBA_EXE install -y matplotlib opencv=4.7.0 pyqt numpy scipy transformers pytorch torchvision torchaudio pytorch-cuda=11.7 -c conda-forge -c huggingface -c pytorch -c nvidia
# pip install d3dshot dxcam Pillow==7.1.2 pywin32

# from transformers import AutoTokenizer, AutoFeatureExtractor
# segmenter_tokenizer = AutoTokenizer.from_pretrained("facebook/mask2former-swin-large-cityscapes-semantic")
# depther_extractor = AutoFeatureExtractor.from_pretrained("Intel/dpt-large")

# from transformers import Mask2FormerForUniversalSegmentation, AutoModelForDepthEstimation
# segmenter = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-cityscapes-semantic")
# depther = AutoModelForDepthEstimation.from_pretrained("Intel/dpt-large")

from collections import OrderedDict

import cv2, dxcam, win32gui, time, numpy as np

from PIL import Image

# Ignore warning about `label_ids_to_fuse` unset.
import transformers
transformers.utils.logging.set_verbosity(transformers.utils.logging.ERROR)

# Check whether pytorch can use the GPU.
import torch, torchvision
print("CUDA available:", torch.cuda.is_available())

from transformers import GLPNImageProcessor, GLPNForDepthEstimation
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from transformers import pipeline

# from typing import Optional, Union, List
# from transformers.utils.generic import TensorType
from transformers.image_processing_utils import BatchFeature
from transformers.image_utils import valid_images, PILImageResampling


def clamped_resize(image: torch.tensor, size_divisor, resample='bilinear') -> torch.tensor:
    """
    Resize the image, rounding the (height, width) dimensions down to the closest multiple of size_divisor.

    If the image is of dimension (3, 260, 170) and size_divisor is 32, the image will be resized to (3, 256, 160).
    """

    if isinstance(resample, int):
        # Use the PILImageResampling enum
        for key in PILImageResampling.__members__.keys():
            if PILImageResampling[key] == resample:
                resample = str(key).lower()
                break

    height, width = image.shape[-2:]
    # Rounds the height and width down to the closest multiple of size_divisor
    if isinstance(size_divisor, int):
        size_divisor = dict(height=size_divisor, width=size_divisor)
    size_divisor_h = size_divisor.get('height', 1)
    size_divisor_w = size_divisor.get('width', 1)
    new_h = max(size_divisor_h, height // size_divisor_h * size_divisor_h)
    new_w = max(size_divisor_w, width // size_divisor_w * size_divisor_w)

    # If needed, add a batch_dimension:
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
        did_add_batch_dim = True
    else:
        did_add_batch_dim = False
    image = torch.nn.functional.interpolate(image, size=(new_h, new_w), mode=resample, align_corners=False)
    if did_add_batch_dim:
        image = image.squeeze(0)

    return image


def plain_rescale(image: torch.tensor, scale: float) -> torch.tensor:
    return image * scale


class CUDAGLPNImageProcessor(GLPNImageProcessor):
    """Remove the dumb cast to numpy in the parent"""

    def preprocess(
        self, images, do_resize=None, size_divisor=None, resample='bilinear', do_rescale=None):
        """
        Preprocess the given images.

        Args:
            images (`PIL.Image.Image` or `TensorType` or `List[np.ndarray]` or `List[TensorType]`):
                The image or images to preprocess.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the input such that the (height, width) dimensions are a multiple of `size_divisor`.
            size_divisor (`int`, *optional*, defaults to `self.size_divisor`):
                When `do_resize` is `True`, images are resized so their height and width are rounded down to the
                closest multiple of `size_divisor`.
            resample (`PIL.Image` resampling filter, *optional*, defaults to `self.resample`):
                `PIL.Image` resampling filter to use if resizing the image e.g. `PILImageResampling.BILINEAR`. Only has
                an effect if `do_resize` is set to `True`.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether or not to apply the scaling factor (to make pixel values floats between 0. and 1.).
        """
        do_resize = do_resize if do_resize is not None else self.do_resize
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        size_divisor = size_divisor if size_divisor is not None else self.size_divisor
        resample = resample if resample is not None else self.resample

        if do_resize and size_divisor is None:
            raise ValueError("size_divisor is required for resizing")

        if not valid_images(images):
            raise ValueError("Invalid image(s)")

        # All transformations expect numpy arrays. NOT
        # images = [to_numpy_array(img) for img in images]

        if do_resize:
            images = self.resize(images, size_divisor=size_divisor, resample=resample)

        if do_rescale:
            images = self.rescale(images, scale=1 / 255)

        data = {"pixel_values": images}
        return BatchFeature(data=data, tensor_type='pt')
    
    def resize(self, image: torch.tensor, size_divisor: int, resample='bilinear') -> torch.tensor:
        return clamped_resize(image, size_divisor, resample=resample)

    def rescale(self, image: torch.tensor, scale: float) -> torch.tensor:
        return plain_rescale(image, scale)


class CUDASegformerImageProcessor(SegformerImageProcessor):
    """Remove the cast to numpy in the parent class."""
    
    def resize(self, image, size, resample='bilinear'):
        return clamped_resize(image, size, resample=resample)
    
    def rescale(self, image, scale):
        return plain_rescale(image, scale)
    
    def _preprocess_image(
        self, image, do_resize=None, size=None, resample='bilinear', do_rescale=None, rescale_factor=None,
        do_normalize=None, image_mean=None, image_std=None, **kw_ignored) -> torch.tensor:
        """Preprocesses a single image."""
        # All transformations DO NOT expect numpy arrays.
        # image = to_numpy_array(image)
        image = self._preprocess(
            image=image,
            do_reduce_labels=False,
            do_resize=do_resize,
            size=size,
            resample=resample,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
        )
        # if data_format is not None:
        #     image = to_channel_dimension_format(image, data_format)
        return image
    
    def normalize(
        self, image, mean, std, **kw_ignored) -> torch.tensor:
        """
        Normalize an image. image = (image - image_mean) / image_std.

        Args:
            image (`np.ndarray`):
                Image to normalize.
            image_mean (`float` or `List[float]`):
                Image mean.
            image_std (`float` or `List[float]`):
                Image standard deviation.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
        """

        from typing import Iterable
            
        # input_data_format = infer_channel_dimension_format(image)
        # channel_axis = get_channel_dimension_axis(image)
        input_data_format = 'FIRST'
        if len(image.shape) == 3:
            channel_axis = 0
        else:
            assert len(image.shape) == 4
            channel_axis = 1
        num_channels = image.shape[channel_axis]

        if isinstance(mean, Iterable):
            if len(mean) != num_channels:
                raise ValueError(f"mean must have {num_channels} elements if it is an iterable, got {len(mean)}")
        else:
            mean = [mean] * num_channels
        mean = torch.tensor(mean, dtype=image.dtype).to(image.device) # Is this bad for performance? It's just 3 numbers.

        if isinstance(std, Iterable):
            if len(std) != num_channels:
                raise ValueError(f"std must have {num_channels} elements if it is an iterable, got {len(std)}")
        else:
            std = [std] * num_channels
        std = torch.tensor(std, dtype=image.dtype).to(image.device)

        # Add a batch dim if needed.
        if len(image.shape) == 3:
            im_for_op = image.unsqueeze(0)
        else:
            im_for_op = image

        if input_data_format == 'LAST':
            pass

        else:
            # Put the channel dim last.
            im_for_op = im_for_op.permute(0, 2, 3, 1)

        # Do the op -- channel needs to be last for the broadcasting to work.
        im_for_op = (im_for_op - mean) / std
            
        if input_data_format != 'LAST':
            # Put the channel dim back where it was.
            im_for_op = im_for_op.permute(0, 3, 1, 2)

        # Remove the batch dim if needed.
        if len(image.shape) == 3:
            im_for_op = im_for_op.squeeze(0)
            
        return im_for_op
        
    def preprocess(
        self, images, segmentation_maps=None, do_resize=None, size=None, resample='bilinear', do_rescale=None,
        rescale_factor=None, do_normalize=None, image_mean=None, image_std=None, do_reduce_labels=None,
        **kw_ignored) -> torch.tensor:
        
        do_resize = do_resize if do_resize is not None else self.do_resize
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        do_reduce_labels = do_reduce_labels if do_reduce_labels is not None else self.do_reduce_labels
        resample = resample if resample is not None else self.resample
        size = size if size is not None else self.size
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std

        # images = make_list_of_images(images)
        # if segmentation_maps is not None:
        #     segmentation_maps = make_list_of_images(segmentation_maps, expected_ndims=2)

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        if segmentation_maps is not None and not valid_images(segmentation_maps):
            raise ValueError(
                "Invalid segmentation map type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        if do_resize and size is None or resample is None:
            raise ValueError("Size and resample must be specified if do_resize is True.")

        if do_rescale and rescale_factor is None:
            raise ValueError("Rescale factor must be specified if do_rescale is True.")

        if do_normalize and (image_mean is None or image_std is None):
            raise ValueError("Image mean and std must be specified if do_normalize is True.")

        # images = [
        #     self._preprocess_image(
        #         image=img,
        #         do_resize=do_resize,
        #         resample=resample,
        #         size=size,
        #         do_rescale=do_rescale,
        #         rescale_factor=rescale_factor,
        #         do_normalize=do_normalize,
        #         image_mean=image_mean,
        #         image_std=image_std,
        #     )
        #     for img in images
        # ]
        images = self._preprocess_image(
            image=images,
            do_resize=do_resize,
            resample=resample,
            size=size,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
        )

        data = {"pixel_values": images}

        if segmentation_maps is not None:
            segmentation_maps = [
                self._preprocess_mask(
                    segmentation_map=segmentation_map,
                    do_reduce_labels=do_reduce_labels,
                    do_resize=do_resize,
                    size=size,
                )
                for segmentation_map in segmentation_maps
            ]
            data["labels"] = segmentation_maps

        return BatchFeature(data=data, tensor_type='pt')


def show_debug(arr):
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    if str(arr.dtype).startswith('float'):
        if arr.max() <= 1:
            arr = arr * 255
        arr = arr.astype('uint8')
    if len(arr.shape) == 4:
        arr = arr[0]
    if np.argmin(arr.shape) == 0:
        arr = arr.transpose(1, 2, 0)
    windowarr_pil = Image.fromarray(arr)
    windowarr_pil.show()


def numpicize(arr, copy=None):
    was_tensor = isinstance(arr, torch.Tensor)
    if was_tensor:
        if arr.dtype == torch.float32:
            assert bool(arr.max() <= 1.0)
            arr = (arr * 255).to(torch.uint8)
        arr = arr.detach().cpu().numpy()
    if copy or copy is None and was_tensor:
        arr = arr.copy(order='C') # instead of np.copy
    return arr


class DepthGetter:

    def __init__(self, vehicle_settings='valtra8750', sshot_method='dxcam', 
                 CUDAmodels=True, semantic_annotation_method='fast',
                 input_alpha=0.5,
                 depth_alpha=0.6,
                 semantic_alpha=0.5,
                 ):
        self.semantic_annotation_method = semantic_annotation_method
        self.input_alpha = input_alpha
        self.depth_alpha = depth_alpha
        self.semantic_alpha = semantic_alpha
        self.depther_kw = dict(
            model='vinvino02/glpn-nyu', # 0.5s
            # model='vinvino02/glpn-kitti', # 0.4s, ok results
            # model='ChristianOrr/madnet_keras',  # doesn't work (missing config.json)
            # model='Sohaib36/MonoScene', # doesn't work (not a valid model identifier??)
            # model='Intel/dpt-large', # 5s; good, but reversed direction??
            # model='hf-tiny-model-private/tiny-random-DPTForDepthEstimation', #.03 s, bad results
            device='cuda:0' if CUDAmodels else 'cpu',
        )
        if CUDAmodels:
            self.depth_feature_extractor = CUDAGLPNImageProcessor.from_pretrained(self.depther_kw['model'])
        else:
            self.depth_feature_extractor = GLPNImageProcessor.from_pretrained(self.depther_kw['model']) # This extractor inserts a mandatory transformation to CPU numpy, so we won't actually use it.
        self.depth_feature_extractor.do_rescale = False # handled by screenshotter
        self.depther = GLPNForDepthEstimation.from_pretrained(self.depther_kw['model']).to(self.depther_kw['device'])
        
        self.segmenter_kw = dict(
            # model='facebook/mask2former-swin-large-cityscapes-semantic', # 5 seconds
            model='nvidia/segformer-b0-finetuned-ade-512-512', # 1.2 seconds; no scores
            # model='CIDAS/clipseg-rd64-refined', # doesn't work?
            # model='nvidia/segformer-b0-finetuned-cityscapes-512-1024', # 0.8 seconds; no scores 
            # model='nvidia/segformer-b0-finetuned-cityscapes-1024-1024', # 0.9 seconds; no scores; not great results
            # model='nvidia/segformer-b1-finetuned-cityscapes-1024-1024',  # 6s; no scores
            # model='nvidia/segformer-b4-finetuned-cityscapes-1024-1024',  # 5.3; no scores
            # model='nvidia/segformer-b0-finetuned-cityscapes-640-1280',  # 0.9, no scores, not great results
            # model='shi-labs/oneformer_cityscapes_swin_large',  # 64 seconds!
            # model='shi-labs/oneformer_cityscapes_dinat_large',  # requires natten library http://shi-labs.com/natten/, and install doesn't work.
            # model='facebook/mask2former-swin-tiny-cityscapes-instance',  # 3s; scores. no predictions??
            # model='facebook/mask2former-swin-tiny-cityscapes-semantic', # 2.4s, no predictions?
            device='cuda:0' if CUDAmodels else 'cpu',
        )
        self.segmenter_pipeline = pipeline('image-segmentation', **self.segmenter_kw)
        # Ignore warning about deprecated 'reduce_labels' parameter:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            if CUDAmodels:
                self.segmenter_feature_extractor = CUDASegformerImageProcessor.from_pretrained(self.segmenter_kw['model'])
            else:
                self.segmenter_feature_extractor = SegformerImageProcessor.from_pretrained(self.segmenter_kw['model'])
        self.segmenter_feature_extractor.do_rescale = False # handled by screenshotter
        self.segmenter = SegformerForSemanticSegmentation.from_pretrained(self.segmenter_kw['model']).to(self.segmenter_kw['device'])
        self.class_colors = {  # BGR
            'road': (90, 90, 90),
            'sidewalk': (244, 35, 128),
            'water': (240, 0, 0),
            'grass': (20, 240, 20),
            'plant': (10, 120, 10),
            'building': (20, 20, 240),
            'car': (0, 240, 240),
            'sky': (255, 150, 150),
            'person': (240, 0, 240),
            'signboard': (240, 240, 240),
            'pole': (128, 240, 32),
        }

        self.class_synonyms = {
            'flag': 'signboard',
            'hill': 'terrain',
            'escalator': 'stairs',
            'path': 'road',
            'dirt track': 'road',
            'field': 'grass',
            'tree': 'plant',
            'vegetation': 'plant',
            'truck': 'car',
            'van': 'car',
            'rock': 'earth',
            'mountain': 'earth',
            # 'fence': 'wall',
            # 'railing': 'wall',
            'fence': 'building',
            'wall': 'building',
            'rail': 'building',
            'house': 'building',
            'hovel': 'building',
            'door': 'building',
            'window': 'building',
            'bridge': 'building',
            'lake': 'water',
            'river': 'water',
        }

        # Truck windshield:
        if vehicle_settings == 'tesla':
            self.left_fraction = 0.31500000000000006
            self.right_fraction = 0.23499999999999993
            self.top_fraction = 0.12999999999999984
            self.bottom_fraction = 0.5950000000000002

        elif vehicle_settings == 'lawnmower_maximal':
            self.left_fraction = 0.02
            self.right_fraction = 0.01
            self.top_fraction = 0.12 # .16 to avoid notifications, I think
            self.bottom_fraction = 0.23499999999999993

        elif vehicle_settings == 'lawnmower_tight':
            self.left_fraction = 0.23000000000000007
            self.right_fraction = 0.18000000000000002
            self.top_fraction = 0.1899999999999999
            self.bottom_fraction = 0.25499999999999995

        elif vehicle_settings == 'tractor_centered':
            self.left_fraction = 0.32500000000000007 
            self.right_fraction = 0.325
            self.top_fraction = 0.14999999999999986  
            self.bottom_fraction = 0.5950000000000002

        elif vehicle_settings == 'valtra8750':
            self.left_fraction = 0.29500000000000004
            self.right_fraction = 0.26499999999999996
            self.top_fraction = 0.12999999999999984
            self.bottom_fraction = 0.6450000000000002
            self.output_scaling = 0.8552427184466019
        
        elif vehicle_settings == 'full':
            self.left_fraction = 0.01
            self.right_fraction = 0.01
            self.top_fraction = 0.01
            self.bottom_fraction = 0.01

        else:
            assert vehicle_settings == 'mercedes_urban_truck'
            self.left_fraction = 0.3350000000000001
            self.right_fraction = 0.23499999999999993
            self.top_fraction = 0.33
            self.bottom_fraction = 0.305
            
        self.output_scaling = 170
        self.last_call_time = time.time()
        self.scores_available = None
        
        
        self.time_info = OrderedDict()

        # Set up the camera
        self.sshot_method = sshot_method
        if sshot_method == 'dxcam':
            self.camera = dxcam.create()
        else:
            assert sshot_method == 'd3dshot'
            import d3dshot
            self.camera = d3dshot.create(capture_output="pytorch_float") # pytorch_float_gpu, pytorch_float, pil, numpy, pytorch_gpu, pytorch
            # self.camera.display = d.displays[0]

        # # Make a dummy image for testing the feature extractors.
        inp = self.get_window()
        h, w = inp.shape[:2]

        if self.output_scaling > 10:
            # Find the scaling that gets h to this.
            self.output_scaling = self.output_scaling / float(h)

        # # put channels first
        # inp = inp.permute(2, 0, 1).unsqueeze(0)
        # # By default this "resizes" the image to the closest multiple of 32, then "rescales" it from uint8 [0, 255] to float32 [0, 1].
        # depth_features = self.depth_feature_extractor(images=inp)  # Scaling and maybe uint8 range to float32 0-1 conversion happens here.
        # seg_features = self.segmenter_feature_extractor(images=inp)
        # depth_pixel_values = depth_features['pixel_values']
        # seg_pixel_values = seg_features['pixel_values']
        # for ar, name in zip([depth_pixel_values, seg_pixel_values], ['depth', 'seg']):
        #     if isinstance(ar, list):
        #         ar = ar[0]
        #     device = getattr(ar, 'device', 'cpu')
        #     print(name, ar.shape, ar.dtype, device, ar.min(), ar.max())

    @property
    def left_fraction(self):
        return max(min(1.0, self._left_fraction), 0.0)
    @left_fraction.setter
    def left_fraction(self, value):
        self._left_fraction = value

    @property
    def top_fraction(self):
        return max(min(1.0, self._top_fraction), 0.0)
    @top_fraction.setter
    def top_fraction(self, value):
        self._top_fraction = value

    @property
    def right_fraction(self):
        return max(min(1.0, self._right_fraction), 0.0)
    @right_fraction.setter
    def right_fraction(self, value):
        self._right_fraction = value

    @property
    def bottom_fraction(self):
        return max(min(1.0, self._bottom_fraction), 0.0)
    @bottom_fraction.setter
    def bottom_fraction(self, value):
        self._bottom_fraction = value

    @property
    def output_scaling(self):
        return max(self._output_scaling, 0.001)
    @output_scaling.setter
    def output_scaling(self, value):
        self._output_scaling = value

    def get_output_shape(self, h_in, w_in):
        return (int(self.output_scaling * h_in), int(self.output_scaling * w_in))        

    def get_window(self):
        
        if self.sshot_method == 'dxcam':
            frame = None
            nattempts = 0
            while frame is None:
                nattempts += 1
                frame = self.camera.grab() # uint8 numpy array
            frame = frame.astype('float32') / 255.0

        else:
            assert self.sshot_method == 'd3dshot'
            self.camera.capture()  # TODO: use saving thread option instead
            frame = self.camera.screenshot() # float [0, 1]

        if isinstance(frame, np.ndarray):
            frame = torch.from_numpy(frame)

        frame = frame.to(self.depther_kw['device'])

        # Get the bounding box of the window called "Farming Simulator 22".
        def get_window_rect(name):
            def callback(hwnd, extra):
                if win32gui.GetWindowText(hwnd) == name:
                    rect = win32gui.GetWindowRect(hwnd)
                    extra.append(rect)
            extra = []
            win32gui.EnumWindows(callback, extra)
            return extra[0]

        bbox = get_window_rect('Farming Simulator 22')

        # Excerpt that from the framegrab.
        # Any negative numbers in bbox means we go out out of frame.
        # Any larger than the respective size means we go out of frame to the right or bottom.
        bbox_internal = [
            min(max(0, bbox[0]), frame.shape[1]),
            min(max(0, bbox[1]), frame.shape[0]),
            min(max(0, bbox[2]), frame.shape[1]),
            min(max(0, bbox[3]), frame.shape[0]),
        ]
        windowarr = frame[bbox_internal[1]:bbox_internal[3], bbox_internal[0]:bbox_internal[2]]
        bigheight, bigwidth, bigchannels = windowarr.shape

        # Take the middle 3/4 of the framegrab.
        left = int(self.left_fraction * bigwidth)
        right = bigwidth - int(self.right_fraction * bigwidth)
        top = int(self.top_fraction * bigheight)
        bottom = bigheight - int(self.bottom_fraction * bigheight)
        windowarr = windowarr[top:bottom, left:right]

        return windowarr
    
    def get_preds(self, which=('semantic',)):
        time_since_last = time.time() - self.last_call_time
        self.time_info['loop'] = time_since_last
        self.last_call_time = time.time()


        tick = time.time()
        windowarr = self.get_window()
        self.time_info['capture'] = time.time() - tick
        # show_debug(windowarr)

        height, width, channels = windowarr.shape
        height_out, width_out = self.get_output_shape(height, width)

        # # Show the framegrab.
        # cv2.imshow('frame', windowarr)
        # cv2.waitKey(0)

        # It wants the shape to be batch_size, num_channel, height, width.
        # inp = windowarr.transpose(2, 0, 1).reshape(1, 3, height, width)
        if isinstance(windowarr, np.ndarray):
            inp = torch.from_numpy(windowarr).permute(2, 0, 1).unsqueeze(0)
        else:
            # Put channel dimension first, and add batch dimension before that.
            inp = windowarr.permute(2, 0, 1).unsqueeze(0)

            # # Put the models on the same device as the input.
            # self.depther.model = self.depther.model.to(inp.device)
            # self.segmenter.model = self.segmenter.model.to(inp.device)

        # Make sure it's on the right device.
        inp = inp.to(self.depther_kw['device'])


        out = {}

        self.t_capture = time.time() - tick

        if 'depth' in which:
            # Evaluate the depth model on the framegrab.
            # # Convert to torch tensor.
            # inp = torch.from_numpy(inp)
            tick = time.time()
            # No gradients:
            with torch.no_grad():
                depth_features = self.depth_feature_extractor(inp)
                depth_info = self.depther(**depth_features)
            self.time_info['depth'] = time.time() - tick
            # print(depth_info['depth'].shape)

            tick = time.time()

            # This is a tensor (1, height, width) with the depth in meters, float32.
            depth_m_t = depth_info['predicted_depth']

            # Rescale to input size. https://huggingface.co/docs/transformers/tasks/semantic_segmentation#inference
            depth_m = torch.nn.functional.interpolate(
                depth_m_t.unsqueeze(0), size=(height_out, width_out), mode='bilinear', align_corners=False
            ).to('cpu').numpy().squeeze()

            # import matplotlib.pyplot as plt
            # fig, ax = plt.subplots()
            # fig.colorbar(
            #     plt.imshow(depth_m.squeeze(), cmap='winter', origin='upper',
            #         aspect='equal', extent=[0,  depth_m.shape[1], depth_m.shape[0], 0]), # l, r, b, t
            #     ax=ax,
            #     label='Depth [m]',
            # )
            # # ax.set_xticks([])
            # # ax.set_yticks([])
            # fig.tight_layout()
            # depth_model_id = self.depther_kw['model'].replace('/', '_').replace('\\', '_').replace(':', '_')
            # fig.savefig(f'depth_{depth_model_id}.png', dpi=300)
            # plt.show()
            # plt.close(fig)

            # Scale the max to 255 and min to 0,
            # and make it uint8.
            depth_m_u8 = depth_m - depth_m.min()
            depth_m_u8 = depth_m_u8 / depth_m_u8.max()
            depth_m_u8 = depth_m_u8 * 255
            depth_m_u8 = depth_m_u8.astype('uint8').squeeze()

            # Make it 3-channel, from one color to another
            depth_m_u8 = np.stack([
                np.zeros_like(depth_m_u8),
                255-depth_m_u8,
                depth_m_u8,
            ], axis=2)

            furthest = depth_m.max()
            closest = depth_m.min()
            span = furthest - closest
            if span > 0:
                qA = closest + 0.001 * span
                qB = closest + 0.999 * span
                # Find two places closest to the 10th and 90th quantiles of the depth, for labeling purposes.
                near_Apct_i = np.argmin(np.abs(depth_m.flat - qA))
                near_Bpct_i = np.argmin(np.abs(depth_m.flat - qB))

                # Replace qA and qB with the values at those places.
                # print('Quantile A target is', qA, 'meters')
                # print('Quantile B target is', qB, 'meters')
                qA = depth_m.flat[near_Apct_i]
                qB = depth_m.flat[near_Bpct_i]
                # print('Quantile A is', qA, 'meters')
                # print('Quantile B is', qB, 'meters')
                
                # Convert flat index into (r, c) -- get quotient and modulus.
                rows, cols = depth_m_u8.shape[:2]
                _rowsm, colsm = depth_m.shape[:2]
                # rowApred, colApred = near_Apct_i // colsm, near_Apct_i % colsm
                # rowBpred, colBpred = near_Bpct_i // colsm, near_Bpct_i % colsm
                rowApred, colApred = np.unravel_index(near_Apct_i, depth_m.squeeze().shape)
                rowBpred, colBpred = np.unravel_index(near_Bpct_i, depth_m.squeeze().shape)
                margin_r = 15
                margin_c = 60
                addmargin = lambda cr: (max(margin_c, min(cols-margin_c, cr[0])), max(margin_r, min(rows-margin_r, cr[1])))

                def pred_coords_to_img_coords(col_row):
                    col, row = col_row
                    # the two images are not the same shape.
                    # Assume one is a scaled version of the other?
                    pred_rows, pred_cols = depth_m.squeeze().shape[:2]
                    img_rows, img_cols = depth_m_u8.squeeze().shape[:2]
                    # Got to floaty coordinates, and then back to inty.
                    # print(f'{col} * {img_cols} / {pred_cols} = {col*img_cols/pred_cols}')
                    # print(f'{row} * {img_rows} / {pred_rows} = {row*img_rows/pred_rows}')
                    return int(col * img_cols / pred_cols), int(row * img_rows / pred_rows)
                    # Alternately, maybe one is a cropped version of the other (due to edge padding of a FCN).
                color = (
                    0,
                    0,
                    255,
                )
                # cv2.circle(depth_m_u8, pred_coords_to_img_coords((col, row)), 3, color, 1)
                # no, a MARKER_CROSS instead
                c, r = pred_coords_to_img_coords((colApred, rowApred))
                # print('Convert pred coords', (colApred, rowApred), 'to img coords', (c, r), 'for depth', qA, 'meters')
                cv2.drawMarker(depth_m_u8, (c, r), color, cv2.MARKER_CROSS, 10, 2)
                c, r = addmargin((c, r))
                cv2.putText(depth_m_u8, f'{qA:.2f} m', (c, r), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # Draw a fat white cross at 10, 100
                cv2.drawMarker(depth_m_u8, (10, 100), (255, 255, 255), cv2.MARKER_CROSS, 10, 4)
                

                color = (
                    0,
                    255,
                    0,
                )
                c, r = pred_coords_to_img_coords((colBpred, rowBpred))
                # print('Convert pred coords', (colBpred, rowBpred), 'to img coords', (c, r), 'for depth', qB, 'meters')
                cv2.drawMarker(depth_m_u8, (c, r), color, cv2.MARKER_CROSS, 10, 2)
                c, r = addmargin((c, r))
                cv2.putText(depth_m_u8, f'{qB:.2f} m', (c, r), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            out['depth'] = depth_m_u8
            self.time_info['depth_annotations'] = time.time() - tick

        if 'semantic' in which:
            tick = time.time()
            with torch.no_grad():
                sem_features = self.segmenter_feature_extractor(inp)
                semantic_info_cuda = self.segmenter(**sem_features)
            logits = semantic_info_cuda['logits']
            self.time_info['semantic'] = time.time() - tick

            # Get the semantic segmentation annotations.
            tick = time.time()

            # Rescale to input size. https://huggingface.co/docs/transformers/tasks/semantic_segmentation#inference
            # upsampled_logits = torch.nn.functional.interpolate(logits, size=(height_out, width_out), mode="bilinear", align_corners=False)

            # pred_seg = torch.argmax(upsampled_logits, dim=1).squeeze(0).detach().cpu().numpy()

            # inp_pil = Image.fromarray(inp.detach().cpu().numpy().squeeze(0).transpose(1, 2, 0).astype('uint8'))
            # pipe_out = self.segmenter_pipeline(inp_pil)

            if self.semantic_annotation_method == 'fast':
                semantic = logits.argmax(dim=1).squeeze(0).detach().cpu().numpy()
                # Apply a colormap with vmin=0 vmax=150
                # Scale the 0-150 to 0-255.
                nclasses = 150
                semantic = (255 * semantic / nclasses).astype('uint8')
                semantic = cv2.applyColorMap(semantic, cv2.COLORMAP_TURBO)
            else:
                from transformers.utils.generic import ModelOutput
                # First dim of logits needs to match len(target_size) (both 1).
                assert logits.shape[0] == 1
                model_outputs = ModelOutput(logits=logits.to('cpu'), target_size=[(height_out, width_out)],)
                semantic_info = self.segmenter_pipeline.postprocess(model_outputs)


                semantic = np.zeros((height_out, width_out, 3), dtype='uint8')

                # Merge synonyms.
                canonicalize = lambda label: self.class_synonyms.get(label, label)

                merged = {}
                for key in set([
                    canonicalize(item['label'])
                    for item in semantic_info
                    ]):
                    # get all the items with this label.
                    items = [item for item in semantic_info if canonicalize(item['label']) == key]
                    sz = None if items[0]['score'] is None else np.zeros_like(items[0]['score'])
                    merged_item = {
                        'score': sz,
                        'label': key,
                        'mask': np.zeros_like(items[0]['mask']),
                    }
                    for item in items:
                        if sz is not None:
                            merged_item['score'][item['mask']] = item['score'][item['mask']]
                        merged_item['mask'][np.asarray(item['mask'], dtype='bool')] = 1
                    merged_item['mask'] = Image.fromarray(merged_item['mask'])
                    merged[key] = merged_item
                original_semantic_info = semantic_info
                semantic_info = list(merged.values())

                # For each class...
                for item in semantic_info:
                    # if we haven't seen it before, assign it a random color.
                    class_label = item['label']
                    if class_label not in self.class_colors:
                        self.class_colors[class_label] = np.random.randint(0, 255, size=3, dtype='uint8')
                
                    # Then, fill in the masked areas with that color.
                    rows, cols = rows_cols = np.argwhere(item['mask']).T
                    color = np.asarray(self.class_colors[class_label])
                    # Scale the color by the logconfidence.
                    sc = item['score']
                    if sc is None:
                        if self.scores_available is None:
                            # Haven't talked about it yet.
                            print('No scores available.')
                        self.scores_available = False
                        sc = 0.9
                    else:
                        self.scores_available = True
                    logscore = max(np.log(1. - sc), -10)  # from 0 to -10, with -10 being most confident.
                    item['lightness'] = lightness = 1.0 if not self.scores_available else (-logscore/10.) # now from 0 to 1, with 1 being most confident.
                    confident_color = (color * lightness).astype('uint8')
                    semantic[rows, cols, :] = confident_color

                    # Point closest to the centroid.
                    centroid_r = np.mean(rows).astype('int')
                    centroid_c = np.mean(cols).astype('int')
                    iclosest = np.argmin(np.linalg.norm(rows_cols.T - np.array([centroid_r, centroid_c]), axis=1))
                    item['centroid'] = rows[iclosest], cols[iclosest]

                    uint8 = lambda f: int(0 if f < 0 else (255 if f > 255 else f))
                    item['confident_label_color'] = (uint8(255 * lightness), uint8(255 * lightness), uint8(255 * lightness))

                # Loop again to draw the labels on top.
                for item in semantic_info:
                    rowApred, colApred = item['centroid']
                    light = item['lightness']
                    confident_label_color = item['confident_label_color']
                    cv2.putText(
                        semantic, 
                        item['label'], 
                        (colApred, rowApred), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, # font size
                        confident_label_color,  # color
                        1 if not self.scores_available else (2 if light > 0.9 else 1) # thickness
                    )

            semantic = cv2.cvtColor(semantic, cv2.COLOR_RGB2BGR)

            out['semantic'] = semantic
            self.time_info['semantic_annotations'] = time.time() - tick
            
        # Scale down the outputs.
        tick = time.time()
        # Optionally scale down the output baseimage.
        if self.output_scaling == 1:
            scaled_input = numpicize(windowarr)
        else:
            def rescale(arr):
                arr_in = arr
                if isinstance(arr, np.ndarray):
                    return cv2.resize(np.asarray(arr), (width_out, height_out), interpolation=cv2.INTER_LINEAR)
                else:
                    assert isinstance(arr, torch.Tensor)
                    # Add batch dim if needed.
                    if len(arr.shape) == 3:
                        arr = arr.unsqueeze(0)

                    # Put channel dim first.
                    arr = arr.permute(0, 3, 1, 2)

                    # Scale.
                    arr = torch.nn.functional.interpolate(arr, size=(height_out, width_out), mode='bilinear', align_corners=False)

                    # Put channel dim last.
                    arr = arr.permute(0, 2, 3, 1)

                    # Remove batch dim if needed.
                    if len(arr_in.shape) == 3:
                        arr = arr.squeeze(0)

                    return arr


                
            scaled_input = numpicize(rescale(windowarr))
            for key in which:
                out[key] = rescale(out[key])
        self.time_info['annotation_scaling'] = time.time() - tick

        # Write time-since-last to the framegrab.
        out_to_display = scaled_input.copy()
        row_offset = 19
        for class_label, elapsed in self.time_info.items():
            col_row = (10, row_offset)
            row_offset += 20
            cv2.putText(out_to_display, f'{class_label}: {elapsed:.4f} s', col_row, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        out['input'] = out_to_display

        # Mix the annotated into the original at a reduced alpha.
        for key in out:
            alpha = {
                'input': self.input_alpha,
                'depth': self.depth_alpha,
                'semantic': self.semantic_alpha,
            }.get(key, 0.5)
            A = out[key]
            B = scaled_input# if key != 'input' else numpicize(windowarr)
            out[key] = cv2.addWeighted(A, alpha, B, 1 - alpha, 0)

        return out

    def show_preds(self, which, ignore_errors=False):
        t_last = time.time()
        print('Showing predictions. Press q to stop.')
        while True:
            def break_time():
                preds = self.get_preds(which=which)
                for key in sorted(preds.keys()):
                    rgb = preds[key]
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    cv2.imshow(key, bgr)
                k = cv2.waitKey(50) & 0xFF
                if k == ord('q'):
                    print('Quitting.')
                    return True
            if ignore_errors:
                try:
                    if break_time():
                        break
                except Exception as e:
                    print('Exception:', e)
                    #if it's a KeyboardInterrupt, re-raise it to quit.
                    if isinstance(e, KeyboardInterrupt):
                        raise e
            else:
                if break_time():
                    break
            print('Loop time:', time.time() - t_last, 's')
            t_last = time.time()

    def adjust_geometry(self, ignore_errors=False):
        print('''Q to quit,
        a|d to move left pointer,
        A|D to move right pointer,
        w|s to move top pointer,
        W|S to move bottom pointer,
        R|F to move output scaling,
        p to print current settings.
        ''')
        while True:
            # Get the current frame with current geometry settings.
            def do():
                frame = self.get_window()
                frame = numpicize(frame)

                # Convert from RGB to BGR.
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # Show the framegrab.
                cv2.imshow('input', frame)

                # Show the output scale
                frame_copy = frame.copy()
                scaled_frame = cv2.resize(frame_copy, (0, 0), fx=self.output_scaling, fy=self.output_scaling)

                # Show the output scale
                cv2.imshow('output', scaled_frame)
            if not ignore_errors:
                do()
            else:
                try:
                    do()
                except Exception as e:
                    print('Exception:', e)
                    #if it's a KeyboardInterrupt, re-raise it to quit.
                    if isinstance(e, KeyboardInterrupt):
                        raise e

            # After showing the images, immediately return to the program to ask for inputs.
            cv2.waitKey(1)

            # Ask for changes.
            # def get_val(key):
            #     while True:
            #         existing = getattr(self, key)
            #         msg = f'Enter new {key} (enter for existing={existing}): '
            #         new = input(msg)
            #         if new == '':
            #             val = existing
            #         else:
            #             try:
            #                 val = float(new)
            #             except ValueError:
            #                 print('Invalid input.')
            #                 continue
            #         break
            #     return val
            
            # self.left_fraction = get_val('left_fraction')
            # self.right_fraction = get_val('right_fraction')
            # self.top_fraction = get_val('top_fraction')
            # self.bottom_fraction = get_val('bottom_fraction')
            # self.output_scaling = get_val('output_scaling')

            # Get as key.
            k = cv2.waitKey(0) & 0xFF
            if k == ord('q'):
                print('Quitting.')
                # CLose the two image windows.
                cv2.destroyAllWindows()
                break
            elif k == ord('a'):
                # Decrease left fraction.
                self.left_fraction -= 0.01
            elif k == ord('d'):
                self.left_fraction += 0.01
            elif k == ord('A'):
                self.right_fraction -= 0.01
            elif k == ord('D'):
                self.right_fraction += 0.01
            elif k == ord('w'):
                self.top_fraction -= 0.01
            elif k == ord('s'):
                self.top_fraction += 0.01
            elif k == ord('W'):
                self.bottom_fraction -= 0.01
            elif k == ord('S'):
                self.bottom_fraction += 0.01
            elif k == ord('R'):
                self.output_scaling -= 0.01
            elif k == ord('F'):
                self.output_scaling += 0.01
            elif k == ord('p'):
                print('self.left_fraction =', self.left_fraction)
                print('self.right_fraction =', self.right_fraction)
                print('self.top_fraction =', self.top_fraction)
                print('self.bottom_fraction =', self.bottom_fraction)
                print('self.output_scaling =', self.output_scaling)


def main(do_depth=True, do_semantic=True, **kw_getter):
    # Run in a loop, showing the depth each time we recompute it.
    # Stop with Ctrl-C or q
    dg = DepthGetter(**kw_getter)
    # dg.adjust_geometry()


    which = ()
    if do_depth:
        which += ('depth',)
    if do_semantic:
        which += ('semantic',)
    dg.show_preds(which)


if __name__ == '__main__':
    main(semantic_annotation_method='fast', semantic_alpha=1.0, depth_alpha=1.0)