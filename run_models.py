
# & $env:MAMBA_EXE install -y matplotlib opencv=4.7.0 pyqt numpy scipy transformers pytorch torchvision torchaudio pytorch-cuda=11.7 -c conda-forge -c huggingface -c pytorch -c nvidia
# pip install d3dshot dxcam Pillow==7.1.2 pywin32

# from transformers import AutoTokenizer, AutoFeatureExtractor
# segmenter_tokenizer = AutoTokenizer.from_pretrained("facebook/mask2former-swin-large-cityscapes-semantic")
# depther_extractor = AutoFeatureExtractor.from_pretrained("Intel/dpt-large")

# from transformers import Mask2FormerForUniversalSegmentation, AutoModelForDepthEstimation
# segmenter = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-cityscapes-semantic")
# depther = AutoModelForDepthEstimation.from_pretrained("Intel/dpt-large")

import cv2, dxcam, win32gui, time, numpy as np

# Ignore warning about `label_ids_to_fuse` unset.
import transformers
transformers.utils.logging.set_verbosity(transformers.utils.logging.ERROR)

# Check whether pytorch can use the GPU.
import torch
print("CUDA available:", torch.cuda.is_available())


class DepthGetter:

    def __init__(self, vehicle_settings='lawnmower_maximal'):
        from transformers import pipeline
        self.depther_kw = dict(
            model='vinvino02/glpn-nyu', # 0.5s
            # model='vinvino02/glpn-kitti', # 0.4s, ok results
            # model='ChristianOrr/madnet_keras',  # doesn't work (missing config.json)
            # model='Sohaib36/MonoScene', # doesn't work (not a valid model identifier??)
            # model='Intel/dpt-large', # 5s; good, but reversed direction??
            # model='hf-tiny-model-private/tiny-random-DPTForDepthEstimation', #.03 s, bad results
        )
        self.depther = pipeline('depth-estimation', **self.depther_kw)
        self.segmenter_kw = dict(
            # model='facebook/mask2former-swin-large-cityscapes-semantic', # 5 seconds
            model='nvidia/segformer-b0-finetuned-ade-512-512' # 1.2 seconds; no scores
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
        )
        self.segmenter = pipeline('image-segmentation', **self.segmenter_kw)
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
            self.input_scaling = 0.65

        elif vehicle_settings == 'lawnmower_maximal':
            self.left_fraction = 0.02
            self.right_fraction = 0.01
            self.top_fraction = 0.12 # .16 to avoid notifications, I think
            self.bottom_fraction = 0.23499999999999993
            self.input_scaling = 0.38

        elif vehicle_settings == 'lawnmower_tight':
            self.left_fraction = 0.23000000000000007
            self.right_fraction = 0.18000000000000002
            self.top_fraction = 0.1899999999999999
            self.bottom_fraction = 0.25499999999999995
            self.input_scaling = 0.5

        elif vehicle_settings == 'tractor_centered':
            self.left_fraction = 0.32500000000000007 
            self.right_fraction = 0.325
            self.top_fraction = 0.14999999999999986  
            self.bottom_fraction = 0.5950000000000002
            self.input_scaling = 0.8
        
        elif vehicle_settings == 'full':
            self.left_fraction = 0.01
            self.right_fraction = 0.01
            self.top_fraction = 0.01
            self.bottom_fraction = 0.01
            self.input_scaling = 0.4

        else:
            assert vehicle_settings == 'mercedes_urban_truck'
            self.left_fraction = 0.3350000000000001
            self.right_fraction = 0.23499999999999993
            self.top_fraction = 0.33
            self.bottom_fraction = 0.305
            self.input_scaling = 0.5
            
        self.output_scaling = 1.0
        self.last_call_time = time.time()
        self.scores_available = None
        
        from collections import OrderedDict
        self.time_info = OrderedDict()

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
    def input_scaling(self):
        return max(self._input_scaling, 0.001)
    @input_scaling.setter
    def input_scaling(self, value):
        self._input_scaling = value

    @property
    def output_scaling(self):
        return max(self._output_scaling, 0.001)
    @output_scaling.setter
    def output_scaling(self, value):
        self._output_scaling = value

    def get_window(self):
        
        # Capture the camera.

        # list the available cameras
        # for i in range(2, 10):
        #     cap = cv2.VideoCapture(i)
        #     if cap.isOpened():
        #         print(i, cap.getBackendName())
        #         # Preview the camera.
        #         while True:
        #             ret, frame = cap.read()
        #             if not ret:
        #                 break
        #             cv2.imshow('frame', frame)
        #             if cv2.waitKey(1) & 0xFF == ord('q'):
        #                 break
        #         cap.release()


        # # Do it with pygame instead.
        # import pygame.camera
        # print(pygame.camera.get_backends())
        # pygame.camera.init(backend='VideoCapture')
        # for cam in pygame.camera.list_cameras():
        #     print(cam)
            
        # Or DXCam
        camera = dxcam.create()
        frame = camera.grab()
        # print(frame.shape)

        # # Or d3dshot
        # import d3dshot
        # d = d3dshot.create(capture_output="numpy")
        # d.display = d.displays[0]
        # d.capture()
        # print(d.screenshot().shape)
        # # program now blocks and never exits ...


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

        # Scale down the input.
        windowarr = cv2.resize(windowarr, (0, 0), fx=self.input_scaling, fy=self.input_scaling)

        return windowarr

    def get_preds(self, which=('semantic',)):

        
        time_since_last = time.time() - self.last_call_time
        self.time_info['loop'] = time_since_last
        self.last_call_time = time.time()


        tick = time.time()
        windowarr = self.get_window()
        self.time_info['capture'] = time.time() - tick

        height, width, channels = windowarr.shape

        # # Show the framegrab.
        # cv2.imshow('frame', windowarr)
        # cv2.waitKey(0)

        # It wants the shape to be batch_size, num_channel, height, width.
        # inp = windowarr.transpose(2, 0, 1).reshape(1, 3, height, width)
        # Convert to PIL
        from PIL import Image
        inp = Image.fromarray(windowarr)


        out = {}

        self.t_capture = time.time() - tick

        if 'depth' in which:
            # Evaluate the depth model on the framegrab.
            # # Convert to torch tensor.
            # import torch
            # inp = torch.from_numpy(inp)
            tick = time.time()
            depth_info = self.depther(inp)
            self.time_info['depth'] = time.time() - tick
            # print(depth_info['depth'].shape)

            tick = time.time()

            # This is a tensor (1, height, width) with the depth in meters, float32.
            depth_m = depth_info['predicted_depth'].numpy().squeeze()

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
            depth_m_u8 = np.asarray(depth_info['depth'])
            # depth_m_u8 = depth_m - depth_m.min()
            # depth_m_u8 = depth_m_u8 / depth_m_u8.max()
            # depth_m_u8 = depth_m_u8 * 255
            # depth_m_u8 = depth_m_u8.astype('uint8').squeeze()

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
            semantic_info = self.segmenter(inp)
            self.time_info['semantic'] = time.time() - tick

            # This is a list of dictionaries, each with keys score: float, label: str, and mask: Image.
            tick = time.time()

            semantic = np.zeros((height, width, 3), dtype='uint8')

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
        if self.output_scaling == 1:
            scaled_input = windowarr
        else:
            rescale = lambda im: cv2.resize(np.asarray(im), (0, 0), fx=self.output_scaling, fy=self.output_scaling)
            scaled_input = rescale(windowarr)
            for key in which:
                out[key] = rescale(out[key])
        self.time_info['annotation_scaling'] = time.time() - tick

        # Write time-since-last to the framegrab.
        out_to_display = np.copy(windowarr)
        row_offset = 19
        for class_label, elapsed in self.time_info.items():
            col_row = (10, row_offset)
            row_offset += 20
            cv2.putText(out_to_display, f'{class_label}: {elapsed:.4f} s', col_row, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        out['input'] = out_to_display

        # Mix the annotated into the original at a reduced alpha.
        for key in out:
            alpha = {
                'input': 0.5,
                'depth': 0.6,
                'semantic': 0.5,
            }.get(key, 0.5)
            A = out[key]
            B = windowarr if key == 'input' else scaled_input
            out[key] = cv2.addWeighted(A, alpha, B, 1 - alpha, 0)

        return out

    def show_preds(self, which, ignore_errors=True):
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

    def adjust_geometry(self):
        print('''Q to quit,
        a|d to move left pointer,
        A|D to move right pointer,
        w|s to move top pointer,
        W|S to move bottom pointer,
        r|f to move input scaling,
        R|F to move output scaling,
        p to print current settings.
        ''')
        while True:
            # Get the current frame with current geometry settings.
            try:
                frame = self.get_window()

                # Convert from RGB to BGR.
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # Show the framegrab.
                cv2.imshow('input', frame)

                # Show the output scale
                frame_copy = frame.copy()
                scaled_frame = cv2.resize(frame_copy, (0, 0), fx=self.output_scaling, fy=self.output_scaling)

                # Show the output scale
                cv2.imshow('output', scaled_frame)
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
            # self.input_scaling = get_val('input_scaling')
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
            elif k == ord('r'):
                self.input_scaling -= 0.01
            elif k == ord('f'):
                self.input_scaling += 0.01
            elif k == ord('R'):
                self.output_scaling -= 0.01
            elif k == ord('F'):
                self.output_scaling += 0.01
            elif k == ord('p'):
                print('self.left_fraction =', self.left_fraction)
                print('self.right_fraction =', self.right_fraction)
                print('self.top_fraction =', self.top_fraction)
                print('self.bottom_fraction =', self.bottom_fraction)
                print('self.input_scaling =', self.input_scaling)
                print('self.output_scaling =', self.output_scaling)
            
def main():
    # Run in a loop, showing the depth each time we recompute it.
    # Stop with Ctrl-C or q
    dg = DepthGetter()
    # dg.adjust_geometry()


    which = (
        'depth',
        'semantic',
    )
    dg.show_preds(which)


if __name__ == '__main__':
    main()