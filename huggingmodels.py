
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


class DepthGetter:

    def __init__(self, vehicle_settings='tesla'):
        from transformers import pipeline
        self.depther = pipeline('depth-estimation', model='vinvino02/glpn-nyu')
        self.segmenter = pipeline('image-segmentation', model='facebook/mask2former-swin-large-cityscapes-semantic')
        self.class_colors = {}

        # Truck windshield:
        if vehicle_settings == 'tesla':
            self.left_fraction = 0.31500000000000006
            self.right_fraction = 0.23499999999999993
            self.top_fraction = 0.12999999999999984
            self.bottom_fraction = 0.5950000000000002
            self.input_scaling = 0.65
        
        elif vehicle_settings == 'full':
            self.left_fraction = 0.01
            self.right_fraction = 0.01
            self.top_fraction = 0.01
            self.bottom_fraction = 0.01
            self.input_scaling = 0.4

        else:
            assert vehicle_settings == 'mercedesurbantruck'
            self.left_fraction = 0.3350000000000001
            self.right_fraction = 0.23499999999999993
            self.top_fraction = 0.33
            self.bottom_fraction = 0.305
            self.input_scaling = 0.5
            
        self.output_scaling = 1.0
        self.last_call_time = time.time()

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

        windowarr = self.get_window()

        height, width, channels = windowarr.shape

        # # Show the framegrab.
        # cv2.imshow('frame', windowarr)
        # cv2.waitKey(0)

        # It wants the shape to be batch_size, num_channel, height, width.
        # inp = windowarr.transpose(2, 0, 1).reshape(1, 3, height, width)
        # Convert to PIL
        from PIL import Image
        inp = Image.fromarray(windowarr)

        time_since_last = time.time() - self.last_call_time
        self.last_call_time = time.time()

        # Write time-since-last to the framegrab.
        out_to_display = np.copy(windowarr)
        cv2.putText(out_to_display, f'Loop time: {time_since_last:.2f} s', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        out = {'input': out_to_display}

        if 'depth' in which:
            # Evaluate the depth model on the framegrab.
            # # Convert to torch tensor.
            # import torch
            # inp = torch.from_numpy(inp)
            depth_info = self.depther(inp)
            # print(depth_info['depth'].shape)

            # This is a tensor (1, height, width) with the depth in meters, float32.
            depth_m = depth_info['predicted_depth'].numpy()

            # Scale the max to 255 and min to 0,
            # and make it uint8.
            depth_m_u8 = np.asarray(depth_info['depth'])
            # depth_m_u8 = depth_m - depth_m.min()
            # depth_m_u8 = depth_m_u8 / depth_m_u8.max()
            # depth_m_u8 = depth_m_u8 * 255
            # depth_m_u8 = depth_m_u8.astype('uint8').squeeze()

            # Make it 3-channel
            depth_m_u8 = np.stack([depth_m_u8, depth_m_u8, depth_m_u8], axis=2)

            furthest = depth_m.max()
            closest = depth_m.min()
            span = furthest - closest
            if span > 0:
                qA = closest + 0.01 * span
                qB = closest + 0.99 * span
                # Find two places closest to the 10th and 90th quantiles of the depth, for labeling purposes.
                near_Apct_i = np.argmin(np.abs(depth_m.flat - qA))
                near_Bpct_i = np.argmin(np.abs(depth_m.flat - qB))
                
                # Convert flat index into (r, c) -- get quotient and modulus.
                rows, cols = depth_m_u8.shape[:2]
                _rowsm, colsm = depth_m.shape[:2]
                margin_r = 15
                margin_c = 60
                addmargin = lambda cr: (max(margin_c, min(cols-margin_c, cr[0])), max(margin_r, min(rows-margin_r, cr[1])))

                row, col = near_Apct_i // cols, near_Apct_i % cols
                def pred_coords_to_img_coords(col_row):
                    col, row = col_row
                    # the two images are not the same shape.
                    # Assume one is a scaled version of the other?
                    pred_rows, pred_cols = depth_m.squeeze().shape[:2]
                    img_rows, img_cols = depth_m_u8.squeeze().shape[:2]
                    # Got to floaty coordinates, and then back to inty.
                    return int(col * img_cols / pred_cols), int(row * img_rows / pred_rows)
                    # Alternately, maybe one is a cropped version of the other (due to edge padding of a FCN).
                color = 250, 0, 0
                # cv2.circle(depth_m_u8, pred_coords_to_img_coords((col, row)), 3, color, 1)
                # no, a MARKER_CROSS instead
                c, r = addmargin(pred_coords_to_img_coords((col, row)))
                cv2.drawMarker(depth_m_u8, (c, r), color, cv2.MARKER_CROSS, 10, 1)
                cv2.putText(depth_m_u8, f'{qA:.2f} m', (c, r), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                row, col = near_Bpct_i // colsm, near_Bpct_i % colsm
                color = 0, 120, 0
                c, r = addmargin(pred_coords_to_img_coords((col, row)))
                cv2.drawMarker(depth_m_u8, (c, r), color, cv2.MARKER_CROSS, 10, 1)
                cv2.putText(depth_m_u8, f'{qB:.2f} m', (c, r), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            out['depth'] = depth_m_u8

        if 'semantic' in which:
            semantic_info = self.segmenter(inp)

            # This is a list of dictionaries, each with keys score: float, label: str, and mask: Image.

            semantic = np.zeros((height, width, 3), dtype='uint8')

            # For each class...
            for item in semantic_info:
                # if we haven't seen it before, assign it a random color.
                if item['label'] not in self.class_colors:
                    self.class_colors[item['label']] = np.random.randint(0, 255, size=3, dtype='uint8')
            
                # Then, fill in the masked areas with that color.
                rows, cols = rows_cols = np.argwhere(item['mask']).T
                color = self.class_colors[item['label']]
                # Scale the color by the logconfidence.
                sc = item['score']
                logscore = max(np.log(1. - sc), -10)  # from 0 to -10, with -10 being most confident.
                item['lightness'] = lightness = -logscore/10. # now from 0 to 1, with 1 being most confident.
                confident_color = (color * lightness).astype('uint8')
                semantic[rows, cols, :] = confident_color

                # Point closest to the centroid.
                centroid_r = np.mean(rows).astype('int')
                centroid_c = np.mean(cols).astype('int')
                iclosest = np.argmin(np.linalg.norm(rows_cols.T - np.array([centroid_r, centroid_c]), axis=1))
                item['centroid'] = rows[iclosest], cols[iclosest]

                uint8 = lambda f: int(0 if f < 0 else (255 if f > 255 else f))
                item['confident_label_color'] = (uint8(255 * lightness), uint8(255 * lightness), uint8(255 * lightness), 1)

            # Loop again to draw the labels on top.
            for item in semantic_info:
                row, col = item['centroid']
                light = item['lightness']
                confident_label_color = item['confident_label_color']
                cv2.putText(
                    semantic, 
                    item['label'], 
                    (col, row), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, # font size
                    confident_label_color,  # color
                    2 if light > 0.9 else 1 # thickness
                )

            semantic = cv2.cvtColor(semantic, cv2.COLOR_RGB2BGR)

            out['semantic'] = semantic

            
        # Scale down the outputs.
        if self.output_scaling != 1:
            for key in which:
                out[key] = cv2.resize(np.asarray(out[key]), (0, 0), fx=self.output_scaling, fy=self.output_scaling)

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