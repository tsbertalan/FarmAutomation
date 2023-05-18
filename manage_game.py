import time, os, shutil, glob
from tqdm.auto import tqdm
import win32ui, win32gui, win32con, win32com.client
import numpy as np, cv2, pytesseract, PIL.Image as Image
# Get tesseract binary installer from https://github.com/UB-Mannheim/tesseract/wiki
# Then, add its bin dir to PATH in Control Panel > Edit the system environment variables > Environment Variables > System variables > Path > Edit > New > C:\Program Files\Tesseract-OCR
from ctypes import windll
import pywintypes

import d3dshot

# Install by copying folders into mods dir.

DESTDIR = r"C:\\Users\\tsbertalan\\Documents\\My Games\\FarmingSimulator2022\\"
HERE = os.path.dirname(os.path.abspath(__file__))


def clear_log():
	
	# Clear the log file.
	if clear_log:
		logfile = os.path.join(DESTDIR, 'log.txt')
		print("Clearing", logfile)
		with open(logfile, 'w') as f:
			f.write('')


def install():
	print("Installing...")
	# @mkdir -p $(DESTDIR)\mods\$(MODNAME)
	# @cp -r $(MODNAME)/* $(DESTDIR)/mods/$(MODNAME)
	# @echo "Done."
	for src in ('Courseplay_FS22', 'FS22_AutoDrive', 'FS22_Telemetry'):
		dst = os.path.join(DESTDIR, 'mods', src)
		print("Copying", src, "to", dst)

		# First, delete the destination if it exists already.
		if os.path.exists(dst):
			#  If files are read-only, make them writable first (for git pack files, it seems).
			def del_rw(action, name, exc):
				import stat
				os.chmod(name, stat.S_IWRITE)
				os.remove(name)
			shutil.rmtree(dst, onerror=del_rw)
		
		# Then, copy the source to the destination.
		shutil.copytree(os.path.join(HERE, src), dst)

def get_hwnd():
	return win32gui.FindWindow(None, "Farming Simulator 22")


def wait_for_keypress():
	time.sleep(0.1)


def _keypress(hwnd, keynum):
	win32gui.PostMessage(hwnd, win32con.WM_KEYDOWN, keynum, 0)
	wait_for_keypress()
	win32gui.PostMessage(hwnd, win32con.WM_KEYUP, keynum, 0)
	wait_for_keypress()


KEYNUM_TILDE = 0xC0


SEARCH_BBOXES = {
	# left, top, right, bot
	'start': (52, 760,  85,  780),
	'loading': (25, 720,  95,  740),
	'console_prompt': (22, 280, 120, 350),
}


class GameWatcher:

	def __init__(self) -> None:
		self.d3d = d3dshot.create(capture_output="numpy")
		self.window_handle = get_hwnd()
		self.shell = win32com.client.Dispatch("WScript.Shell")
		self.savegame_id = 1
		
	def type_text(self, cmd):
		self.shell.SendKeys(cmd)

	def keypress(self, keynum):
		_keypress(self.hwnd, keynum)

	@property
	def window_exists(self):
		return bool(get_hwnd())

	@property
	def window_handle_valid(self):
		return win32gui.IsWindow(self.window_handle)

	@property
	def hwnd(self):
		# Check if the window handle is still valid.
		if not self.window_handle_valid:
			self.window_handle = get_hwnd()
		return self.window_handle
	
	def foreground(self):
		"""Foreground the window."""
		hwnd = self.hwnd
		try:
			win32gui.SetForegroundWindow(hwnd)
		except pywintypes.error:
			raise RuntimeError(f"Failed to foreground window {hwnd}.")

	def get_screenshot(self, target_bbox=None, grayscale=True, foreground_first=True):

		if foreground_first:
			self.foreground()
			time.sleep(0.05)

		# Unfortunately the following fast approach just gives a white screenshot:
		# # Get the window's device context.
		# hwndDC = win32gui.GetWindowDC(hwnd)
		# # Create a memory device context.
		# mfcDC  = win32ui.CreateDCFromHandle(hwndDC)
		# # Create a memory bitmap.
		# saveDC = mfcDC.CreateCompatibleDC()
		# saveBitMap = win32ui.CreateBitmap()
		# # Get the window's dimensions.
		# left, top, right, bot = win32gui.GetWindowRect(hwnd)
		# width = right - left
		# height = bot - top
		# # Create a bitmap compatible with the device context.
		# saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
		# # Select the bitmap into the device context.
		# saveDC.SelectObject(saveBitMap)
		# # Copy the device context to the bitmap.
		# result = windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 0)
		# # Convert the bitmap to a numpy array.
		# bmpinfo = saveBitMap.GetInfo()
		# bmpstr = saveBitMap.GetBitmapBits(True)
		# img = np.frombuffer(bmpstr, dtype='uint8')
		# img.shape = (bmpinfo['bmHeight'], bmpinfo['bmWidth'], 4)
		# # Free the device context.
		# win32gui.DeleteObject(saveBitMap.GetHandle())
		# saveDC.DeleteDC()
		# mfcDC.DeleteDC()
		# win32gui.ReleaseDC(hwnd, hwndDC)

		# Instead, use the following (slow) approach:
		
		window_bbox = win32gui.GetWindowRect(self.hwnd)
		img = self.d3d.screenshot(region=window_bbox)

		# Convert the image to grayscale.
		if grayscale:
			img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

		if target_bbox is not None:
			# Crop the image to the target bbox.
			img = crop_to_bbox(img, target_bbox)

		return img

	def make_screenshot_map(self, indicated_bbox=None):
		# Get a full screenshot of the window.
		img = self.get_screenshot(None, grayscale=False)

		# Put grid lines on the image every 100 pixels.
		for x in range(0, img.shape[1], 100):
			img[:, x] = 255
		for y in range(0, img.shape[0], 100):
			img[y, :] = 255

		# Put a red box around the indicated bbox.
		if indicated_bbox is not None:
			# If the image is grayscale, convert it to color.
			if len(img.shape) == 2:
				img_gray = img
				img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
			else:
				img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			
			cropped = crop_to_bbox(img_gray, indicated_bbox)
			chk = pytesseract.image_to_string(cropped)
			print('Text at', indicated_bbox, 'is', chk)

			# Save with crop.
			Image.fromarray(cropped).save('screen_map_crop.png')

			print('Sanitized: "%s"' % (sanitize(chk, alpha=True, numeric=True, extra=' #'),))

			left, top, right, bottom = indicated_bbox
			bbox_color = 255, 0, 0
			img[top, left:right, :] = bbox_color
			img[bottom, left:right, :] = bbox_color
			img[top:bottom, left, :] = bbox_color
			img[top:bottom, right, :] = bbox_color

		# Save the image.
		img = Image.fromarray(img)
		img.save("screen_map.png")

	def get_status(self, single_key=None, keys=('start', 'loading', 'console_prompt')):
		"""Get the status of the game."""
		out = {}
		img_gray = self.get_screenshot(None, grayscale=True)

		if single_key is not None:
			keys = (single_key,)
		
		for key_ in keys:

			check_kw = {
				'console_prompt': {'extra': '#'},
			}.get(key_, {})

			search_str = {
				'console_prompt': '#',
			}.get(key_, key_)

			search_bbox = SEARCH_BBOXES.get(key_, None)

			out[key_] = check_loc(img_gray, search_str, search_bbox, **check_kw)


		if single_key is not None:
			return out[single_key]

		return out

	def restartSavegame(self):
		# Find the open Farming Simulator 22 window, and send it first the tilde/grave key n times, then "restartMySavegame", then enter.
		
		if not self.window_exists:
			print("No Farming Simulator 22 window found.")

		else:
			# Foreground the window.
			self.foreground()

			self.open_console()
			self.clear_opened_console()

			# Type the command.
			self.type_text(f"cpRestartSaveGame {self.savegame_id}")
			wait_for_keypress()

			# Hit enter.
			self.keypress(win32con.VK_RETURN)
			
			print("Restarted savegame.")

			# Now, wait for the window to be loaded, and then push enter.

			# First, wait for the window to disappear.
			max_sleeps = 8
			n_sleeps = 0
			pbar = tqdm(total=max_sleeps, desc="Closing window")
			while self.window_exists:
				time.sleep(1)
				n_sleeps += 1
				pbar.update(1)
				if n_sleeps > max_sleeps:
					print("Game window did not disappear in time.")
					return
			pbar.close()

			# Then, wait for the window to reappear, then get a new hwnd.
			max_sleeps = 16
			n_sleeps = 0
			pbar = tqdm(total=max_sleeps, desc="Opening window")
			while not self.window_exists:
				time.sleep(1)
				n_sleeps += 1
				pbar.update(1)
				if n_sleeps > max_sleeps:
					print("Game window did not reappear in time.")
					return
			pbar.close()
			
			# Now, wait for the game to be ready to start.
			max_sleeps = 100
			n_sleeps = 0
			pbar = tqdm(total=max_sleeps, desc="Loading")
			while True:
				time.sleep(1)
				n_sleeps += 1
				pbar.update(1)
				status = self.get_status(keys=('start', 'loading'))
				if status['loading']:
					continue
				elif status['start']:
					pbar.close()
					self.close_console()
					print('Game is ready to start.')
					break
				elif n_sleeps > max_sleeps:
					pbar.close()
					print("Game did not load in time.")
					return
		
			# Now, hit enter.
			self.keypress(win32con.VK_RETURN)

	def open_console(self):
		# Open the console -- press tilde until console status is True.
		while not self.get_status('console_prompt'):
			self.keypress(KEYNUM_TILDE)
			time.sleep(0.3)

	def clear_opened_console(self):
		# Push backspace at least 10 times.
		for i in range(np.random.randint(10, 30)):
			self.keypress(win32con.VK_BACK)
		self.keypress(win32con.VK_RETURN)
						
	def close_console(self):
		# First, open it.
		self.open_console()

		# Then, close it with one additional tilde.
		self.keypress(KEYNUM_TILDE)
			
def crop_to_bbox(img, bbox):
	return img[bbox[1]:bbox[3], bbox[0]:bbox[2]]


def sanitize(s, alpha=True, ALPHA=True, numeric=False, extra=None):
	import string
	ok = ''
	if alpha:
		ok += string.ascii_letters
	if ALPHA:
		ok += string.ascii_uppercase
	if numeric:
		ok += string.digits
	if extra is not None:
		ok += extra
	out = ''
	for c in s:
		if c in ok:
			out += c
	return out


def check_loc(img_gray, test_str, bbox, lower=True, **kw_sanitize):
	cropped = crop_to_bbox(img_gray, bbox)
	txt = pytesseract.image_to_string(cropped)
	if lower:
		txt = txt.lower()
	# print(f'Found text: "{txt}" -> "{sanitize(txt, **kw_sanitize)}"')
	# print(f'C.v.        "{test_str}" -> "{sanitize(test_str, **kw_sanitize)}"')
	out = sanitize(test_str, **kw_sanitize) in sanitize(txt, **kw_sanitize)
	return out


if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--skip_install', action='store_true')
	parser.add_argument('--restart_game', action='store_true')
	parser.add_argument('--clear_log', action='store_true')
	args = parser.parse_args()

	if not args.skip_install:
		install()

	if args.clear_log:
		clear_log()

	if args.restart_game:
		game = GameWatcher()

		# game.make_screenshot_map(indicated_bbox=SEARCH_BBOXES['console_prompt'])
		# game.foreground()
		# print('Status:', game.get_status())
		
		game.restartSavegame()
