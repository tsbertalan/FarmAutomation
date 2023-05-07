import os, shutil, glob
# Install by copying folders into mods dir.

DESTDIR = r"C:\\Users\\tsbertalan\\Documents\\My Games\\FarmingSimulator2022\\"
HERE = os.path.dirname(os.path.abspath(__file__))


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

if __name__ == '__main__':
	install()
	