import os
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
		os.system(f'xcopy /E /I /Y "{src}" "{dst}"')


if __name__ == '__main__':
	install()
	