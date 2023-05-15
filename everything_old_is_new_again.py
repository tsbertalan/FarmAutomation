import argparse, os, shutil, sys, time
from xml.etree import ElementTree

parser = argparse.ArgumentParser(description='Renew old vehicles.')
parser.add_argument('savegame_number', type=int, help='Savegame number to renew.')
args = parser.parse_args()
# args.savegame_number = 1

print('Renewing savegame number {}...'.format(args.savegame_number))

HOME = os.path.expanduser('~')
GAMEDIR = os.path.join(HOME, 'Documents', 'My Games', 'FarmingSimulator2022')
savefolder = os.path.join(GAMEDIR, 'savegame{}'.format(args.savegame_number))
if not os.path.isdir(savefolder):
    print('Savegame folder {} does not exist. Exiting.'.format(savefolder))
    sys.exit(1)

vehicles_xml_path = os.path.join(savefolder, 'vehicles.xml')

# Backup vehicles.xml with timestamp.
timestamp = time.strftime('%Y%m%d-%H%M%S')
shutil.copyfile(vehicles_xml_path, vehicles_xml_path.replace('.xml', '_backup_{}.xml'.format(timestamp)))

# Read and parse vehicles.xml (starts with  <?xml version="1.0" encoding="utf-8" standalone="no" ?> line):
vehicles = ElementTree.parse(vehicles_xml_path).getroot()

# In every <vehicle> tag, change <vehicle ... age="15.000" ... /> 
# or whatever to <vehicle ... age="0.0" ... />.
for vehicle in vehicles.findall('vehicle'):
    current_age = float(vehicle.get('age'))
    if current_age != 0:
        print('Updating vehicle with filename', vehicle.get('filename'))
        print('  age was', current_age)
        vehicle.set('age', '0.0')

# Write vehicles.xml.
with open(vehicles_xml_path, 'w', encoding='utf-8') as f:
    f.write('<?xml version="1.0" encoding="utf-8" standalone="no"?>\n')
    # Convert vehicles from bytes to utf-8 string.
    vehicles = ElementTree.tostring(vehicles, encoding='utf-8').decode('utf-8')
    f.write(vehicles)
