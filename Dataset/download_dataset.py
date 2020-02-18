import urllib.request
import os

print('Beginning file download with urllib2...\n')

url = ['https://zenodo.org/record/2599196/files/foa_dev.z01?download=1',
       'https://zenodo.org/record/2599196/files/foa_dev.z02?download=1',
       'https://zenodo.org/record/2599196/files/foa_dev.zip?download=1',
       'https://zenodo.org/record/2599196/files/LICENSE?download=1',
       'https://zenodo.org/record/2599196/files/metadata_dev.zip?download=1',
       'https://zenodo.org/record/2599196/files/README.html?download=1',
       'https://zenodo.org/record/3377088/files/foa_eval.zip?download=1',
       'https://zenodo.org/record/3377088/files/metadata_eval.zip?download=1']

pre = 'https://zenodo.org/record/2599196/files/'
post = '?download=1'
files = os.listdir('.')
for f in url:
  name = f
  name = name[len(pre):]
  name = name[:-len(post)]
  print(name)
  if name not in files:
    print(f"Downloading: {name}")
    urllib.request.urlretrieve(f, name)
  else:
    print(f"Already downloaded: {name}")

import subprocess
commands = [['zip', '-s', '0', 'foa_dev.zip', '--out', 'unsplit_foa_dev.zip'],
            ['unzip', 'unsplit_foa_dev.zip'],
            ['unzip', 'metadata_dev.zip'],
            ['unzip', 'foa_eval.zip'],
            ['unzip', 'metadata_eval.zip']]

for command in commands:
    correct = subprocess.run(command,
        # Probably don't forget these, too
        check=True)