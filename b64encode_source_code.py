import gzip
import base64
from pathlib import Path

# this is base64 encoded source code
file_data = {
    'get_utils.py': '',
    'train.py': '',
    'test.py': '',
    'merge.py': '',
    'config.json': '',
    'siim_loader.py': '',
    'resnet.py': '',
    'unet.py': '',
    'bisenet.py': '',
    'models_utils.py': '',
    'losses.py': '',
    'lovasz_losses.py': '',
    'radam.py': '',
    'scheduler.py': '',
    'metrics.py': '',
    'misc_utils.py': '',
}

for path, encoded in sorted(file_data.items()):
    ##print(path)
    file = open(path, 'rb')
    file_content = file.read()
    base64_content = base64.b64encode(file_content)
    print("'{}': {},".format(path, base64_content))
