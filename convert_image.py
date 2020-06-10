# -*- coding: utf-8 -*-

import glob
from PIL import Image

list_of_files = glob.glob('./benchmarks/textures/*.tiff')           
for file_name in list_of_files:
    im= Image.open(file_name)
    newname = file_name.replace('.tiff', '.png')
    im.save(newname)
  