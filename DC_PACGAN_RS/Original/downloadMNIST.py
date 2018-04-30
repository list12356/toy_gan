from __future__ import print_function
import os
import sys
import gzip
import json
import shutil
import zipfile
import argparse
import requests
import subprocess
import numpy as np
# from tqdm import tqdm
from six.moves import urllib


data_dir = os.path.join("./data", 'mnist')
if os.path.exists(data_dir):
	print('Found MNIST - skip')
else:
	os.mkdir(data_dir)
url_base = 'http://yann.lecun.com/exdb/mnist/'
file_names = ['train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz']
for file_name in file_names:
	url = (url_base+file_name).format(**locals())
	print(url)
	out_path = os.path.join(data_dir,file_name)
	cmd = ['curl', url, '-o', out_path]
	print('Downloading ', file_name)
	subprocess.call(cmd)
	cmd = ['gzip', '-d', out_path]
	print('Decompressing ', file_name)
	subprocess.call(cmd)



# data_dir = os.path.join("./data", self.dataset_name)

# fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
# loaded = np.fromfile(file=fd, dtype=np.uint8)
# print(len(loaded))