# ssimage.py
#   script to semantically segment an image using pspnet
#
import os
import numpy as np
# MES change to deal with headless
import matplotlib as mpl
mpl.use('Agg')
#
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../pspnet-pytorch/')
import semseg
#
# set the input image filename
# set the output image filenames
platform_rootpath = '/home/dockeruser/Documents/AVCES/'
pspnetpath = os.path.join(platform_rootpath, 'pspnet-pytorch/')
config = os.path.join(pspnetpath, 'config/ade20k.yaml')
image_path = os.path.join(platform_rootpath, 'MegaDepth/docs/vertical-street.jpg')
cuda = True
crf = True
# out_class_figure = 'docs/demo_out.png'
out_masked_sky_image = 'docs/image-sky-masked.png'
# run semantic segmentation and get the masked image
masked_image = semseg.semseg(pspnetpath, config, image_path, cuda, crf)
# save the masked image
plt.imsave(out_masked_sky_image, masked_image)
#
