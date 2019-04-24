# ss-depth.py
#   script to semantically segment an image using pspnet
#   and then compute a depth image using MegaDepth
#
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # so the IDs match nvidia-smi
os.environ["CUDA_VISIBLE_DEVICES"] = "1"   #  "0, 1" for multiple gpus
import torch
from torch.autograd import Variable
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
from options.train_options import TrainOptions
opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch
from models.models import create_model
from skimage import io
from skimage.transform import resize

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
# out_masked_sky_image = 'docs/image-sky-masked.png'
out_depth_image = 'docs/image-depth-out.png'
#
# ---- perform semantic segmentation using pspnet-pytorch ----
#
#  run semantic segmentation and get the masked image
masked_image = semseg.semseg(pspnetpath, config, image_path, cuda, crf)
# save the masked image
# plt.imsave(out_masked_sky_image, masked_image)
#
# ---- perform depth estimation with MegaDepth ----
#
model = create_model(opt)

input_height = 384
input_width  = 512

total_loss =0
toal_count = 0
# print("============================= TEST ============================")
model.switch_to_eval()

# img = np.float32(io.imread(img_path))/255.0
# img = np.float32(color.rgba2rgb(io.imread(img_path)))/255.0

# MES set img to output of semantic segmentation, masking
img = masked_image

img = resize(img, (input_height, input_width), order = 1)
input_img =  torch.from_numpy( np.transpose(img, (2,0,1)) ).contiguous().float()
input_img = input_img.unsqueeze(0)

input_images = Variable(input_img.cuda() )
pred_log_depth = model.netG.forward(input_images)
pred_log_depth = torch.squeeze(pred_log_depth)

pred_depth = torch.exp(pred_log_depth)

# visualize prediction using inverse depth, so that we don't need sky segmentation (if you want to use RGB map for visualization, \
# you have to run semantic segmentation to mask the sky first since the depth of sky is random from CNN)
pred_inv_depth = 1/pred_depth
pred_inv_depth = pred_inv_depth.data.cpu().numpy()
# you might also use percentile for better visualization
pred_inv_depth = pred_inv_depth/np.amax(pred_inv_depth)

# Normalize
output_image = pred_inv_depth.astype(np.float32) # convert to float
output_image -= output_image.min() # ensure the minimal value is 0.0
output_image /= output_image.max() # maximum value in image is now 1.0
# save the depth image
plt.imsave(out_depth_image, output_image, cmap='nipy_spectral')
