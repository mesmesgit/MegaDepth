# MES added options 3/14/19
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # so the IDs match nvidia-smi
os.environ["CUDA_VISIBLE_DEVICES"] = "1"   #  "0, 1" for multiple gpus
#
import torch
import sys
from torch.autograd import Variable
import numpy as np
from options.train_options import TrainOptions
opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch
from data.data_loader import CreateDataLoader
from models.models import create_model
from skimage import io
from skimage.transform import resize
# MES additions
from skimage import color
import matplotlib.pyplot as plt
from PIL import Image


# img_path = 'demo.jpg'
img_path = '00024259.jpg'

model = create_model(opt)

input_height = 384
input_width  = 512


def test_simple(model):
    total_loss =0
    toal_count = 0
    print("============================= TEST ============================")
    model.switch_to_eval()

    img = np.float32(io.imread(img_path))/255.0
    # img = np.float32(color.rgba2rgb(io.imread(img_path)))/255.0
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

    # MES change - convert grayscale to RGB
    # pred_inv_depth = color.gray2rgb(pred_inv_depth)
    # Normalize
    output_image = pred_inv_depth.astype(np.float32) # convert to float
    output_image -= output_image.min() # ensure the minimal value is 0.0
    output_image /= output_image.max() # maximum value in image is now 1.0
    # create RGB image, applying colormap
    output_im = Image.fromarray(np.uint8(plt.cm.nipy_spectral(output_image)*255))

    # MES change - output filename
    infile_parts = img_path.split('.')
    output_image_path = infile_parts[0] + '_out_rgb_van.png'

    # MES:  save using PIL
    output_im.save(output_image_path)

    # original image save code
    # io.imsave(output_image_path, pred_inv_depth)
    # print(pred_inv_depth.shape)
    sys.exit()



test_simple(model)
print("We are done")
