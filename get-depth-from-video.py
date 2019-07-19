# get-depth-from-video.py
#
#   Script to run MegaDepth network and generate depth images for all frames in YouTube video
#
import os
import torch
import torchvision
# import torchvision.transforms as T
import numpy as np
# from PIL import Image, ImageDraw
# import random
import cv2
# import pylab as plt
# import conv_rgba_rgb
import get-depth-from-image
#
#  MAIN PROGRAM
#
def main():
    #
    # Device configuration
    device0 = torch.device('cuda:0')
    device1 = torch.device('cuda:1')
    device2 = torch.device('cpu')
    #
    # determine which computer/platform we are running on
    if (os.name == "posix"):
        os_list = os.uname()
        if (os_list[0] == "Darwin"):
            pf_detected = 'MAC'
        elif (os_list[0] == "Linux"):
            if (os_list[1] == 'en4119351l'):
                pf_detected = 'Quadro'
            else:
                pf_detected = 'Exxact'
    else:
        pf_detected = 'PC'

    # set the root path based on the computer/platform
    #   rootPath is path to directory in which webots/ and imdata/ directories reside
    if (pf_detected == 'MAC'):
        rootPath = '/Users/mes/Documents/ASU-Classes/Research/Ben-Amor/code/'
        device = device2

    elif (pf_detected == 'Quadro'):
        rootPath = '/home/local/ASUAD/mestric1/Documents/AVCES/'
        device = device0

    elif (pf_detected == 'Exxact'):
        rootPath = '/home/dockeruser/Documents/AVCES/'
        # device = device0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        gpu_ids = [0, 1]

    elif (pf_detected == 'PC'):
        # rootPath = 'C:\Users\cesar\Desktop\Furi\'
        print("PC platform detected.  Exiting.")
        exit()
    else:
        print("Computer/Platform not detected.  Exiting.")
        exit()
    #
    # init paths
    run_id = 'yt1_001'
    videoPath = os.path.join(rootPath, "imdata/video/" + run_id + "/")
    video_filename = os.path.join(videoPath, "zz688.mp4")
    #  run in a loop from the video
    cap = cv2.VideoCapture(video_filename)
    count = 1
    while True:
        try:
            #  read a frame from the video
            ret, frame = cap.read()
            # resize the frame to 512 x 384 for MegaDepth
            frame = cv2.resize(frame, (512, 384), interpolation = cv2.INTER_LANCZOS4)
            #  write the frame to file
            rgbPath = os.path.join(videoPath, "Run/rgb{0:06d}.png".format(count))
            cv2.imwrite(rgbPath, frame)
            #  create the depth image
            get-depth-from-image.generate_depth_image(rgbPath)
            #
        except:
            print("End of video file reached.")
            # break out of while loop for processing video frames
            break
        #
        #  increment frame count
        print("frame {} complete.".format(count))
        count += 1
    #  end of while loop for processing video frames
    #
    #  write the runparams file
    run_params_filename = os.path.join(videoPath, "run_params.npz")
    num_frames = count - 1
    print("run_params_filename is: ", run_params_filename)
    print("run_id is: ", run_id)
    print("num_frames is: ", num_frames)
    np.savez(run_params_filename, \
            run_id=run_id, \
            num_frames_saved=num_frames)
#
#  end of main()
#
# protect the code, to avoid joblib.Parallel issues
if __name__ == "__main__":
    main()
#
