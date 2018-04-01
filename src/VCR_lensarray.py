import cv2
import numpy as np
import os
import time

# Reconstruction parameters, Unit:mm
depth_start = 10
depth_end = 125
depth_step = 1
pitch = 2
sensor_sizex = 36
focal_length = 230
lens_x = 10
lens_y = 10
# Directory
indir = './dataset/lensarray/'
outdir = indir + 'VCR/'
try:
    os.mkdir(outdir)
except:
    print('done')

# Load elemental images
ei = cv2.imread(indir+'elemental_image.png')
[r, c, d] = ei.shape
eix = np.uint16(np.floor(c/lens_x))
eiy = np.uint16(np.floor(r/lens_y))
sensor_sizey = sensor_sizex/(c/r)
focal_length *= 36/sensor_sizex
ei_block=[]
for x in range(lens_x):
    for y in range(lens_y):
        ei_block.append(cv2.flip(ei[y*eiy:(y+1)*eiy,x*eix:(x+1)*eix,:],-1))
# Reconstruction
s = time.time()
overlap_img = np.ones((eiy, eix, d), 'uint16')
for depth in range(depth_start, depth_end + depth_step, depth_step):
    print(depth)
    shift_x = np.uint16(np.round((eix * pitch * focal_length) / (sensor_sizex * depth)))
    shift_y = np.uint16(np.round((eiy * pitch * focal_length) / (sensor_sizey * depth)))
    reconstructed_img = np.zeros((eiy + (lens_y - 1) * shift_y, eix + (lens_x - 1) * shift_x, d), 'float')
    intensity_img = np.zeros((eiy + (lens_y - 1) * shift_y, eix + (lens_x - 1) * shift_x, d), 'uint16')
    count = 0
    for x in range(lens_x):
        pointer_x = x * shift_x
        for y in range(lens_y):
            pointer_y = y * shift_y
            reconstructed_img[pointer_y:pointer_y + eiy, pointer_x:pointer_x + eix, :] += ei_block[count]
            intensity_img[pointer_y:pointer_y + eiy, pointer_x:pointer_x + eix, :] += overlap_img
            count += 1
    reconstructed_img /= intensity_img
    cv2.imwrite(outdir + str(depth) + 'mm.png', reconstructed_img)
print('evaluation time:' + str(time.time() - s))
