import cv2
import numpy as np
import os
import time

# Reconstruction parameters, Unit:mm
depth_start = 30
depth_end = 200
depth_step = 1
pitch = 2
sensor_sizex = 36
focal_length = 35
lens_x = 5
lens_y = 5
# Directory
indir = './dataset/saii/'
outdir = indir + 'VCR/'
try:
    os.mkdir(outdir)
except:
    print('done')

# Load elemental images
ref = cv2.imread(indir + os.listdir(indir)[5])
[r, c, d] = ref.shape
sensor_sizey = sensor_sizex/(c/r)
focal_length *= 36/sensor_sizex # Focal length using full-frame sensor
elemental_img = np.zeros((r, c, d, lens_x * lens_y), 'uint8')
for x in range(0, lens_x):
    for y in range(0, lens_y):
        elemental_img[:, :, :, x + y * lens_y] = cv2.imread(indir + str(x) + '_' + str(y) + '.png')
# Reconstruction
s = time.time()
overlap_img = np.ones((r, c, d), 'uint16')
for depth in range(depth_start, depth_end + depth_step, depth_step):
    print(depth)
    shift_x = np.uint16(np.round((c * pitch * focal_length) / (sensor_sizex * depth)))
    shift_y = np.uint16(np.round((r * pitch * focal_length) / (sensor_sizey * depth)))
    reconstructed_img = np.zeros((r + (lens_y - 1) * shift_y, c + (lens_x - 1) * shift_x, d), 'float')
    intensity_img = np.zeros((r + (lens_y - 1) * shift_y, c + (lens_x - 1) * shift_x, d), 'uint16')
    for x in range(lens_x):
        pointer_x = x * shift_x
        for y in range(lens_y):
            pointer_y = y * shift_y
            reconstructed_img[pointer_y:pointer_y + r, pointer_x:pointer_x + c, :] += elemental_img[:, :, :, x + y * lens_y]
            intensity_img[pointer_y:pointer_y + r, pointer_x:pointer_x + c, :] += overlap_img
    reconstructed_img /= intensity_img
    cv2.imwrite(outdir + str(depth) + 'mm.png', reconstructed_img)
print('evaluation time:' + str(time.time() - s))
