import cv2
import math
import os
import tensorflow as tf
import numpy as np


def make_depths(front_depth, back_depth, num_planes):
    front_disparity = 1.0 / front_depth
    back_disparity = 1.0 / back_depth
    disparities = tf.linspace(back_disparity, front_disparity, num_planes)
    return 1.0 / disparities


if os.getcwd()[-3:] == 'BTP':
    os.chdir(os.path.join(os.getcwd(), 'MPI'))

# num_layer = int(input())
num_layer = 32

img = cv2.imread(os.path.join(os.getcwd(), "depth.png"))
input_img = cv2.imread(os.path.join(os.getcwd(), "input.jpeg"))
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# cv2.imshow('depth', hsv)
# cv2.waitKey(0)

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
# print((minVal, maxVal, minLoc, maxLoc))

dist_tensor = make_depths(31, 255, num_layer)
dist = dist_tensor.numpy()

output_dir = os.path.join(os.getcwd(), "Output")
# print(output_dir)

layers = []

for i in range(0, num_layer):
    print(dist[i])
    # upper = (0, 0, math.floor(255 * ((i + 1) / num_layer)))
    if i + 1 < num_layer:
        upper = (0, 0, 255 - round(dist[i + 1]))
    else:
        upper = (0, 0, 255)
    # # lower = (0, 0, math.floor(255 * (i / num_layer)))
    lower = (0, 0, 255 - round(dist[i]))
    mask = cv2.inRange(hsv, lower, upper)
    output_img = cv2.bitwise_and(input_img, input_img, mask=mask)
    output_rgba = cv2.cvtColor(output_img, cv2.COLOR_RGB2RGBA)
    cv2.imwrite(os.path.join(output_dir, "output" +
                str(i) + ".png"), output_rgba)

# output0_rgba = cv2.imread(os.path.join(output_dir, "output" + str(0) + ".png"))
# output0 = cv2.cvtColor(output0_rgba, cv2.COLOR_RGBA2RGB)

# bg = cv2.blur(input_img, (25, 25), 0)

# upper = (0, 0, 255 - dist[1])
# lower = (0, 0, 255 - dist[0])
# mask = cv2.inRange(hsv, lower, upper)

# inverted = cv2.bitwise_not(mask)

# bg_masked = cv2.bitwise_and(bg, bg, mask=inverted)

# # bg_new = cv2.bitwise_xor(bg, output0, mask=output0)
# # output0_new = cv2.bitwise_or(output0, bg)
# # bg_new = cv2.addWeighted(output0, 1, bg, 1, 0)
# bg_new = cv2.add(output0, bg_masked)
# bg_rgba = cv2.cvtColor(bg_new, cv2.COLOR_RGB2RGBA)
# cv2.imwrite(os.path.join(output_dir, "output" + str(0) + ".png"), bg_rgba)

layers = []
for i in range(0, num_layer):
    src = cv2.imread(os.path.join(output_dir, "output" + str(i) + ".png"), 1)
    tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
    b, g, r = cv2.split(src)
    rgba = [b, g, r, alpha]
    dst = cv2.merge(rgba, 4)
    layers.append(dst)
    cv2.imwrite(os.path.join(output_dir, "output" + str(i) + ".png"), dst)

alphas = []
for d in dist:
    a = 1 - math.exp(-1 / d)
    alphas.append(a)
    # print(a)

# for alpha in alphas:
    # alpha = alpha / alphas[num_layer - 1]

# for i in range(0, num_layer):
#     alphas[i] = alphas[i] / alphas[num_layer - 1]

# print(alphas)

for i in range(num_layer - 1, -1, -1):
    # src = cv2.imread(os.path.join(
    #     output_dir, "output" + str(i) + ".png"), 1)
    src = layers[i]
    if i == num_layer - 1:
        pass
    else:
        # prev = cv2.imread(os.path.join(
        #     output_dir, "output" + str(i + 1) + ".png"), 1)
        # alpha = alphas[i + 1]
        # beta = 1
        # dst = cv2.addWeighted(prev, alpha, src, beta, 0.0)
        # cv2.imshow("output" + str(i) + ".png", dst)
        # cv2.waitKey(0)
        dst = src
        # for j in range(i + 1, num_layer):
        #     # prev = cv2.imread(os.path.join(
        #     #     output_dir, "output" + str(j) + ".png"), 1)
        #     prev = layers[j]
        #     # prev = cv2.blur(prev, (25, 25), 0)
        #     alpha = alphas[j]
        #     beta = 1 - alpha
        #     dst = cv2.addWeighted(prev, alpha, dst, beta, 0.0)
        prev = layers[i + 1]
        prev = cv2.blur(prev, (25, 25), 0)
        # alpha = alphas[i + 1]
        # beta = 1 - alpha
        dst = cv2.addWeighted(prev, 0.3, dst, 1, 0.0)
        cv2.imwrite(os.path.join(output_dir, "output" + str(i) + ".png"), dst)
