import tensorflow as tf
import os
import sys

# from functions import multiply_3by4, inverse_3by4, broadcasting_matmul, collapse_dimension, split_dimension, broadcast_to_match
# from graphics import intrinsic_matrix, inverse_intrinsics_matrix, homogenize, dehomogenize, pixel_center_grid


sys.path.append(os.path.abspath("d:/Downloads(D)/BTP/MPI"))

import graphics.graphics as graphics
import functions.functions as mat_fun

def inverse_homography(source_pose, source_intrinsics, target_pose, target_intrinsics, plane):
    target_to_source_pose = mat_fun.multiply_3by4(
        source_pose, mat_fun.inverse_3by4(target_pose))
    rotation, translation = tf.split(target_to_source_pose, [3, 1], axis=-1)
    # print(plane)
    plane_normal = plane[Ellipsis, tf.newaxis, :3]
    plane_offset = plane[Ellipsis, tf.newaxis, 3:]
    denom = mat_fun.broadcasting_matmul(
        plane_normal, translation) + plane_offset
    numer = mat_fun.broadcasting_matmul(
        mat_fun.broadcasting_matmul(-translation, plane_normal), rotation)
    return mat_fun.broadcasting_matmul(
        graphics.intrinsic_matrix(source_intrinsics),
        mat_fun.broadcasting_matmul(
            rotation + tf.divide(numer, denom),
            graphics.inverse_intrinsics_matrix(target_intrinsics)
        )
    )


def apply_homography(homography, coords):
    height = tf.shape(coords)[-3]
    coords = graphics.homogenize(mat_fun.collapse_dimension(coords, -2))
    coords = mat_fun.broadcasting_matmul(coords, homography, transpose_b=True)
    return mat_fun.split_dimension(graphics.dehomogenize(coords), -2, height)


def warp_homography(image, homography, height=None, width=None, clamp=True):
    (image, homography) = mat_fun.broadcast_to_match(
        image, homography, ignore_axes=(3, 2))
    if height is None:
        height = image.shape.as_list()[-3]
    if width is None:
        height = image.shape.as_list()[-2]
    target_coords = graphics.pixel_center_grid(height)
    source_coords = apply_homography(homography, target_coords)
    return graphics.image_sample(image, source_coords, clamp=clamp)
