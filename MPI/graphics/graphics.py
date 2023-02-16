
from __future__ import absolute_import
from __future__ import division
from numpy import source

import os
import sys
import tensorflow as tf
import math
from tensorflow_addons import image as tfa_image

# from functions.functions import build_matrix, transform_3by4, flatten_batch

# sys.path.append(os.path.abspath(os.getcwd()))

sys.path.append(os.path.abspath("d:/Downloads(D)/BTP/MPI/functions"))

import functions.functions as functions


def make_depths(front_depth, back_depth, num_planes):
    if (front_depth < back_depth):
        front_disparity = 1.0 / front_depth
        back_disparity = 1.0 / back_depth
        disparities = tf.linspace(back_disparity, front_disparity, num_planes)

    return 1.0 / disparities


def create_planes(depths):
    # initialize normals & depths
    normals = tf.constant([0.0, 0.0, 1.0], shape=[1, 3])
    depths = -depths[Ellipsis, tf.newaxis]  # [..., L, 1]

    # broadcast tensors for concatenation
    normals, depths = functions.broadcast_to_match(
        normals, depths, ignore_axes=1)

    # check shape
    # if (normals.shape.as_list()[:-1] == depths.shape.as_list()[:-1]):
    return tf.concat([normals, depths], axis=-1)  # [..., L, 4]


def culling(pose, target_pose, planes, layers):

    source_to_target_pose = functions.multiply_3by4(
        target_pose, functions.inverse_3by4(pose))

  # Convert planes to target camera space. target_planes is [..., L, 4]
    source_to_target_pose = functions.broadcast_axis(
        source_to_target_pose, ignore_axis=2)
    planes = functions.broadcast_axis(planes, ignore_axis=2)

    # check shape condition
    if (source_to_target_pose.shape.as_list()[:-2] == planes.shape.as_lists()[:-2]):
        inv_pose = functions.inverse_3by4(source_to_target_pose)

        # make pose from ..3*4 to ..4*4 by adding [0.0 0.0 0.0 1.0] at the end
        pose_shape = inv_pose.shape.as_list()
        new_dimension = tf.constant(
            [0.0, 0.0, 0.0, 1.0], shape=len(pose_shape[:-2]) * [1] + [1, 4])
        new_dimension = tf.tile(new_dimension, pose_shape[:-2] + [1, 1])
        inv_pose_4by4 = tf.concat([inv_pose, new_dimension], axis=-2)

        # ..3*4 by ..4*4
        target_planes = tf.matmul(planes, inv_pose_4by4)

        # Fourth coordinate of plane is negative distance in front of the camera.
        # target_visible is [..., L]
        target_visible = tf.cast(
            target_planes[Ellipsis, -1] < 0.0, dtype=tf.float32)

        # per_layer_alpha is [..., L, 1, 1, 1]
        per_layer_alpha = target_visible[Ellipsis,
                                         tf.newaxis, tf.newaxis, tf.newaxis]

        # Multiply alpha channel by per_layer_alpha:
        non_alpha_channels = layers[Ellipsis, :-1]
        alpha = layers[Ellipsis, -1:] * per_layer_alpha

        return tf.concat([non_alpha_channels, alpha], axis=-1)


def position_to_6degree(vec):
    shape = vec.shape.as_list()
    extra_dims = shape[:-1]
    translation, rx, ry, rz = tf.split(vec, [3, 1, 1, 1], -1)
    rx = tf.squeeze(tf.clip_by_value(rx, -math.pi, math.pi), axis=-1)
    ry = tf.squeeze(tf.clip_by_value(ry, -math.pi, math.pi), axis=-1)
    rz = tf.squeeze(tf.clip_by_value(rz, -math.pi, math.pi), axis=-1)
    cos_x = tf.cos(rx)
    sin_x = tf.sin(rx)
    cos_y = tf.cos(ry)
    sin_y = tf.sin(ry)
    cos_z = tf.cos(rz)
    sin_z = tf.sin(rz)
    zero = tf.zeros(extra_dims)
    one = tf.ones(extra_dims)
    rotate_z = build_matrix(
        [[cos_z, -sin_z, zero], [sin_z, cos_z, zero], [zero, zero, one]])
    rotate_y = build_matrix(
        [[cos_y, zero, sin_y], [zero, one, zero], [-sin_y, zero, cos_y]])
    rotate_x = build_matrix(
        [[one, zero, zero], [zero, cos_x, -sin_x], [zero, sin_x, cos_x]])
    rotation = tf.matmul(tf.matmul(rotate_x, rotate_y), rotate_z)
    pose = tf.concat([rotation, translation[Ellipsis, tf.newaxis]], axis=-1)
    return pose


def intrinsic_matrix(intrinsics):
    fx = intrinsics[Ellipsis, 0]
    fy = intrinsics[Ellipsis, 1]
    cx = intrinsics[Ellipsis, 2]
    cy = intrinsics[Ellipsis, 3]
    zero = tf.zeros_like(fx)
    one = tf.ones_like(fx)
    return build_matrix([[fx, zero, cx], [zero, fy, cy], [zero, zero, one]])


def inverse_intrinsics_matrix(intrinsics):
    fxi = 1.0 / intrinsics[Ellipsis, 0]
    fyi = 1.0 / intrinsics[Ellipsis, 1]
    cx = intrinsics[Ellipsis, 2]
    cy = intrinsics[Ellipsis, 3]
    zero = tf.zeros_like(cx)
    one = tf.ones_like(cx)
    return build_matrix([[fxi, zero, -cx * fxi], [zero, fyi, -cy * fyi], [zero, zero, one]])


def homogenize(coords):
    ones = tf.ones_like(coords[Ellipsis, :1])
    return tf.concat([coords, ones], axis=-1)


def dehomogenize(coords):
    return tf.math.divide_no_nan(coords[Ellipsis, :-1], coords[Ellipsis, -1:])


def camera_coord_from_texture(coords, intrinsics):
    focal_length, optical_center = tf.split(intrinsics, [2, 2], axis=-1)
    xy_coords = (coords - optical_center) / focal_length
    return homogenize(xy_coords)


def texture_coords_from_camera(coords, intrinsics):
    xy_coords = tf.math.divide_no_nan(
        coords[Ellipsis, :2], coords[Ellipsis, 2:])
    focal_length, optical_center = tf.split(intrinsics, [2, 2], axis=-1)
    xy_coords = (xy_coords * focal_length) + optical_center
    return xy_coords


def get_camera_relative_points(indices, point, pose):
    point_shape = point.shape.as_list()
    assert (point_shape is not None and len(
        point_shape) and point_shape[0] is not None)
    batch_size = point_shape[0]
    coordinates = []
    for item in range(batch_size):
        coordinates.append(tf.gather(point[item], indices[item]))
    extracted_points = tf.stack(coordinates)
    return transform_3by4(pose, extracted_points)


def pixel_center_grid(height, width):
    height_f = tf.cast(height, dtype=tf.float32)
    width_f = tf.cast(width, dtype=tf.float32)
    ys = tf.linspace(0.5 / height_f, 1.0 - 0.5 / height_f, height)
    xs = tf.linspace(0.5 / width_f, 1.0 - 0.5 / width_f, width)
    xs, ys = tf.meshgrid(xs, ys)
    grid = tf.stack([xs, ys], axis=-1)
    return grid


def camera_rays(intrinsics, height, width):
    coords = pixel_center_grid(height, width)
    intrinsics = intrinsics[Ellipsis, tf.newaxis, tf.newaxis, :]
    rays = camera_coord_from_texture(coords, intrinsics)
    return rays


def clip_texture_coords(coords, height, width):
    min_x = 0.5 / width
    min_y = 0.5 / height
    max_x = 1.0 - min_x
    max_y = 1.0 - min_y
    return tf.clip_by_value(coords, [min_x, min_y], [max_x, max_y])


def image_sample(image, coords, clamp=True):
    tfshape = tf.shape(image)[-3:-1]
    height = tf.cast(tfshape[0], dtype=tf.float32)
    width = tf.cast(tf.shape[1], dtype=tf.float32)
    if clamp:
        coords = clip_texture_coords(coords, height, width)
    pixel_coords = coords * [width, height] - 0.5
    batch_dims = len(image.shape.as_list()) - 3
    assert (image.shape.as_list()[:batch_dims] ==
            pixel_coords.shape.as_list()[:batch_dims])
    batched_image, _ = flatten_batch(image, batch_dims)
    batched_coords, unflatten_coords = flatten_batch(pixel_coords, batch_dims)
    resampled = tfa_image.resampler(batched_image, batched_coords)
    resampled = unflatten_coords(resampled)
    return resampled


def layer_visibility(alphas):
    return tf.math.cumprod(1.0 - alphas, axis=-4, exclusive=True, reverse=True)


def layer_weights(alphas):
    return alphas * layer_visibility(alphas)


def compose_back_to_front(images):
    weights = layer_weights(images[Ellipsis, -1:])
    return tf.reduce_sum(images[Ellipsis, :-1] * weights, axis=-4)


def layers_to_disparity(layers, depths):
    disparities = 1.0 / depths
    disparities = disparities[Ellipsis, tf.newaxis, tf.newaxis, tf.newaxis]
    weights = layer_weights(layers[Ellipsis, -1:])
    return tf.reduce_sum(disparities * weights, axis=-4)



