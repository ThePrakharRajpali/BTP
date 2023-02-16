from __future__ import absolute_import
from __future__ import division
from numpy import source

import os
import sys
import tensorflow as tf


def broadcast_to_match(a, b, ignore_axes=0):
    a = tf.convert_to_tensor(a)
    b = tf.convert_to_tensor(b)
    a_shape = a.shape.as_list()
    b_shape = b.shape.as_list()
    if isinstance(ignore_axes, tuple) or isinstance(ignore_axes, list):
        ignore_a = ignore_axes[0]
        ignore_b = ignore_axes[1]
    else:
        ignore_a = ignore_axes
        ignore_b = ignore_axes
    if ignore_a:
        a_shape = a_shape[:-ignore_a]
    if ignore_b:
        b_shape = b_shape[:-ignore_b]
    if a_shape == b_shape:
        return (a, b)
    za = tf.zeros(a_shape + [1] * ignore_b, dtype=b.dtype)
    zb = tf.zeros(b_shape + [1] * ignore_a, dtype=a.dtype)
    a += za
    b += zb
    a_new_shape = a.shape.as_list()
    b_new_shape = b.shape.as_list()
    if ignore_a:
        a_new_shape = a_new_shape[:-ignore_a]
    if ignore_b:
        b_new_shape = b_new_shape[:-ignore_b]
    # assert a_new_shape == b_new_shape
    return (a, b)


def collapse_dimension(tensor, axis):
    tensor = tf.convert_to_tensor(tensor)
    shape = tf.shape(tensor)
    newshape = tf.concat([shape[:axis][:-1], [-1], shape[axis:][1:]], 0)
    return tf.reshape(tensor, newshape)


def split_dimension(tensor, axis, factor):
    tensor = tf.convert_to_tensor(tensor)
    shape = tf.shape(tensor)
    newshape = tf.concat(
        [shape[:axis], [factor, shape[axis] // factor], shape[axis:][1:]], 0)
    return tf.reshape(tensor, newshape)


def flatten_batch(tensor, axes):
    shape = tf.shape(tensor)
    prefix = shape[:axes]
    rest = shape[axes:]
    static_shape = tensor.shape.as_list()
    prod = 1
    for size in static_shape[:axes]:
        if size is None:
            raise ValueError(
                'flatten_batch requires batch dimensions to be statically known.' % static_shape[:axes])
        prod *= size
    output = tf.reshape(tensor, tf.concat([tf.constant([prod]), rest], 0))

    def unflatten(flattened):
        flattened_shape = tf.shape(flattened)
        return tf.reshape(flattened, tf.concat([prefix, flattened_shape[1:]], 0))

    return output, unflatten


def broadcast_axis(a, ignore_axis=0):
    # find shape
    a_shape = a.shape.as_list()

    # define new shape
    if (ignore_axis):
        a_shape = a_shape[:-ignore_axis]

    # add zero tensor to new space
    a = a + tf.zeros(a_shape + [1] * ignore_axis, dtype=a.dtype)

    return a


def check_3by4(tensor):
    shape = tensor.shape.as_list()
    if (shape[-1] != 4 or shape[-2] != 3):
        print("tensor not in 3*4 form")


def multiply_3by4(a, b):
    # check whether tensors are 3*4 or not
    check_3by4(a)
    check_3by4(b)

    a = broadcast_axis(a, ignore_axis=2)
    b = broadcast_axis(b, ignore_axis=2)
    shape_a = a.shape.as_list()[:-2]
    shape_b = b.shape.as_list()[:-2]

    if (shape_a == shape_b):
        # Split translation part off from the rest
        a_3by3, a_translate = tf.split(a, [3, 1], axis=-1)
        b_3by3, b_translate = tf.split(b, [3, 1], axis=-1)

        ab_translate = a_translate + tf.matmul(a_3by3, b_translate)
        # Compute parts of the product
        a_by_b = tf.matmul(a_3by3, b_3by3)
        return tf.concat([a_by_b, ab_translate], axis=-1)


def inverse_3by4(a):
    check_3by4(a)
    rest, translation = tf.split(a, [3, 1], axis=-1)
    inverse = tf.linalg.matrix_transpose(rest)
    inverse_translation = -tf.matmul(inverse, translation)
    return tf.concat([inverse, inverse_translation], axis=-1)


def convert_3by4_4by4(matrix):
    shape = matrix.shape.as_list()
    # check_3by4(matrix)
    extra_dims = shape[:-2]
    filler = tf.constant([0.0, 0.0, 0.0, 0.1],
                         shape=len(extra_dims) * [1] + [1, 4])
    filler = tf.tile(filler, extra_dims + [1, 1])
    return tf.concat([matrix, filler], axis=-2)


def convert_3by3_4by4(matrix):
    shape = matrix.shape.as_list()
    extra_dims = shape[:-2]
    zeros = tf.zeros(extra_dims + [3, 1], dtype=matrix.dtype)
    return convert_3by4_4by4(tf.concat([matrix, zeros], axis=-1))


def transform_3by4(m, v):
    (m, v) = broadcast_to_match(m, v, ignore_axes=2)
    rotation = m[Ellipsis, :3]
    translation = m[Ellipsis, 3]
    translation = translation[Ellipsis, tf.newaxis, :]
    return tf.matmul(v, rotation, transpose_b=True) + translation


def transform_3by4_plane(m, p):
    (m, p) = broadcast_to_match(m, p, ignore_axes=2)
    return tf.matmul(p, convert_3by4_4by4(inverse_3by4(m)))


def build_matrix(elements):
    rows = []
    for row in elements:
        rows.append(tf.stack(row, axis=-1))
    return tf.stack(rows)


def broadcasting_matmul(a, b, **kwargs):
    (a, b) = broadcast_to_match(a, b, ignore_axes=2)
    return tf.matmul(a, b, **kwargs)
