from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import math
import matplotlib.pyplot as plt
import cv2
import math

plt.rcParams["figure.figsize"] = (20, 10)

if os.getcwd()[-3:] == 'BTP':
    os.chdir(os.path.join(os.getcwd(), 'MPI'))


def name_scope(target):
    """A decorator to add a tf name scope for a function."""
    name = target.__name__

    def wrapper(*args, **kwargs):
        with tf.name_scope(name):
            return target(*args, **kwargs)
    return wrapper


def broadcast_to_match(a, b, ignore_axes=0):
    """Returns (a', b') which are the inputs broadcast up to have the same shape.
    Suppose you want to apply an operation to tensors a and b but it doesn't
    support broadcasting. As an example maybe we have tensors of these shapes:
      a    [5, 1, 3, 4]
      b [2, 1, 8, 4, 2]
    Considering the last two dimensions as matrices, we may want to multiply
    a by b to get a tensor [2, 5, 8, 3, 2] of (2x3) matrices. However, tf.matmul
    doesn't support this because the outer dimensions don't match. Calling
    tf.matmul(a, b) directly will fail.
    However, the dimensions do match under broadcasting, so we can do the
    multiplication like this:
      a, b = broadcast_to_match(a, b, ignore_axes=2)
      c = tf.matmul(a, b)
    The ignore_axes parameter tells us to ignore the last two dimensions of a
    and b and just make the rest match.
    Args:
      a: Any shape
      b: Any shape
      ignore_axes: If present, broadcasting will not apply to the final this many
        axes. For example, if you are planning to call tf.matmul(a, b) on the
        result, then set ignore_axes=2 because tf.matmul operates on the last two
        axes, only the rest need to match. To ignore a different number of axes
        for inputs a and b, pass a pair of number to ignore_axes.
    Returns:
      a', b': Identical to the two inputs except tiled so that the shapes
          match. See https://www.tensorflow.org/performance/xla/broadcasting.
          If the shapes already match, no tensorflow graph operations are added,
          so this is cheap.
    """
    a = tf.convert_to_tensor(a)
    b = tf.convert_to_tensor(b)
    a_shape = a.shape.as_list()
    b_shape = b.shape.as_list()
    # Extract the part of the shape that is required to match.
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
    # Addition supports broadcasting, so add a tensor of zeroes.
    za = tf.zeros(a_shape + [1] * ignore_b, dtype=b.dtype)
    zb = tf.zeros(b_shape + [1] * ignore_a, dtype=a.dtype)
    a += zb
    b += za

    a_new_shape = a.shape.as_list()
    b_new_shape = b.shape.as_list()
    if ignore_a:
        a_new_shape = a_new_shape[:-ignore_a]
    if ignore_b:
        b_new_shape = b_new_shape[:-ignore_b]
    assert a_new_shape == b_new_shape
    return (a, b)


def collapse_dim(tensor, axis):
    """Collapses one axis of a tensor into the preceding axis.
    This is a fast operation since it just involves reshaping the
    tensor.
    Example:
      a = [[[1,2], [3,4]], [[5,6], [7,8]]]
      collapse_dim(a, -1) = [[1,2,3,4], [5,6,7,8]]
      collapse_dim(a, 1) = [[1,2], [3,4], [5,6], [7,8]]
    Args:
      tensor: a tensor of shape [..., Di-1, Di, ...]
      axis: the axis to collapse, i, in the range (-n, n). The first axis may not
        be collapsed.
    Returns:
      a tensor of shape [..., Di-1 * Di, ...] containing the same values.
    """
    tensor = tf.convert_to_tensor(tensor)
    shape = tf.shape(tensor)
    # We want to extract the parts of the shape that should remain unchanged.
    # Naively one would write shape[:axis-1] or shape[axis+1:] for this, but
    # this will be wrong if, for example, axis is -1. So the safe way is to
    # first slice using [:axis] or [axis:] and then remove an additional element.
    newshape = tf.concat([shape[:axis][:-1], [-1], shape[axis:][1:]], 0)
    return tf.reshape(tensor, newshape)


def split_dim(tensor, axis, factor):
    """Splits a dimension into two dimensions.
    Opposite of collapse_dim.
    Args:
      tensor: an n-dimensional tensor of shape [..., Di, ...]
      axis: the axis to split, i, in the range [-n, n)
      factor: the size of the first of the two resulting axes. Must divide Di.
    Returns:
      an (n+1)-dimensional tensor of shape [..., factor, Di / factor, ...]
      containing the same values as the input tensor.
    """
    tensor = tf.convert_to_tensor(tensor)
    shape = tf.shape(tensor)
    newshape = tf.concat(
        [shape[:axis], [factor, shape[axis] // factor], shape[axis:][1:]], 0)
    return tf.reshape(tensor, newshape)


def flatten_batch(tensor, axes):
    """Reshape a tensor to collapse multiple axes into a single batch axis.
    This is useful when you are working with multiple layers of batching, but you
    need to call functions that assume only one layer of batching, and then
    convert the output back to the shape with multiple layers of batching.
    Args:
      tensor: a tensor of shape [D0, ... Dn-1].
      axes: the number of initial axes i to collapse. i <= n.
    Returns:
      output: A tensor which contains the same values as input, but which has
        shape [P, Di, Di+1, ... Dn-1] where P is the product D0 * D1 * .. Di-1.
        The sizes D0, ... Di-1 must be statically known.
      unflatten: A function which can be applied to any tensor of known shape
        [P, ...] to convert back to shape [D0, D1, ... Di-1, ...].
    Raises:
      ValueError: if you attempt to flatten_batch tensor of insufficiently known
        shape, or unflatten a tensor with incompatible shape.
    """
    tensor = tf.convert_to_tensor(tensor)
    shape = tf.shape(tensor)
    prefix = shape[:axes]
    rest = shape[axes:]
    static_shape = tensor.shape.as_list()
    product = 1
    for size in static_shape[:axes]:
        if size is None:
            raise ValueError(
                'flatten_batch requires batch dimensions to be statically known.' %
                static_shape[:axes])
        product *= size
    output = tf.reshape(tensor, tf.concat([tf.constant([product]), rest], 0))

    def unflatten(flattened):
        flattened_shape = tf.shape(flattened)
        return tf.reshape(flattened, tf.concat([prefix, flattened_shape[1:]], 0))

    return output, unflatten


"""Geometry utilities.
In these functions:
* Shapes are known statically. Exception: functions dealing with
  points lists, whose length is data-dependent.
* Where possible, utility functions operate on the last one or two
  dimensions of their inputs, and will function irrespective of how many
  preceding dimensions are present. Where it makes sense, functions support
  broadcasting on the part of the shape preceding the fixed dimensions.
  This is to allow preceding dimensions to freely be used for batching or
  other purposes.
* Camera poses are representated as 3x4 matrices (consisting of a 3x3 rotation
  matrix and a 3-coordinate translation vector):
    [[ r r r tx ]
     [ r r r ty ]
     [ r r r tz ]]
  The matrix maps a position in world-space into a position relative to the
  camera position. (Conventionally, the camera position has the Z axis pointing
  into the screen and the Y axis pointing down.) Functions to manipulate such
  matrices have names beginning "mat34_".
* Camera intrinsics are represented as a tensor of last dimension 4. The four
  elements are fx, fy (focal length) and cx, cy (principal point). Intrinsics
  are independent of image-size, they are expressed as if the image runs from
  (0,0) to (1,1). So typically cx == cy == 0.5, and for a 90-degree field of
  view, fx == 0.5.
* Points (whether 2D or 3D) are represented using the last axis of a tensor.
  A set of N 3D points would have shape [N, 3].
* Planes in 3D are represented as 4-vectors. A point x is on the plane p exactly
  when p.x == 0.
* We use texture coordinates to represent points in an image. They go from (0,0)
  in the top-left corner of an image to (1,1) in the bottom right. It is
  convenient to work with these coordinates rather than counts of pixels,
  because they are resolution-independent.
This file is organised in the following sections:
  MATRICES, PLANES, POINTS
    – basic 3D geometry operations.
  CAMERAS
    – intrinsics, projection, camera-relative points.
  IMAGES AND SAMPLING
    – bilinear-sampling from images.
  WARPS AND HOMOGRAPHIES
    – plane sweep, homography, flow warping, depth warping.
"""

import math
from tensorflow_addons import image as tfa_image

# ========== MATRICES, PLANES, POINTS ==========


def check_input_shape(name, tensor, axis, value):
    """Utility function for checking tensor shapes."""
    shape = tensor.shape.as_list()
    if shape[axis] != value:
        raise ValueError('Input "%s": dimension %d should be %s. Shape = %s' %
                         (name, axis, value, shape))


def check_input_m34(name, tensor):
    check_input_shape(name, tensor, -1, 4)
    check_input_shape(name, tensor, -2, 3)


def broadcasting_matmul(a, b, **kwargs):
    (a, b) = broadcast_to_match(a, b, ignore_axes=2)
    return tf.matmul(a, b, **kwargs)


def mat34_to_mat44(matrix):
    """Converts 3x4 matrices to 4x4 matrices by adding filler.
    Considering the last two dimensions of the input tensor, where m
    indicates a matrix coefficient and t a matrix coefficient for translation,
    this function does the following:
         [[m, m, m, t],           [[m, m, m, t],
          [m, m, m, t],    ===>    [m, m, m, t],
          [m, m, m, t]]            [m, m, m, t],
                                   [0, 0, 0, 1]]
    Args:
      matrix: [..., 3, 4] matrix
    Returns:
      A [..., 4, 4] tensor with an extra row [0, 0, 0, 1] added to each matrix.
      Dimensions other than that last two are the same as for the input.
    Raises:
      ValueError: if input has incompatible shape.
    """
    shape = matrix.shape.as_list()
    check_input_m34('matrix', matrix)

    extra_dims = shape[:-2]
    filler = tf.constant([0.0, 0.0, 0.0, 1.0],
                         shape=len(extra_dims) * [1] + [1, 4])
    filler = tf.tile(filler, extra_dims + [1, 1])
    return tf.concat([matrix, filler], axis=-2)


def mat33_to_mat44(matrix):
    """Converts 3x3 matrices to 4x4 by adding zero translation and filler.
    Considering the last two dimensions of the input tensor, where m indicates
    a matrix entry, this function does the following:
         [[m, m, m],           [[m, m, m, 0],
          [m, m, m],    ===>    [m, m, m, 0],
          [m, m, m]]            [m, m, m, 0],
                                [0, 0, 0, 1]]
    Args:
      matrix: A [..., 3, 3] tensor.
    Returns:
      A [..., 4, 4] matrix tensor. Dimensions other than the last two are
      the same as for the input matrix.
    Raises:
      ValueError: if input has incompatible shape.
    """
    shape = matrix.shape.as_list()
    check_input_shape('matrix', matrix, -1, 3)
    check_input_shape('matrix', matrix, -2, 3)

    extra_dims = shape[:-2]
    zeros = tf.zeros(extra_dims + [3, 1], dtype=matrix.dtype)
    return mat34_to_mat44(tf.concat([matrix, zeros], axis=-1))


def mat34_product(a, b):
    """Returns the product of a and b, 3x4 matrices.
    Args:
      a: [..., 3, 4] matrix
      b: [..., 3, 4] matrix
    Returns:
      The product ab. The product is computed as if we added an extra row
      [0, 0, 0, 1] to each matrix, multiplied them, and then removed the extra
      row. The shapes of a and b must match, either directly or via
      broadcasting.
    Raises:
      ValueError: if a or b are not 3x4 matrices.
    """
    check_input_m34('a', a)
    check_input_m34('b', b)

    (a, b) = broadcast_to_match(a, b, ignore_axes=2)
    # Split translation part off from the rest
    a33, a_translate = tf.split(a, [3, 1], axis=-1)
    b33, b_translate = tf.split(b, [3, 1], axis=-1)
    # Compute parts of the product
    ab33 = tf.matmul(a33, b33)
    ab_translate = a_translate + tf.matmul(a33, b_translate)
    # Assemble
    return tf.concat([ab33, ab_translate], axis=-1)


def mat34_transform(m, v):
    """Transform a set of 3d points by a 3x4 pose matrix.
    Args:
      m: [..., 3, 4] matrix
      v: [..., N, 3] set of N 3d points.
    Returns:
      The transformed points mv. The transform is computed as if we added an
      extra coefficient with value 1.0 to each point, performed a matrix
      multiplication, and removed the extra coefficient again. The parts of the
      shape indicated by "..." must match, either directly or via broadcasting.
    Raises:
      ValueError: if inputs are the wrong shape.
    """
    check_input_m34('m', m)
    check_input_shape('v', v, -1, 3)
    (m, v) = broadcast_to_match(m, v, ignore_axes=2)
    rotation = m[Ellipsis, :3]
    # See b/116203395 for why I didn't do the next two lines together as
    # translation = m[..., tf.newaxis, :, 3].
    translation = m[Ellipsis, 3]
    # Now shape is [..., 1, 3].
    translation = translation[Ellipsis, tf.newaxis, :]
    # Points are stored as (N * 3) rather than (3 * N), so multiply in reverse
    # rather than transposing them.
    return tf.matmul(v, rotation, transpose_b=True) + translation


def mat34_transform_planes(m, p):
    """Transform a set of 3d planes by a 3x4 pose matrix.
    Args:
      m: [..., 3, 4] matrix, from source space to target space
      p: [..., N, 4] set of N planes in source space.
    Returns:
      The transformed planes p' in target space.
      If point x is on the plane p, then point Mx is on the plane p'. The parts of
      the shape indicated by "..." must match either directly or via broadcasting.
    Raises:
      ValueError: if inputs are the wrong shape.
    """
    check_input_m34('m', m)
    check_input_shape('p', p, -1, 4)
    (m, p) = broadcast_to_match(m, p, ignore_axes=2)

    # If x is on the plane p, then p . x = 0. We want to find p' such that
    # p' . (M x) = 0. Writing T for transpose and i for inverse, this gives us
    # p'T M x = 0, so p'T = pT Mi.
    # Planes are stored as (N * 4) rather than (4 * N), i.e. pT rather than p, so
    # we can use this directly to compute p'T:
    return tf.matmul(p, mat34_to_mat44(mat34_pose_inverse(m)))


def mat34_pose_inverse(matrix):
    """Invert a 3x4 matrix.
    Args:
      matrix: [..., 3, 4] matrix where [..., 3, 3] is a rotation matrix
    Returns:
      The inverse matrix, of the same shape as the input. It is computed as
      if we added an extra row with values [0, 0, 0, 1], inverted the
      matrix, and removed the row again.
    Raises:
      ValueError: if input is not a 3x4 matrix.
    """
    check_input_m34('matrix', matrix)
    rest, translation = tf.split(matrix, [3, 1], axis=-1)
    inverse = tf.linalg.matrix_transpose(rest)
    inverse_translation = -tf.matmul(inverse, translation)
    return tf.concat([inverse, inverse_translation], axis=-1)


def build_matrix(elements):
    """Stacks elements along two axes to make a tensor of matrices.
    Args:
      elements: [n, m] matrix of tensors, each with shape [...].
    Returns:
      [..., n, m] tensor of matrices, resulting from concatenating
        the individual tensors.
    """
    rows = [tf.stack(row_elements, axis=-1) for row_elements in elements]
    return tf.stack(rows, axis=-2)

# TODO: Graphic


def pose_from_6dof(vec):
    """Converts vector containing 6DoF pose parameters to pose matrices.
    Args:
      vec: [..., 6] parameters in the order tx, ty, tz, rx, ry, rz. rx, ry and rz
        are Euler angles in radians. Rotation is first by z, then by y, then by x,
        and translation happens last. Each rotation is counterclockwise about its
        axis.
    Returns:
      rigid world-to-camera transformation matrix [..., 3, 4] corresponding
      to the input. Rotation angles are clamped to +/- π before conversion.
    """
    check_input_shape('vec', vec, -1, 6)
    shape = vec.shape.as_list()
    extra_dims = shape[:-1]

    # Get translation as [..., 3] and rx, ry, rz each as [..., 1].
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

    rotate_z = build_matrix([[cos_z, -sin_z, zero], [sin_z, cos_z, zero],
                             [zero, zero, one]])

    rotate_y = build_matrix([[cos_y, zero, sin_y], [zero, one, zero],
                             [-sin_y, zero, cos_y]])

    rotate_x = build_matrix([[one, zero, zero], [zero, cos_x, -sin_x],
                             [zero, sin_x, cos_x]])

    rotation = tf.matmul(tf.matmul(rotate_x, rotate_y), rotate_z)
    pose = tf.concat([rotation, translation[Ellipsis, tf.newaxis]], axis=-1)
    return pose


# ========== CAMERAS ==========

# TODO
def intrinsics_matrix(intrinsics):
    """Make a matrix mapping camera space to homogeneous texture coords.
    Args:
      intrinsics: [..., 4] intrinsics. Last dimension (fx, fy, cx, cy)
    Returns:
      [..., 3, 3] matrix mapping camera space to image space.
    """
    fx = intrinsics[Ellipsis, 0]
    fy = intrinsics[Ellipsis, 1]
    cx = intrinsics[Ellipsis, 2]
    cy = intrinsics[Ellipsis, 3]
    zero = tf.zeros_like(fx)
    one = tf.ones_like(fx)
    return build_matrix(
        [[fx, zero, cx], [zero, fy, cy], [zero, zero, one]])

# TODO


def inverse_intrinsics_matrix(intrinsics):
    """Return the inverse of the intrinsics matrix..
    Args:
      intrinsics: [..., 4] intrinsics. Last dimension (fx, fy, cx, cy)
    Returns:
      [..., 3, 3] matrix mapping homogeneous texture coords to camera space.
    """
    fxi = 1.0 / intrinsics[Ellipsis, 0]
    fyi = 1.0 / intrinsics[Ellipsis, 1]
    cx = intrinsics[Ellipsis, 2]
    cy = intrinsics[Ellipsis, 3]
    zero = tf.zeros_like(cx)
    one = tf.ones_like(cx)
    return build_matrix(
        [[fxi, zero, -cx * fxi], [zero, fyi, -cy * fyi], [zero, zero, one]])

# TODO


def homogenize(coords):
    """Convert (x, y) to (x, y, 1), or (x, y, z) to (x, y, z, 1)."""
    ones = tf.ones_like(coords[Ellipsis, :1])
    return tf.concat([coords, ones], axis=-1)

# TODO


def dehomogenize(coords):
    """Convert (x, y, w) to (x/w, y/w) or (x, y, z, w) to (x/w, y/w, z/w)."""
    return tf.math.divide_no_nan(coords[Ellipsis, :-1], coords[Ellipsis, -1:])

# TODO


def texture_to_camera_coordinates(coords, intrinsics):
    """Convert texture coordinates to x,y,1 coordinates relative to camera.
    Args:
      coords: [..., 2] texture coordinates
      intrinsics: [..., 4] (resolution-independent) camera intrinsics. Last
        dimension (fx, fy, cx, cy).
    Returns:
      [..., 3] coordinates, transformed by scaling down by image size and
      applying the inverse of the intrinsics. z-coordinates are all 1.
    Raises:
      ValueError: if coords is the wrong shape.
    """
    check_input_shape('coords', coords, -1, 2)

    # Shift to optical center and divide by focal length.
    # (These are element-wise operations on the x and y coords.)
    focal_length, optical_center = tf.split(intrinsics, [2, 2], axis=-1)
    xy_coords = (coords - optical_center) / focal_length
    return homogenize(xy_coords)

# TODO


def camera_to_texture_coordinates(coords, intrinsics):
    """Convert (x,y,z) coordinates relative to camera to texture coordinates.
    Args:
      coords: [..., 3] coordinates
      intrinsics: [..., 4] camera intrinsics. Last dimension (fx, fy, cx, cy)
    Returns:
      [..., 2] coordinates, transformed by dividing by Z, applying camera
      intrinsics and scaling to image size.
    Raises:
      ValueError: if coords is the wrong shape.
    """
    check_input_shape('coords', coords, -1, 3)
    xy_coords = tf.math.divide_no_nan(
        coords[Ellipsis, :2], coords[Ellipsis, 2:])

    # Scale by focal length and shift optical center.
    # (These are element-wise operations on the x and y coords.)
    focal_length, optical_center = tf.split(intrinsics, [2, 2], axis=-1)
    xy_coords = (xy_coords * focal_length) + optical_center
    return xy_coords

# TODO


def get_camera_relative_points(indices, point, pose):
    """Get tensor of camera-relative 3d points in a frame.
    Args:
      indices: [B, P] Indices into point of coordinates to retrieve.
      point: [B, N, 3] A set of N (x,y,z) coordinates per batch item
      pose: [B, 3, 4] Camera pose
    Returns:
      [B, P, 3] Point coordinates corresponding to the indices.
      Specifically result[b, p, :] = point[b, indices[b, p], :].
    """
    # There is no "batched gather" so we either must loop over the batch, or
    # use gather_nd. Looping over the batch is simpler so we'll do that.
    point_shape = point.shape.as_list()
    # Batch size must be statically known
    assert (point_shape is not None and len(point_shape) and
            point_shape[0] is not None)
    batch_size = point_shape[0]

    coordinates = []
    for item in range(batch_size):
        coordinates.append(tf.gather(point[item], indices[item]))
    extracted_points = tf.stack(coordinates)
    # Convert points to be camera-relative.
    return mat34_transform(pose, extracted_points)


# ========== IMAGES AND SAMPLING ==========

# TODO
def pixel_center_grid(height, width):
    """Produce a grid of (x,y) texture-coordinate pairs of pixel centers.
    Args:
      height: (integer) height, not a tensor
      width: (integer) width, not a tensor
    Returns:
      A tensor of shape [height, width, 2] where each entry gives the (x,y)
      texture coordinates of the corresponding pixel center. For example, for
      pixel_center_grid(2, 3) the result is:
         [[[1/6, 1/4], [3/6, 1/4], [5/6, 1/4]],
          [[1/6, 3/4], [3/6, 3/4], [5/6, 3/4]]]
    """
    height_float = tf.cast(height, dtype=tf.float32)
    width_float = tf.cast(width, dtype=tf.float32)
    ys = tf.linspace(0.5 / height_float, 1.0 - 0.5 / height_float, height)
    xs = tf.linspace(0.5 / width_float, 1.0 - 0.5 / width_float, width)
    xs, ys = tf.meshgrid(xs, ys)
    grid = tf.stack([xs, ys], axis=-1)
    assert grid.shape.as_list() == [height, width, 2]
    return grid

# TODO


def camera_rays(intrinsics, height, width):
    """A tensor of rays from the camera to the plane at z=1, one per pixel.
    Args:
      intrinsics: [..., 4] camera intrinsics
      height: output height in pixels
      width: output width in pixels
    Returns:
      [..., H, W, 3] A grid of H x W rays. Each ray is a vector (x, y, 1) in
      camera space. For example, for a pixel at the principal point, the
      corresponding ray is (0, 0, 1).
    """
    coords = pixel_center_grid(height, width)
    intrinsics = intrinsics[Ellipsis, tf.newaxis, tf.newaxis, :]
    rays = texture_to_camera_coordinates(coords, intrinsics)
    return rays

# TODO


def clip_texture_coords_to_corner_pixels(coords, height, width):
    """Clip texture coordinates to the centers of the corner pixels."""
    min_x = 0.5 / width
    min_y = 0.5 / height
    max_x = 1.0 - min_x
    max_y = 1.0 - min_y
    return tf.clip_by_value(coords, [min_x, min_y], [max_x, max_y])

# TODO


def sample_image(image, coords, clamp=True):
    """Sample points from an image, using bilinear filtering.
    Args:
      image: [B0, ..., Bn-1, height, width, channels] image data
      coords: [B0, ..., Bn-1, ..., 2] (x,y) texture coordinates
      clamp: if True, coordinates are clamped to the coordinates of the corner
        pixels -- i.e. minimum value 0.5/width, 0.5/height and maximum value
        1.0-0.5/width or 1.0-0.5/height. This is equivalent to extending the image
        in all directions by copying its edge pixels. If False, sampling values
        outside the image will return 0 values.
    Returns:
      [B0, ..., Bn-1, ..., channels] image data, in which each value is sampled
      with bilinear interpolation from the image at position indicated by the
      (x,y) texture coordinates. The image and coords parameters must have
      matching batch dimensions B0, ..., Bn-1.
    Raises:
      ValueError: if shapes are incompatible.
    """
    check_input_shape('coords', coords, -1, 2)
    tfshape = tf.shape(image)[-3:-1]
    height = tf.cast(tfshape[0], dtype=tf.float32)
    width = tf.cast(tfshape[1], dtype=tf.float32)
    if clamp:
        coords = clip_texture_coords_to_corner_pixels(coords, height, width)

    # Resampler expects coordinates where (0,0) is the center of the top-left
    # pixel and (width-1, height-1) is the center of the bottom-right pixel.
    pixel_coords = coords * [width, height] - 0.5

    # tfa_image.resampler only works with exactly one batch dimension, i.e. it
    # expects image to be [batch, height, width, channels] and pixel_coords to be
    # [batch, ..., 2]. So we need to reshape, perform the resampling, and then
    # reshape back to what we had.
    batch_dims = len(image.shape.as_list()) - 3
    assert (image.shape.as_list()[:batch_dims] == pixel_coords.shape.as_list()
            [:batch_dims])

    batched_image, _ = flatten_batch(image, batch_dims)
    batched_coords, unflatten_coords = flatten_batch(
        pixel_coords, batch_dims)
    resampled = tfa_image.resampler(batched_image, batched_coords)

    # Convert back to the right shape to return
    resampled = unflatten_coords(resampled)
    return resampled


# ========== WARPS AND HOMOGRAPHIES ==========

# TODO : Hompgraphy

def inverse_homography(source_pose, source_intrinsics, target_pose,
                       target_intrinsics, plane):
    """Compute inverse homography from source to target.
    This function computes a matrix H which relates the image of the plane P
    in the source and target cameras by matrix multiplication as follows:
        (source_u, source_v, source_w) = H (target_u, target_v, target_w)
    where (u, v, w) are the homogeneous coordinates of the point in the
    image-spaces of the source and target cameras.
    The plane P is specified as a normal vector (plane[0:3]) in the source
    camera-space plus an offset (plane[3]). A point p in source-camera-space
    is in the plane when (p_x, p_y, p_z, 1) . P == 0.
    Args:
      source_pose: [..., 3, 4] source camera pose
      source_intrinsics: [..., 4] last dimension (fx, fy, cx, cy)
      target_pose: [..., 3, 4] target camera pose
      target_intrinsics: [..., 4] last dimension (fx, fy, cx, cy)
      plane: [..., 4] The plane P.
    Returns:
      [..., 3, 3] Homography matrix H.
    """
    target_to_source_pose = mat34_product(source_pose,
                                          mat34_pose_inverse(target_pose))
    rotation, translation = tf.split(target_to_source_pose, [3, 1], axis=-1)
    plane_normal = plane[Ellipsis, tf.newaxis, :3]
    plane_offset = plane[Ellipsis, tf.newaxis, 3:]

    # Everything now has 2 final dimensions for matrix operations, i.e.
    #   rotation     [..., 3, 3]  from target to source
    #   translation  [..., 3, 1]  from target to source, in source space
    #   plane_normal [..., 1, 3]  in source space
    #   plane_offset [..., 1, 1]  in source space

   # print(plane_normal.shape.as_list())
   # print(plane_normal)
   # print(translation.shape.as_list())
   # print(translation)

   # print(plane_offset)


    denominator = broadcasting_matmul(plane_normal, translation) + plane_offset
    numerator = broadcasting_matmul(
        broadcasting_matmul(-translation, plane_normal), rotation)

    return broadcasting_matmul(
        intrinsics_matrix(source_intrinsics),
        broadcasting_matmul(rotation + tf.divide(numerator, denominator),
                            inverse_intrinsics_matrix(target_intrinsics)))

# TODO : Hompgraphy


def apply_homography(homography, coords):
    """Transform grid of (x,y) texture coordinates by a homography.
    Args:
      homography: [..., 3, 3]
      coords: [..., H, W, 2] (x,y) texture coordinates
    Returns:
      [..., H, W, 2] transformed coordinates.
    """
    height = tf.shape(coords)[-3]
    coords = homogenize(collapse_dim(coords, -2))  # [..., H*W, 3]
    # Instead of transposing the coords, transpose the homography and
    # swap the order of multiplication.
    coords = broadcasting_matmul(coords, homography, transpose_b=True)
    # coords is now [..., H*W, 3]
    return split_dim(dehomogenize(coords), -2, height)

# TODO : Hompgraphy


def homography_warp(image, homography, height=None, width=None, clamp=True):
    """Warp an image according to an inverse homography.
    Args:
      image: [..., H, W, C] input image
      homography: [..., 3, 3] homography mapping output to input
      height: desired output height (or None to use input height)
      width: desired output width (or None to use input width)
      clamp: whether to clamp image coordinates (see sample_image doc)
    Returns:
      [..., height, width, C] warped image.
    """
    (image, homography) = broadcast_to_match(
        image, homography, ignore_axes=(3, 2))
    if height is None:
        height = image.shape.as_list()[-3]
    if width is None:
        width = image.shape.as_list()[-2]

    target_coords = pixel_center_grid(height, width)
    source_coords = apply_homography(homography, target_coords)
    return sample_image(image, source_coords, clamp=clamp)


"""Utilities for working with Multiplane Images (MPIs).
A multiplane image is a set of RGB + alpha textures, positioned as fronto-
parallel planes at specific depths from a reference camera. It represents a
lightfield and can be used to render new views from nearby camera positions
by warping each texture according to its plane homography and combining the
results with an over operation. More detail at:
   https://people.eecs.berkeley.edu/~tinghuiz/projects/mpi/
In this code, an MPI is represented by a tensor of layer textures and a tensor
of depths:
  layers: [..., L, H, W, 4] -- L is the number of layers, last dimension is
          typically RGBA but it can be any number of channels as long as the
          last channel is alpha.
  depths: [..., L] -- distances of the planes from the reference camera.
Layers and depths are stored back-to-front, i.e. farthest layer ("layer 0")
comes first. Typically the depths are chosen so that the corresponding
disparities (inverse depths) form an arithmetic sequence.
"""

# TODO : Graphic


def layer_visibility(alphas):
    """Compute visibility for each pixel in each layer.
    Visibility says how unoccluded each pixel is by the corresponding pixels in
    front of it (i.e. those pixels with the same (x,y) position in subsequent
    layers). The front layer has visibility 1 everywhere since nothing can occlude
    it. Each other layer has visibility equal to the product of (1 - alpha) for
    all the layers in front of it.
    Args:
      alphas: [..., L, H, W, 1] Alpha channels for L layers, back to front.
    Returns:
      [..., L, H, W, 1] visibilities.
    """
    return tf.math.cumprod(
        1.0 - alphas, axis=-4, exclusive=True, reverse=True)

# TODO : Graphic


def layer_weights(alphas):
    """Compute contribution weights for each layer from a set of alpha channels.
    The weights w_i for each layer are determined from the layer alphas so that
    to composite the layers we simple multiply each by its weight and add them
    up. In other words, the weight says how much each layer contributes to the
    final composed image.
    For alpha-blending, the weight of a layer at a point is its visibility at that
    point times its alpha at that point, i.e:
         alpha_i * (1 - alpha_i+1) * (1 - alpha_i+2) * ... (1 - alpha_n-1)
    If the first (i.e. the back) layer has alpha=1 everywhere, then the output
    weights will sum to 1 at each point.
    Args:
       alphas: [..., L, H, W, 1] Alpha channels for L layers, back to front.
    Returns:
       [..., L, H, W, 1] The resulting layer weights.
    """
    return alphas * layer_visibility(alphas)

# TODO : Graphic


def compose_back_to_front(images):
    """Compose a set of images (for example, RGBA), back to front.
    Args:
      images: [..., L, H, W, C+1] Set of L images, with alpha in the last channel.
    Returns:
      [..., H, W, C] Composed image.
    """
    weights = layer_weights(images[Ellipsis, -1:])
    return tf.reduce_sum(images[Ellipsis, :-1] * weights, axis=-4)

# TODO : Graphic


def disparity_from_layers(layers, depths):
    """Compute disparity map from a set of MPI layers.
    From reference view.
    Args:
      layers: [..., L, H, W, C+1] MPI layers, back to front.
      depths: [..., L] depths for each layer.
    Returns:
      [..., H, W, 1] Single-channel disparity map from reference viewpoint.
    """
    disparities = 1.0 / depths
    # Add height, width and channel axes to disparities, so it can broadcast.
    disparities = disparities[Ellipsis, tf.newaxis, tf.newaxis, tf.newaxis]
    weights = layer_weights(layers[Ellipsis, -1:])

    # Weighted sum of per-layer disparities:
    return tf.reduce_sum(disparities * weights, axis=-4)

# TODO : Graphic


def make_depths(front_depth, back_depth, num_planes):
    """Returns a list of MPI plane depths, back to front.
    The first element in the list will be back_depth, and last will be
    near-depth, and in between there will be num_planes intermediate
    depths, which are interpolated linearly in disparity.
    Args:
      front_depth: The depth of the front-most MPI plane.
      back_depth: The depth of the back-most MPI plane.
      num_planes: The total number of planes to create.
    Returns:
      [num_planes] A tensor of depths sorted in descending order (so furthest
      first). This order is useful for back to front compositing.
    """
    assert front_depth < back_depth

    front_disparity = 1.0 / front_depth
    back_disparity = 1.0 / back_depth
    disparities = tf.linspace(back_disparity, front_disparity, num_planes)
    return 1.0 / disparities


def render_layers(layers,
                  depths,
                  pose,
                  intrinsics,
                  target_pose,
                  target_intrinsics,
                  height=None,
                  width=None,
                  clamp=True):
    """Render target layers from MPI representation.
    Args:
      layers: [..., L, H, W, C] MPI layers, back to front.
      depths: [..., L] MPI plane depths, back to front.
      pose: [..., 3, 4] reference camera pose.
      intrinsics: [..., 4] reference intrinsics.
      target_pose: [..., 3, 4] target camera pose.
      target_intrinsics: [..., 4] target intrinsics.
      height: height to render to in pixels (or None for input height).
      width: width to render to in pixels (or None for input width).
      clamp: whether to clamp image coordinates (see geometry.sample_image doc),
        i.e. extending the image beyond its size or not.
    Returns:
      [..., L, height, width, C] The layers warped to the target view by applying
      an appropriate homography to each one.
    """


    source_to_target_pose = mat34_product(
        target_pose, mat34_pose_inverse(pose))
    

    # Add a dimension to correspond to L in the poses and intrinsics.
    pose = pose[Ellipsis, tf.newaxis, :, :]  # [..., 1, 3, 4]
    target_pose = target_pose[Ellipsis, tf.newaxis, :, :]  # [..., 1, 3, 4]
    intrinsics = intrinsics[Ellipsis, tf.newaxis, :]  # [..., 1, 4]
    # [..., 1, 4]
    target_intrinsics = target_intrinsics[Ellipsis, tf.newaxis, :]

    # Fronto-parallel plane equations at the given depths, in the reference
    # camera's frame.
    normals = tf.constant([0.0, 0.0, 1.0], shape=[1, 3])
    depths = -depths[Ellipsis, tf.newaxis]  # [..., L, 1]
    normals, depths = broadcast_to_match(normals, depths, ignore_axes=1)
    planes = tf.concat([normals, depths], axis=-1)  # [..., L, 4]

    homographies = inverse_homography(pose, intrinsics, target_pose,
                                      target_intrinsics,
                                      planes)  # [..., L, 3, 3]
    # Each of the resulting [..., L] homographies knows how to inverse-warp one
    # of the [..., (H,W), L] images into a new [... (H',W')] target images.
    target_layers = homography_warp(
        layers, homographies, height=height, width=width, clamp=clamp)

    # The next few lines implement back-face culling.
    #
    # We don't want to render content that is behind the camera. (If we did, we
    # might see upside-down images of the layers.) A typical graphics approach
    # would be to test each pixel of each layer against a near-plane and discard
    # those that are in front of it. Here we implement something cheaper:
    # back-face culling. If the target camera sees the "back" of a layer then we
    # set that layer's alpha to zero. This is simple and sufficient in practice
    # to avoid nasty artefacts.

    # Convert planes to target camera space. target_planes is [..., L, 4]
    target_planes = mat34_transform_planes(source_to_target_pose, planes)

    # Fourth coordinate of plane is negative distance in front of the camera.
    # target_visible is [..., L]
    target_visible = tf.cast(
        target_planes[Ellipsis, -1] < 0.0, dtype=tf.float32)
    # per_layer_alpha is [..., L, 1, 1, 1]
    per_layer_alpha = target_visible[Ellipsis,
                                     tf.newaxis, tf.newaxis, tf.newaxis]
    # Multiply alpha channel by per_layer_alpha:
    non_alpha_channels = target_layers[Ellipsis, :-1]
    alpha = target_layers[Ellipsis, -1:] * per_layer_alpha

    target_layers = tf.concat([non_alpha_channels, alpha], axis=-1)
    return target_layers


def render(layers,
           depths,
           pose,
           intrinsics,
           target_pose,
           target_intrinsics,
           height=None,
           width=None,
           clamp=True):
    """Render target image from MPI representation.
    Args:
      layers: [..., L, H, W, C+1] MPI layers back to front, alpha in last channel.
      depths: [..., L] MPI plane depths, back to front
      pose: [..., 3, 4] reference camera pose
      intrinsics: [..., 4] reference intrinsics
      target_pose: [..., 3, 4] target camera pose
      target_intrinsics: [..., 4] target intrinsics
      height: height to render to in pixels (or None for input height)
      width: width to render to in pixels (or None for input width)
      clamp: whether to clamp image coordinates (see geometry.sample_image doc).
        i.e. extending the image beyond its size or not
    Returns:
      [...., height, width, C] Rendered image at the target view.
    """
    target_layers = render_layers(
        layers,
        depths,
        pose,
        intrinsics,
        target_pose,
        target_intrinsics,
        height=height,
        width=width,
        clamp=clamp)
    return compose_back_to_front(target_layers)


inputfiles = []
num_layer = 32
output_dir = os.path.join(os.getcwd(), "Output")

layers_array = tf.TensorArray(tf.float32, size = 0, dynamic_size = True)
iterator = int(0)



for i in tf.range(num_layer):
    # print (str(iterator))
    inputfile = os.path.join(output_dir,"output" + str(iterator) + ".png")
    # print (inputfile)
    input_rgb = tf.image.decode_image(tf.io.read_file(inputfile), dtype=tf.float32)
    # print (input_rgb.shape.as_list())
    layers_array = layers_array.write(i, input_rgb)
    # print(layers_array.size())
    iterator = iterator + 1

layers = layers_array.stack()

print (layers.shape.as_list())

# fn = lambda i: tf.image.decode_image(tf.io.read_file(os.path.join(output_dir,"output" + str(i) + ".jpg")), dtype=tf.float32)

# tensor_list = tf.map_fn(fn=fn, elems=tf.constant([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]))
# layers = tf.stack(tensor_list)
# print(tf.constant(range(0, num_layer))[3])

# inputfile_00 = os.getcwd() + "\img_test_1_00.png"
# inputfile_01 = os.getcwd() + "\img_test_1_01.png"
# inputfile_02 = os.getcwd() + "\img_test_1_02.png"
# inputfile_03 = os.getcwd() + "\img_test_1_03.png"

# input_rgb_00 = tf.image.decode_image(
#     tf.io.read_file(inputfile_00), dtype=tf.float32)
# input_rgb_01 = tf.image.decode_image(
#     tf.io.read_file(inputfile_01), dtype=tf.float32)
# input_rgb_02 = tf.image.decode_image(
#     tf.io.read_file(inputfile_02), dtype=tf.float32)
# input_rgb_03 = tf.image.decode_image(
#     tf.io.read_file(inputfile_03), dtype=tf.float32)

# layers_array_alter = tf.TensorArray(tf.float32, size = 0, dynamic_size = True)
# iterator = 0
# for i in tf.range(4):
#     # print (str(iterator))
#     inputfile = os.path.join(os.getcwd(), "img_test_1_0" + str(iterator) + ".png")
#     # print (inputfile)
#     input_rgb = tf.image.decode_image(tf.io.read_file(inputfile), dtype=tf.float32)
#     # print (input_rgb.shape.as_list())
#     layers_array_alter = layers_array_alter.write(i, input_rgb)
#     # print(layers_array.size())
#     iterator = iterator + 1

# layers_alter = tf.stack([input_rgb_03,
#                    input_rgb_02,
#                    input_rgb_01,
#                    input_rgb_00])
# layers_alter_2 = layers_array_alter.stack()
# print(layers_alter_2.shape.as_list())

depths = make_depths(31, 255, 32)

# print(depths)

# The reference camera position can just be the identity
reference_pose = tf.constant([ [1.0, 0.0, 0.0, 0.0],
                               [0.0, 1.0, 0.0, 0.0],
                               [0.0, 0.0, 1.0, 0.0] ])

# Accurate intrinsics are only important if we are trying to match a ground
# truth output. Here we just give intrinsics for a 16:9 image with the
# principal point in the center.
intrinsics = tf.constant([16/9, 1, 0.5, 0.5])


def render_img(xoffset, yoffset, zoffset):
    # The translation is the final column of the pose matrix
    target_pose = tf.constant([[1.0, 0.0, 0.0, -xoffset],
                               [0.0, 1.0, 0.0, -yoffset],
                               [0.0, 0.0, 1.0, -zoffset]])

    image = render(layers,
                   depths,
                   reference_pose,
                   intrinsics,
                   target_pose,
                   intrinsics,)
    return image

render_dir = os.path.join(os.getcwd(), "Render")
# First move the camera along the X axis (left to right):
for i in range(5):
    #yoffset = ( i + 4) * 0.05
    xoffset = (i - 2) * 0.1
    plt.subplot(1, 5, i + 1)
    img = render_img(xoffset, 0.0, 0.0)
    img = img.numpy()
    plt.imshow(img)
    fname = "render" + str(i) + ".png"
    plt.imsave(os.path.join(render_dir, fname), img)
    # print(img)
    plt.title('Iteration')
    plt.axis('off')
plt.show()




