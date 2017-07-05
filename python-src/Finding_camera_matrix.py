# Author: S. Chris Colbert <sccolbert@gmail.com>
# Copyright (c) 2009, S. Chris Colbert
# License: BSD Style

from __future__ import print_function

# this import is here because we need to ensure that matplotlib uses the
# wx backend and having regular code outside the main block is PyTaboo.
# It needs to be imported first, so that matplotlib can impose the
# version of Wx it requires.
import matplotlib
matplotlib.use('WXAgg')
import pylab as pl


import numpy as np
from mayavi import mlab
from mayavi.core.ui.mayavi_scene import MayaviScene

def get_world_to_view_matrix(mlab_scene):
    """returns the 4x4 matrix that is a concatenation of the modelview transform and
    perspective transform. Takes as input an mlab scene object."""

    if not isinstance(mlab_scene, MayaviScene):
        raise TypeError('argument must be an instance of MayaviScene')


    # The VTK method needs the aspect ratio and near and far clipping planes
    # in order to return the proper transform. So we query the current scene
    # object to get the parameters we need.
    scene_size = tuple(mlab_scene.get_size())
    clip_range = mlab_scene.camera.clipping_range
    aspect_ratio = float(scene_size[0])/float(scene_size[1])

    # this actually just gets a vtk matrix object, we can't really do anything with it yet
    vtk_comb_trans_mat = mlab_scene.camera.get_composite_projection_transform_matrix(
                                aspect_ratio, clip_range[0], clip_range[1])

     # get the vtk mat as a numpy array
    np_comb_trans_mat = vtk_comb_trans_mat.to_array()

    return np_comb_trans_mat


def get_view_to_display_matrix(mlab_scene):
    """ this function returns a 4x4 matrix that will convert normalized
        view coordinates to display coordinates. It's assumed that the view should
        take up the entire window and that the origin of the window is in the
        upper left corner"""

    if not (isinstance(mlab_scene, MayaviScene)):
        raise TypeError('argument must be an instance of MayaviScene')

    # this gets the client size of the window
    x, y = tuple(mlab_scene.get_size())

    # normalized view coordinates have the origin in the middle of the space
    # so we need to scale by width and height of the display window and shift
    # by half width and half height. The matrix accomplishes that.
    view_to_disp_mat = np.array([[x/2.0,      0.,   0.,   x/2.0],
                                 [   0.,  -y/2.0,   0.,   y/2.0],
                                 [   0.,      0.,   1.,      0.],
                                 [   0.,      0.,   0.,      1.]])

    return view_to_disp_mat


def apply_transform_to_points(points, trans_mat):
    """a function that applies a 4x4 transformation matrix to an of
        homogeneous points. The array of points should have shape Nx4"""

    if not trans_mat.shape == (4, 4):
        raise ValueError('transform matrix must be 4x4')

    if not points.shape[1] == 4:
        raise ValueError('point array must have shape Nx4')

    return np.dot(trans_mat, points.T).T


if __name__ == '__main__':
    f = mlab.figure()

    N = 4

    # create a few points in 3-space
    X = np.random.random_integers(-3, 3, N)
    Y = np.random.random_integers(-3, 3, N)
    Z = np.random.random_integers(-3, 3, N)

    # plot the points with mlab
    pts = mlab.points3d(X, Y, Z)

    # now were going to create a single N x 4 array of our points
    # adding a fourth column of ones expresses the world points in
    # homogenous coordinates
    W = np.ones(X.shape)
    hmgns_world_coords = np.column_stack((X, Y, Z, W))

    # applying the first transform will give us 'unnormalized' view
    # coordinates we also have to get the transform matrix for the
    # current scene view
    comb_trans_mat = get_world_to_view_matrix(f.scene)
    view_coords = \
            apply_transform_to_points(hmgns_world_coords, comb_trans_mat)

    # to get normalized view coordinates, we divide through by the fourth
    # element
    norm_view_coords = view_coords / (view_coords[:, 3].reshape(-1, 1))

    # the last step is to transform from normalized view coordinates to
    # display coordinates.
    view_to_disp_mat = get_view_to_display_matrix(f.scene)
    disp_coords = apply_transform_to_points(norm_view_coords, view_to_disp_mat)

    # at this point disp_coords is an Nx4 array of homogenous coordinates
    # where X and Y are the pixel coordinates of the X and Y 3D world
    # coordinates, so lets take a screenshot of mlab view and open it
    # with matplotlib so we can check the accuracy
    img = mlab.screenshot()
    pl.imshow(img)

    for i in range(N):
        print('Point %d:  (x, y) ' % i, disp_coords[:, 0:2][i])
        pl.plot([disp_coords[:, 0][i]], [disp_coords[:, 1][i]], 'ro')

    pl.show()

    # you should check that the printed coordinates correspond to the
    # proper points on the screen

    mlab.show()

#EOF