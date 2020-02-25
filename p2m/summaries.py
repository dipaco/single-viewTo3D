import tfmpl
import numpy as np
import trimesh
import pdb
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pylab as plt
os.environ['PYOPENGL_PLATFORM'] = 'egl'
#os.environ['EGL_DEVICE_ID'] = os.environ['SLURM_STEP_GPUS'].split(',')[0]
import pyrender

@tfmpl.figure_tensor
def draw_scatter(scaled, colors):
    '''Draw scatter plots. One for each color.'''
    figs = tfmpl.create_figures(len(colors), figsize=(4,4))
    for idx, f in enumerate(figs):
        ax = f.add_subplot(111)
        ax.axis('off')
        ax.scatter(scaled[:, 0], scaled[:, 1], c=colors[idx])
        f.tight_layout()

    return figs

@tfmpl.figure_tensor
def draw_render(gt_points, vertices, faces, render_type='matplotlib'):

    # INFO: For visualization purpose on matplotlib we rotate all the points 90º around the x-axis
    # to aling the z axis with the up direction.
    p_idx = np.random.choice(gt_points.shape[0], 1024, replace=False)
    rot_mat = np.array([[1.0, 0.0, 0.0],
                        [0, 0.0, 1.0],
                        [0.0, 1.0, 0.0]])
    gt_points = np.einsum('dc,nc -> nd', rot_mat, gt_points[p_idx, :3])
    vertices = np.einsum('dc,nc -> nd', rot_mat, vertices)

    angles = [0, np.pi/3]
    '''Draw scatter plots. One for each color.'''
    figs = tfmpl.create_figures(len(angles) + 1, figsize=(3, 3), dpi=200)
    for idx in range(len(angles)):
        f = figs[idx]
        ax = f.add_subplot(111, projection='3d')
        #ax.axis('off')

        # getting the triangles of the mesh
        vertices -= vertices.mean(axis=0)
        vertices /= np.linalg.norm(vertices, axis=1).max()
        triangles = vertices[faces]

        # rotate the mesh
        ang = angles[idx]
        rot_mat = np.array([[np.cos(ang), -np.sin(ang), 0.0],
                            [np.sin(ang), np.cos(ang), 0.0],
                            [0.0, 0.0, 1.0]])
        triangles = np.einsum('dc,npc -> npd', rot_mat, triangles)

        ax.add_collection3d(
            Poly3DCollection(triangles, facecolors='lightgray', linewidths=.1, edgecolors='black', alpha=0.2))
        _set_unit_limits_in_3d_plot(ax)
        f.tight_layout()

    ax = figs[-1].add_subplot(111, projection='3d')
    #ax.axis('off')
    gt_points = gt_points - gt_points.mean(axis=0)
    gt_points /= np.linalg.norm(gt_points, axis=1).max()
    ax.scatter(gt_points[:, 0], gt_points[:, 1], gt_points[:, 2], c='gray', marker='.', s=10)
    _set_unit_limits_in_3d_plot(ax)
    figs[-1].tight_layout()

    return figs


def _pyrender_render(vertices, faces):
    colors = ['r']
    '''Draw scatter plots. One for each color.'''
    figs = tfmpl.create_figures(len(colors), figsize=(4,4))
    for idx, f in enumerate(figs):
        ax = f.add_subplot(111)
        ax.axis('off')

        # Creating the PyRender mesh object
        aux = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh = pyrender.Mesh.from_trimesh(aux, smooth=False)

        # compose scene
        scene = pyrender.Scene(ambient_light=[.1, .1, .3], bg_color=[0, 0, 0])
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=2e3)

        scene.add(mesh, pose=np.eye(4))
        scene.add(light, pose=np.eye(4))

        c = 2 ** -0.5
        scene.add(camera, pose=[[1, 0, 0, 0],
                                [0, c, -c, -2],
                                [0, c, c, 2],
                                [0, 0, 0, 1]])

        # render scene
        r = pyrender.OffscreenRenderer(128, 128)
        color, _ = r.render(scene)

        # plotting the redered image
        ax.imshow(color)
        f.tight_layout()

    return figs


def _set_unit_limits_in_3d_plot(ax):
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-0.8, 0.8)
    ax.set_ylim(-0.8, 0.8)
    ax.set_zlim(-0.8, 0.8)
    #ax.set_zticks([-1, 0, 1])
    #plt.xticks([-1, 0, 1])
    #plt.yticks([-1, 0, 1])
