import tfmpl
import numpy as np
import trimesh
import pdb
import os
import matplotlib.pyplot as plt
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
def draw_render(vertices, faces):
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
