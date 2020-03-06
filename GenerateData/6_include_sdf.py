import matplotlib
import matplotlib.pyplot as plt
import glob
import sys
import os
import click
import pdb
import trimesh
import numpy as np
import pickle
import scipy
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.coo import coo_matrix
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

camera_transform_module = __import__('3_camera_transform')

#SHAPENET_FOLDER = sys.argv[1]
#PATH_TO_3D_R2N2 = sys.argv[2]

def load_obj(fn, no_normal=False):

    fin = open(fn, 'r')
    lines = [line.rstrip() for line in fin]
    fin.close()

    vertices = []; normals = []; faces = [];
    for line in lines:
        if line.startswith('v '):
            vertices.append(np.float32(line.split()[1:4]))
        elif line.startswith('vn '):
            normals.append(np.float32(line.split()[1:4]))
        elif line.startswith('f '):
            faces.append(np.int32([item.split('/')[0] for item in line.split()[1:4]]))

    raw_mesh = dict()
    raw_mesh['faces'] = np.vstack(faces)
    raw_mesh['vertices'] = np.vstack(vertices)

    if (not no_normal) and (len(normals) > 0):
        if len(normals) == len(vertices):
            mesh['normals'] = np.vstack(normals)
        else:
            print('WARNING: vertices != #normals')

    mesh = trimesh.Trimesh(vertices=raw_mesh['vertices'], faces=(raw_mesh['faces'] - 1), process=False)
    mesh = trimesh.load('cow2.ply')

    return mesh

def get_edges(vertices, faces):
    pdb.set_trace()

    all_edges = np.vstack((faces[:, 0:2], faces[:, 1:3], np.roll(faces, axis=1, shift=1)[:, 1:3]))
    sorted_edges =  np.sort(all_edges, axis=1)
    unique_edges, unique_idx = np.unique(sorted_edges, axis=0, return_index=True)

    edges = all_edges[unique_idx, :]

    return edges

def get_object_info(view_filename, shapenet_folder):
    parts = view_filename.split('/')

    # loads the ground thruth mesh
    mesh_filename = os.path.join(shapenet_folder, parts[2], parts[3], 'model.obj')
    mesh = load_obj(mesh_filename)

    # loads the 3D-R2N2 data
    sampled_points = pickle.load(open(os.path.join('..', view_filename), 'rb'), encoding='latin1')  # removes the last \n character

    # loads the rendered image
    img = plt.imread(os.path.join('..', view_filename.replace('dat', 'png')))

    # Get rendering parameters
    rendering_metadata_filename = os.path.join('..', *parts[:-1], 'rendering_metadata.txt')
    rendering_metadata = np.loadtxt(rendering_metadata_filename)
    idx = int(parts[-1][:2])

    return mesh, sampled_points, rendering_metadata[idx, :], img

def compute_sdf(object_info): 
   
    n_voxels = 64
    coords_min = data[:, 0:3].min(axis=0)
    coords_max = data[:, 0:3].max(axis=0)
    
    lx = np.linspace(coords_min[0], coords_max[0], n_voxels)
    ly = np.linspace(coords_min[1], coords_max[1], n_voxels)
    lz = np.linspace(coords_min[2], coords_max[2], n_voxels)

    XX, YY, ZZ = np.meshgrid(lx, ly, lz)

    # array with all the coordinates of the pixels in the SDF grid
    grid_points = np.array([XX.reshape(-1), YY.reshape(-1), ZZ.reshape(-1)]).T
    sign_factor = np.ones_like(grid_points)

    #nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(data[:, 0:3])
    #distances, indices = nbrs.kneighbors(grid_points)
    #dist_matrix = scipy.spatial.distance.cdist(grid_points, data[:, 0:3], 'euclidean')
    ret = -1*trimesh.proximity.signed_distance(mesh, grid_points)
    #nn_dist = dist_matrix.min(axis=1)

    
    return mesh.vertices, mesh.faces

def apply_rendering_transform_1(vertices, rendering_info):
    """
    Camera transformation from the original mesh to the render view and pose.
    This algorithm uses the original p2m implementation.
    :param vertices:
    :param face_normals:
    :param vertex_normals:
    :param rendering_info:
    :return:
    """

    #INFO: This constant is defined in the original P2M code
    position = vertices * 0.57
    cam_mat, cam_pos = camera_transform_module.camera_info(rendering_info)

    pt_trans = np.dot(position - cam_pos, cam_mat.transpose())
    #face_norm_trans = np.dot(face_normals, cam_mat.transpose())
    #vertex_norm_trans = np.dot(vertex_normals, cam_mat.transpose())

    return pt_trans #, face_norm_trans, vertex_norm_trans

def apply_rendering_transform_2(vertices, rendering_info):
    """
        Camera transformation from the original mesh to the render view and pose.
        This algorithm uses our own implementation determining the correct transformation
        for the pose and rotations.
        :param vertices:
        :param face_normals:
        :param vertex_normals:
        :param rendering_info:
        :return:
    """

    raise ValueError('Wrong implementation.')

    #INFO: This scale constant was arbitrarily defined in the original P2M code
    vertices *= 0.57
    vertices = np.hstack((vertices, np.ones((vertices.shape[0], 1))))

    translation = np.array([[1.0, 0.0, 0.0, rendering_info[3]],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0]])

    #
    az_angle = 180 - rendering_info[0]*np.pi/180
    az_rot = np.array([[np.cos(az_angle), 0.0, np.sin(az_angle), 0.0],
                       [0.0, 1.0, 0.0, 0.0],
                       [-np.sin(az_angle), 0.0, np.cos(az_angle), 0.0],
                       [0.0, 0.0, 0.0, 1.0]])

    el_angle = rendering_info[1]*np.pi/180
    '''el_rot = np.array([[1.0, 0.0, 0.0, 0.0],
                       [0.0, np.cos(el_angle), -np.sin(el_angle), 0.0],
                       [0.0, np.sin(el_angle), np.cos(el_angle), 0.0],
                       [0.0, 0.0, 0.0, 1.0]])'''
    el_rot = np.array([[np.cos(el_angle), -np.sin(el_angle), 0.0, 0.0],
                       [np.sin(el_angle), np.cos(el_angle), 0.0, 0.0],
                       [0.0, 0.0, 1.0, 0.0],
                       [0.0, 0.0, 0.0, 1.0]])

    rot_matrix = np.dot(el_rot, az_rot)
    full_matrix = np.dot(translation, rot_matrix)

    vertices = np.einsum('ji, ni -> nj', el_rot.T, vertices)
    vertices = np.einsum('ji, ni -> nj', az_rot.T, vertices)
    vertices = np.einsum('ji, ni -> nj', translation, vertices)

    #face_normals = np.einsum('ji, ni -> nj', rot_matrix[:3, :3], face_normals)
    #vertex_normals = np.einsum('ji, ni -> nj', rot_matrix[:3, :3], vertex_normals)
    
    return vertices[:, :3] #, face_normals[:, :3], vertex_normals[:, :3]

def _set_unit_limits_in_3d_plot(ax):
    # In ShapeNet the Y axis goes up
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')

    ax.set_xlim(-0.4, 0.4)
    ax.set_ylim(-0.4, 0.4)
    ax.set_zlim(-0.4, 0.4)

@click.command()
@click.option('--shapenet-folder')
@click.option('--file-list')
def add_mesh_info(shapenet_folder, file_list, save_fig=False):
    #print(glob.glob(os.path.join(path_to_3d, '*/*/models/*.obj')))
    with open(file_list, 'r') as files:
        info = map(lambda f: get_object_info(f[:-1], shapenet_folder) + (f[:-1],), files.readlines())
        pbar = tqdm(info)
        for i, (mesh, sampled_points, render_info, render_img, fn) in enumerate(pbar):

            # Rotates the vertices to match the rendering camera
            #print(f'{i} processing mesh: {fn}')
            #pbar.update(f'Processing mesh: ({i}) {fn}')
            vertices = apply_rendering_transform_1(mesh.vertices, render_info)

            # Computes the gauss and mean curvatures of the gt mesh

            save_curvature_figure(mesh, figname=f'{i}')

            # save the mesh info
            mesh_info = {
                'vertices': mesh.vertices,
                'faces': mesh.faces,
                'face_normals': None,
                'vertex_normals': None,
                'sdf': None,
                'edges': mesh.edges_unique,
                'gauss_curvature': mesh.vertex_defects,
                'mean_curvature': None
            }
            pickle.dump(mesh_info, open(os.path.join('..', fn).replace('.dat', '_gt_mesh.dat'), 'wb'))

            if save_fig:
                save_figures(i, render_img, sampled_points, vertices)


def save_curvature_figure(mesh, figname=None):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Defining the colors for every curvature value
    k_face_value = mesh.vertex_defects[mesh.faces].mean(axis=1)
    minima = k_face_value.min()
    maxima = k_face_value.max()

    norm = matplotlib.colors.Normalize(vmin=-np.pi, vmax=np.pi, clip=True)
    mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap='coolwarm')

    #pdb.set_trace()

    ax.add_collection3d(
        Poly3DCollection(mesh.triangles, facecolors=mapper.to_rgba(k_face_value), linewidths=.1, alpha=0.2)
    )
    _set_unit_limits_in_3d_plot(ax)

    fig_filename = f'gauss_curvature_{figname}.png' if figname is not None else f'gauss_curvature.png'
    plt.savefig(fig_filename)

def save_figures(i, render_img, sampled_points, vertices):
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(131, projection='3d')
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2])
    ax.scatter(0, 0, 0, color='r', s=2)
    # ax.view_init(elev=30, azim=90)
    _set_unit_limits_in_3d_plot(ax)
    ax = fig.add_subplot(132, projection='3d')
    ax.scatter(sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2])
    # ax.view_init(elev=30, azim=90)
    _set_unit_limits_in_3d_plot(ax)
    plt.subplot(133)
    plt.imshow(render_img)
    # ax.set_xlim(0.0, render_img.shape[1])
    # ax.set_ylim(0.0, render_img.shape[0])
    plt.savefig(f'test_{i}.png')


if __name__ == '__main__':
    add_mesh_info()
