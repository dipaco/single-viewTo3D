import sys
import os
import pathlib
sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(), '../p2m/'))
import unittest
import trimesh
import metrics # from ../p2m


class SelfIntersection(unittest.TestCase):

    def __init__(self, *args):
        super().__init__(*args)

        current_path = pathlib.Path(__file__).parent.absolute()
        self.test_resources = {
            'self_intersection_mesh_1': os.path.join(current_path, 'self_intersection_mesh_1.obj'),
            'self_intersection_mesh_2': os.path.join(current_path, 'self_intersection_mesh_2.obj')
        }

    def test_count_self_intersection(self):

        mesh1 = trimesh.load(self.test_resources['self_intersection_mesh_1'])
        mesh2 = trimesh.load(self.test_resources['self_intersection_mesh_2'])

        meshes_dict = {
            'data': [
                {
                    'verts': mesh1.vertices,
                    'faces': mesh1.faces,
                },
                {
                    'verts': mesh2.vertices,
                    'faces': mesh2.faces,
                },
            ],
            'num_verts_per_mesh': (mesh1.vertices.shape[0], mesh2.vertices.shape[0]),
            'num_faces_per_mesh': (mesh1.faces.shape[0], mesh2.faces.shape[0])
        }

        loss, ratio = metrics.mesh_self_inter_metric(meshes_dict)

        total_faces = sum(meshes_dict['num_faces_per_mesh'])
        self.assertEqual(2.0, ratio*total_faces)


if __name__ == '__main__':
    unittest.main()
