import tensorflow as tf
import numpy as np
import sys

#auctionmatch_module = tf.load_op_library('./external/tf_auctionmatch_so.so')
sys.path.append('./external')
import tensorflow as tf
# FIXME: Uncomment the next 3 lines
#import tf_sampling
#import tf_auctionmatch
#from .chamfer import nn_distance


def emd_distance(pred, placeholders, block_id=3):
    """
        Computes the EMD distance  for a pair of point clouds
        input: xyz1: (batch_size,#points_1,3)  the first point cloud
        input: xyz2: (batch_size,#points_2,3)  the second point cloud
        output: dist: (batch_size,#point_1)   distance from first to second
        output: matched_out:  (batch_size,#point_1, 3)   nearest neighbor from first to second
    """
    # TODO: properly manage the number of points on each set
    n_points = 1024
    gt_pt = placeholders['labels'][:, :3][:n_points, ...][None, ...]  # gt points
    pred = pred[:n_points, ...][None, ...]

    # Auction assignment
    matchl_out, matchr_out = tf_auctionmatch.auction_match(gt_pt, pred)
    matched_out = tf_sampling.gather_point(pred, matchl_out)

    # Compute the distances from the original points to the assignment
    dist = tf.sqrt(tf.reduce_sum((gt_pt - matched_out) ** 2, axis=2))

    return dist, matched_out


def chamfer_metric(pred, placeholders, block_id=3):
    """
        Computes the symmetric chamfer distance  for a pair of point clouds
        input: xyz1: (batch_size,#points_1,3)  the first point cloud
        input: xyz2: (batch_size,#points_2,3)  the second point cloud
        output: dist: (batch_size,#point_1)   symmetric Chamfer distance between the two point clouds
    """

    gt_pt = placeholders['labels'][:, :3]  # gt points
    pred = pred
    dist1, idx1, dist2, idx2 = nn_distance(gt_pt, pred)
    cd = tf.reduce_mean(dist1) + tf.reduce_mean(dist2)

    return cd


def edge_length_metric(pred, placeholders, block_id=3):
    # edge in graph
    nod1 = tf.gather(pred, placeholders['edges'][block_id - 1][:, 0])
    nod2 = tf.gather(pred, placeholders['edges'][block_id - 1][:, 1])
    edge = tf.subtract(nod1, nod2)

    # edge length loss
    edge_length = tf.reduce_sum(tf.square(edge), 1)
    edge_loss = tf.reduce_mean(edge_length)

    return edge_loss


def get_face_normals(verts, faces):
    """
    Compute the normals on each face and vertex
    :param verts: (n, 3) Array of vertices.
    :param faces: (m, 3) Array with the indices of the vertices in each triangle.
    :return:
        vertex_normals: (n, 3) Normal at each vertex
        face_normals: (m, 3) Normal at each face.
    """

    # Compute the face normals as the cross product of 2 of the edges of the triangle
    # NOTE: The order of the vertices matters so that the normal points in the right direction
    l1 = verts[faces[:, 1]] - verts[faces[:, 0]]
    l2 = verts[faces[:, 2]] - verts[faces[:, 1]]
    face_normals = np.cross(l1, l2)

    # The vertex normals is computed as the average of all the
    vertex_normals = list()
    for v in range(verts.shape[0]):
        vertex_normals.append(face_normals[np.where(np.any(faces == v, axis=1))[0], :].mean(axis=0))
    vertex_normals = np.array(vertex_normals)

    # Normalize the vectors
    vertex_normals /= np.linalg.norm(vertex_normals, axis=1)[:, None]
    face_normals /= np.linalg.norm(face_normals, axis=1)[:, None]

    return vertex_normals, face_normals


def mesh_self_inter_metric(meshes):
    """Compute self intersection loss and ratio."""
    EPS = 1e-6

    def split_ids(s1):
        """Return ids for points on the same side and opposite sides of a plane."""
        # sign of vertex on the other side of plane
        s1sign = np.sign(s1)
        opposite_val = -s1sign.sum(axis=1, keepdims=True)
        _, opposite_id = np.where(s1sign == opposite_val)
        _, same_id = np.where(s1sign != opposite_val)
        same_id = same_id.reshape(-1, 2)
        return opposite_id, same_id

    def segment_plane_inter(n, v, p0, p1):
        """Intersection between segment between p0 and p1 and plane with normal n passing through v."""
        # WARNING: issues with coplanarity!
        return p0 + (p1 - p0) * ((n * (v - p0)).sum(axis=1, keepdims=True) /
                                 (n * (p1 - p0)).sum(axis=1, keepdims=True))

    def triangle_plane_inter(n, v, verts, dists):
        """Triangle and plane intersection, where p0 is on the opposite side."""
        # opposite and same side vertices
        oid, sid = split_ids(dists)
        n_p = len(verts)
        #vo = tf.gather(verts, 1, oid[:, None, None].expand(np, 1, 3))
        vo = np.take_along_axis(verts, indices=np.tile(oid[:, None, None], (1, 1, 3)), axis=1)
        #vs = tf.gather(verts, 1, sid[:, :, None].expand(n_p, 2, 3))
        vs = np.take_along_axis(verts, indices=np.tile(sid[:, :, None], (1, 1, 3)), axis=1)
        #p0, p1, p2 = vo[:, 0], *vs.unbind(dim=1)
        p0, p1, p2 = vo[:, 0], *[np.squeeze(p) for p in np.split(vs, axis=1, indices_or_sections=2)]

        o1 = segment_plane_inter(n, v, p0, p1)
        assert not np.any(np.isnan(o1))
        o2 = segment_plane_inter(n, v, p0, p2)
        assert not np.any(np.isnan(o2))
        return o1, o2

    def get_pairs(verts, faces, meshes, n_interval=4):
        """Partition mesh spatially and only return pairs over each partition."""
        pairs = []
        i_f, i_v = 0, 0
        for nv, nf in zip(meshes['num_verts_per_mesh'], meshes['num_faces_per_mesh']):
            #nf, nv = nf.item(), nv.item()
            f = faces[i_f:i_f + nf]
            v = verts[i_v:i_v + nv]

            limx, limy, limz = [[np.min(v[:, i]), np.max(v[:, i])]
                                for i in range(3)]

            #COMPAT: intx, inty, intz = [tf.linspace(l[0], l[1], n_interval).to(v)
            intx, inty, intz = [np.linspace(l[0], l[1], n_interval)
                                for l in [limx, limy, limz]]

            vf = v[f - i_v]
            for x0, x1 in zip(intx, intx[1:]):
                for y0, y1 in zip(inty, inty[1:]):
                    for z0, z1 in zip(intz, intz[1:]):
                        inside = ((vf[..., 0] >= x0) & (vf[..., 0] <= x1) &
                                  (vf[..., 1] >= y0) & (vf[..., 1] <= y1) &
                                  (vf[..., 2] >= z0) & (vf[..., 2] <= z1))
                        # if any vertex is in, we assign the face to this partition
                        inside = np.where(np.any(inside, axis=1))[0] + i_f
                        # print(f'adding {len(inside)} to partition')
                        # partitions.append(inside)
                        n = len(inside)
                        pairs.append(
                            np.stack(
                                [
                                    np.tile(inside[None, :], reps=(n, 1)).reshape(-1),
                                    np.tile(inside[:, None], reps=(1, n)).reshape(-1)
                                ], axis=-1
                            )
                        )

            i_f += nf
            i_v += nv

        pairs = np.concatenate(pairs)
        return pairs

    def dot_nonorm(x, y):
        assert x.ndim == y.ndim == 2
        return (x * y).sum(axis=1)

    def values_packed(meshes, att):
        """ Concatenates pack the elements of each mesh along the first dimension. att can be  ['verts', 'faces']"""
        arrs = list()
        face_offset = 0
        for d in meshes['data']:
            if att == 'faces':
                arrs.append(d[att] + face_offset)
                face_offset += d[att].shape[0]
            else:
                arrs.append(d[att])

        return np.concatenate(arrs, axis=0)

    # TODO: send vertex as a concatenation of arrays on the first dimension
    #verts = meshes['verts']
    verts = values_packed(meshes, 'verts')
    #faces = meshes['faces']
    faces = values_packed(meshes, 'faces')

    pairs = get_pairs(verts, faces, meshes)
    # print(f'n pairs={len(pairs)}')
    # compute box limits per face
    bmin = verts[faces].min(axis=1)
    bmax = verts[faces].max(axis=1)
    # drop all pairs where rectangles don't intersect
    # this usually drastically reduces candidates
    invalid = (np.any(bmax[pairs[:, 0]] < bmin[pairs[:, 1]], axis=1) +
               np.any(bmax[pairs[:, 1]] < bmin[pairs[:, 0]], axis=1))

    # print(f'non-inter rectangle pairs ratio={invalid.sum().item() / len(pairs)}')
    pairs = pairs[~invalid]

    # intersections: check the sign of each point in P1 wrt P2
    #                if one sign is different; triangle intersects plane
    #                if dot product is too small, point is on the plane
    #                (will happen for adjacent faces)
    _, face_normals = get_face_normals(verts, faces)
    # normals and a barycenter for face1 (hoping bary is more stable)
    n1, p1 = face_normals[pairs[:, 0]], verts[faces[pairs[:, 0]]].mean(axis=1)
    # and for face 2
    n2, p2 = face_normals[pairs[:, 1]], verts[faces[pairs[:, 1]]].mean(axis=1)

    # intersection only happens when both pairs triangle/plane intersect
    # these are from face2 to face1
    s2 = np.stack([(n1 * (np.squeeze(p) - p1)).sum(-1)
                      for p in np.split(verts[faces[pairs[:, 1]]], axis=1, indices_or_sections=3)], axis=1)
    # and from face1 to face2
    s1 = np.stack([(n2 * (np.squeeze(p) - p2)).sum(-1)
                      for p in np.split(verts[faces[pairs[:, 0]]], axis=1, indices_or_sections=3)], axis=1)

    # let us threshold @ 1e-8 -> these are adjacent
    invalid = np.any(np.abs(s1) < EPS, axis=1) | np.any(np.abs(s2) < EPS, axis=1)

    # print(f'neighboring pairs ratio={invalid.sum().item() / len(pairs)}')
    pairs, s1, s2, n1, p1, n2, p2 = [x[~invalid] for x in [pairs, s1, s2, n1, p1, n2, p2]]

    s1bool = s1 > 0
    s2bool = s2 > 0
    nointer = (((s1bool[:, 0] == s1bool[:, 1]) & (s1bool[:, 0] == s1bool[:, 2])) |
               ((s2bool[:, 0] == s2bool[:, 1]) & (s2bool[:, 0] == s2bool[:, 2])))

    # these are the possibly intersecting pairs
    pairs, s1, s2, n1, p1, n2, p2 = [x[~nointer] for x in [pairs, s1, s2, n1, p1, n2, p2]]

    # eliminate coplanar
    # WARNING: this is just to simplify the code;
    # we could handle coplanar intersections as a special case
    # FIXME: As Carlos if this should be OR or AND ( I think should be AND)
    v1 = verts[faces[pairs[:, 0]]]
    v2 = verts[faces[pairs[:, 1]]]
    invalid = ((np.abs(dot_nonorm(n1, v2[:, 0] - v2[:, 1])) < EPS) &
               (np.abs(dot_nonorm(n1, v2[:, 0] - v2[:, 2])) < EPS) &
               (np.abs(dot_nonorm(n1, v2[:, 1] - v2[:, 2])) < EPS) &
               (np.abs(dot_nonorm(n2, v1[:, 0] - v1[:, 1])) < EPS) &
               (np.abs(dot_nonorm(n2, v1[:, 0] - v1[:, 2])) < EPS) &
               (np.abs(dot_nonorm(n2, v1[:, 1] - v1[:, 2])) < EPS))

    # print(f'coplanar faces ratio={invalid.sum().item() / len(pairs)}')
    pairs, s1, s2, n1, p1, n2, p2 = [x[~invalid] for x in [pairs, s1, s2, n1, p1, n2, p2]]

    # gradients are recorded from here
    # we allow backprop through the vertices but not normals
    v1 = verts[faces[pairs[:, 0]]]
    v2 = verts[faces[pairs[:, 1]]]

    o11, o12 = triangle_plane_inter(n2, p2, v1, s1)
    o21, o22 = triangle_plane_inter(n1, p1, v2, s2)

    # some intersections are point-edge and can be ignored
    invalid = ((np.linalg.norm(o11 - o12, axis=1) < EPS) |
               (np.linalg.norm(o21 - o22, axis=1) < EPS))
    pairs, n1, n2, p1, p2, o11, o12, o21, o22 = [x[~invalid] for x in [pairs, n1, n2, p1, p2, o11, o12, o21, o22]]
    # print(f'point-edge inter ratio={invalid.sum().item() / len(pairs)}')

    # intersection line
    line = (o12 - o11) / np.linalg.norm(o12 - o11, axis=1, keepdims=True)
    # distances along line
    # o11 is 0
    to12 = ((o12 - o11) * line).sum(axis=1)
    to21 = ((o21 - o11) * line).sum(axis=1)
    to22 = ((o22 - o11) * line).sum(axis=1)
    to11 = np.zeros_like(to12)

    to11, to12 = np.minimum(to11, to12), np.maximum(to11, to12)
    to21, to22 = np.minimum(to21, to22), np.maximum(to21, to22)
    inter = np.minimum(to12, to22) - np.maximum(to11, to21)

    selfinter = inter > 0
    pairs, inter = [x[selfinter] for x in [pairs, inter]]

    # note: we do not normalize by # of pairs
    #       since loss is supposed to be L1 it shouldn't matter much
    loss = inter.sum() / len(meshes['data'])

    self_inter_faces = np.unique(pairs)
    ratio = np.array(len(self_inter_faces) / len(faces))
    # print(f'ratio of self-intersecting faces={len(self_inter_faces) / len(faces)}')

    return loss, ratio
