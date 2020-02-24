import tensorflow as tf
import sys

#auctionmatch_module = tf.load_op_library('./external/tf_auctionmatch_so.so')
sys.path.append('./external')
import tensorflow as tf
import tf_sampling
import tf_auctionmatch
import tf_nndistances
import tf_approxmatch


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


def nn_metrics(pred, placeholders, block_id=3):
    """
        Computes the EMD distance  for a pair of point clouds with the method from the P2M method
        input: xyz1: (batch_size,#points_1,3)  the first point cloud
        input: xyz2: (batch_size,#points_2,3)  the second point cloud
        output: dist: (batch_size,#point_1)   distance from first to second
        output: matched_out:  (batch_size,#point_1, 3)   nearest neighbor from first to second
    """

    n_points = 1024
    gt_pt = placeholders['labels'][:, :3][:n_points, ...][None, ...]  # gt points
    pred = pred[:n_points, ...][None, ...]

    dist1, idx1, dist2, idx2 = tf_nndistances.nn_distance(gt_pt, pred)
    # earth mover distance, notice that emd_dist return the sum of all distance
    match = tf_approxmatch.approx_match(gt_pt, pred)

    # EMD distance
    emd_dist = tf.reduce_mean(tf_approxmatch.match_cost(gt_pt, pred, match))

    # Chamfer distance
    cd = tf.reduce_mean(dist1) + tf.reduce_mean(dist2)

    return emd_dist, cd


def edge_length_metric(pred, placeholders, block_id=3):
    # edge in graph
    nod1 = tf.gather(pred, placeholders['edges'][block_id - 1][:, 0])
    nod2 = tf.gather(pred, placeholders['edges'][block_id - 1][:, 1])
    edge = tf.subtract(nod1, nod2)

    # edge length loss
    edge_length = tf.reduce_sum(tf.square(edge), 1)
    edge_loss = tf.reduce_mean(edge_length)

    return edge_loss
