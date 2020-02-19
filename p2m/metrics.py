import tensorflow as tf
import sys

#auctionmatch_module = tf.load_op_library('./external/tf_auctionmatch_so.so')
sys.path.append('./external')
import tensorflow as tf
import tf_sampling
import tf_auctionmatch
from tensorflow.python.framework import ops


def emd_distance(pred, placeholders, block_id):
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

