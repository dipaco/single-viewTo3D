import tensorflow as tf
import numpy as np

import tfmpl


@tfmpl.figure_tensor
def draw_scatter(scaled, colors):
    '''Draw scatter plots. One for each color.'''
    figs = tfmpl.create_figures(len(colors), figsize=(4, 4))
    for idx, f in enumerate(figs):
        ax = f.add_subplot(111)
        ax.axis('off')
        ax.scatter(scaled[:, 0], scaled[:, 1], c=colors[idx])
        f.tight_layout()

    return figs


with tf.Session(graph=tf.Graph()) as sess:
    # A point cloud that can be scaled by the user
    points = tf.constant(
        np.random.normal(loc=0.0, scale=1.0, size=(100, 2)).astype(np.float32)
    )
    scale = tf.placeholder(tf.float32)
    scaled = points * scale

    # Note, `scaled` above is a tensor. Its being passed `draw_scatter` below.
    # However, when `draw_scatter` is invoked, the tensor will be evaluated and a
    # numpy array representing its content is provided.
    image_tensor = draw_scatter(scaled, ['r', 'g'])
    image_summary = tf.summary.image('scatter', image_tensor)
    all_summaries = tf.summary.merge_all()

    writer = tf.summary.FileWriter('log', sess.graph)
    summary = sess.run(all_summaries, feed_dict={scale: 2.})
    writer.add_summary(summary, global_step=0)