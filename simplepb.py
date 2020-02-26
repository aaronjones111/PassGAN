import tensorflow as tf
    
"""
Try to generate graph information for visualizing in tesorboard

tensorboard --logdir ./rockyou/graph/
"""

g = tf.Graph()

with g.as_default() as g:
    tf.train.import_meta_graph('./rockyou/checkpoints/checkpoint_200000.ckpt.meta')

with tf.Session(graph=g) as sess:
    file_writer = tf.summary.FileWriter(logdir='./rockyou/graph', graph=g)