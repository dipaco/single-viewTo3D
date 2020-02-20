#  Copyright (C) 2019 Nanyang Wang, Yinda Zhang, Zhuwen Li, Yanwei Fu, Wei Liu, Yu-Gang Jiang, Fudan University
#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import tensorflow as tf
from p2m.utils import *
from p2m.models import GCN
from p2m.summaries import *
from p2m.fetcher import *
import os
import sys
import yaml
import time

#os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Set random seed
seed = 1024
np.random.seed(seed)
tf.set_random_seed(seed)

# Arguments -- Read a yaml file with configuration parameters to run the project
BASE_FOLDER = os.path.dirname(__file__)
if len(sys.argv) < 2:
	yml_filename = os.path.join(BASE_FOLDER, 'default.yaml')
else:
	yml_filename = sys.argv[1]

with open(yml_filename, 'r') as ymlfile:
	args = yaml.load(ymlfile)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_list', args['dataset']['train_files'], 'Data list.') # training data list
flags.DEFINE_float('learning_rate', 1e-5, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 5, 'Number of epochs to train.')
flags.DEFINE_integer('hidden', 256, 'Number of units in hidden layer.') # gcn hidden layer channel
flags.DEFINE_integer('feat_dim', 963, 'Number of units in feature layer.') # image feature dim
flags.DEFINE_integer('coord_dim', 3, 'Number of units in output layer.') 
flags.DEFINE_float('weight_decay', 5e-6, 'Weight decay for L2 loss.')


# Sets a timer to control training time
TIMEOUT_TERMINATION_SECS = args['training']['timeout']
script_starting_time = time.time()

# Define placeholders(dict) and model
num_blocks = 3
num_supports = 2
placeholders = {
    'features': tf.placeholder(tf.float32, shape=(None, 3)),
    'img_inp': tf.placeholder(tf.float32, shape=(224, 224, 3)),
    'labels': tf.placeholder(tf.float32, shape=(None, 6)),
    'support1': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'support2': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'support3': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'faces': [tf.placeholder(tf.int32, shape=(None, 4)) for _ in range(num_blocks)],  #for face loss, not used.
    'edges': [tf.placeholder(tf.int32, shape=(None, 2)) for _ in range(num_blocks)], 
    'lape_idx': [tf.placeholder(tf.int32, shape=(None, 10)) for _ in range(num_blocks)], #for laplace term
    'pool_idx': [tf.placeholder(tf.int32, shape=(None, 2)) for _ in range(num_blocks-1)] #for unpooling
}
model = GCN(placeholders, logging=True, save_dir=args['logging']['save_dir'])

# Load data, initialize session
data = DataFetcher(FLAGS.data_list)
data.setDaemon(True) ####
data.start()
config=tf.ConfigProto()
#config.gpu_options.allow_growth=True
config.allow_soft_placement=True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

# Restore a previous checkpoint if it exist
if model.has_checkpoint():
	model.load(sess)
	
# Train graph model
train_loss = open('record_train_loss.txt', 'a')
train_loss.write('Start training, lr =  %f\n'%(FLAGS.learning_rate))
pkl = pickle.load(open('Data/ellipsoid/info_ellipsoid_p3.dat', 'rb'), encoding='latin1')
feed_dict = construct_feed_dict(pkl, placeholders)

# Creates summaries to visualize the training
[tf.summary.scalar(metric['name'], metric['var']) for metric in model.metrics()] # a summary for every evaluation metric
tf.summary.scalar('total_loss', model.loss)
image_tensor = draw_render(model.output3, tf.convert_to_tensor(pkl[5][2]), 'matplotlib')
image_summary = tf.summary.image('Render 1', image_tensor)
merged = tf.summary.merge_all()
train_writer = model.get_train_writer(sess)
test_writer = model.get_train_writer(sess)

train_number = data.number
saved_epoch = sess.run(model.epoch_var)
for epoch in range(saved_epoch, FLAGS.epochs):
	all_loss = np.zeros(train_number, dtype='float32')
	for iters in range(train_number):
		# Fetch training data
		img_inp, y_train, data_id = data.fetch()
		feed_dict.update({placeholders['img_inp']: img_inp})
		feed_dict.update({placeholders['labels']: y_train})

		# Training step
		_, step, summary, dists,out1,out2,out3 = sess.run([model.opt_op, model.get_step(), merged, model.loss,model.output1,model.output2,model.output3], feed_dict=feed_dict)

		# Add the summaries to tensorboard
		train_writer.add_summary(summary, step)

		all_loss[iters] = dists
		mean_loss = np.mean(all_loss[np.where(all_loss)])
		if (iters+1) % 128 == 0:
			print('Epoch %d, Iteration %d'%(epoch + 1,iters + 1))
			print('Mean loss = %f, iter loss = %f, %d'%(mean_loss,dists,data.queue.qsize()))

		# Kills the process after one hour of processing.
		if time.time() - script_starting_time > TIMEOUT_TERMINATION_SECS:
			print('Time out termination.')
			exit()

	# increase epoch counter
	increase_epoch = tf.assign(model.epoch_var, epoch + 1)
	sess.run(increase_epoch)


	# Save model
	model.save(sess)
	train_loss.write('Epoch %d, loss %f\n'%(epoch+1, mean_loss))
	train_loss.flush()

data.shutdown()
print('Training Finished!')
