#! /usr/bin/env python

from __future__ import absolute_import
from __future__ import division

import time, os, sys, random, io

# Condor: adapt PYTHONPATH and LD_LIBRARY_PATH
if 'PYTHONPATH' not in os.environ:
        if not os.path.isdir('/usr/lib/nvidia') and not os.path.isfile('/usr/lib64/libcuda.so'):
                os.environ['PYTHONPATH'] = '/users/spraak/spch/prog/spch/tensorflow_cpu-1.4.0/lib/python2.7/site-packages'
        else:
                os.environ['PYTHONPATH'] = '/users/spraak/spch/prog/spch/tensorflow-1.4.0/lib/python2.7/site-packages'
        if 'LD_LIBRARY_PATH' not in os.environ:
                os.environ['LD_LIBRARY_PATH'] = '/users/spraak/spch/prog/spch/cuda-8.0/lib64:/users/spraak/spch/prog/spch/cuda-9.0/extras/CUPTI/lib64'
        try:
                os.system('/usr/bin/python ' + ' '.join(sys.argv))
                sys.exit(0)
        except Exception, exc:
                print('failed executing')
                sys.exit(1)


import numpy as np
import tensorflow as tf

from writer import writer
import configuration, lm_data, multiple_lm_data, trainer, lm, run_epoch

print('TensorFlow version: {0}'.format(tf.__version__))

# command line arguments
flags = tf.flags
flags.DEFINE_string("config", None,"Configuration file")
flags.DEFINE_boolean("train", True,"Train the model or not.")
flags.DEFINE_boolean("valid", True,"Validate the model or not.")
flags.DEFINE_boolean("test", True,"Test the model or not.")
flags.DEFINE_string("device", None,"Specify 'cpu' if you want to run on cpu.")
FLAGS = flags.FLAGS

def read_data(config, eval_config, (TRAIN, VALID, TEST)):
	'''
	Reads data from file.
	Inputs:
		config: dictionary containing configuration options (for training and validation)
		eval_config: dictionary containing configuration options (for testing)
		(TRAIN, VALID, TEST): tuple of booleans indicating whether we should train, validate and/or test
	Returns:
		config: dictionary containing configuration options (for training and validation)
		eval_config: dictionary containing configuration options (for testing)
		data: data object
		train_data: training data mapped to indices (can be single list or tuple of lists depending on the type of model)
		valid_data: validation data mapped to indices
		test_data: test data mapped to indices
		(TRAIN, VALID, TEST): tuple of booleans indicating whether we should train, validate and/or test
	'''

	data = lm_data.LMData(config, eval_config, TRAIN, VALID, TEST)
	all_data, vocab_size, _ = data.get_data()

	# update vocab_size
	config['vocab_size'] = vocab_size
	eval_config['vocab_size'] = vocab_size

	train_data = all_data[0]
	valid_data = all_data[1]
	test_data = all_data[2]

	return config, eval_config, data, train_data, valid_data, test_data, (TRAIN, VALID, TEST)

def create_lm(config, is_training, reuse, test=False):
	'''
	Creates language model.
	'''
	lm_obj = lm.lm(config, is_training=is_training, reuse=reuse)

	return lm_obj


def main(_):
	# process command line arguments and configuration
	if FLAGS.config == None:
		raise ValueError("Please specify a configuration file (with --config).")
	else:
		config = configuration.get_config(FLAGS.config)

	TRAIN = FLAGS.train
	VALID = FLAGS.valid
	TEST = FLAGS.test

	if TRAIN and not VALID:
		raise ValueError("Training and validation are always combined. Set both TRAIN = True and VALID = True.")

	device = FLAGS.device
	if device == 'cpu':
		os.environ['CUDA_VISIBLE_DEVICES']="" # if you don't want to use GPU


	eval_config = config.copy() # same parameters for evaluation, except for:
	eval_config['batch_size'] = 1 # batch_size
	eval_config['num_steps'] = 1

	# make sure save_path and log_dir exist
	try:
		os.makedirs(config['save_path'])
	except OSError:
		pass

	try:
		os.makedirs(config['log_dir'])
	except OSError:
		pass

	# if no name for the log file is specified, use the same name as save_path
	if 'log' in config:
		log_file = os.path.join(config['log_dir'], '{0}.log'.format(os.path.basename(os.path.normpath(config['log']))))
	else:
		log_file = os.path.join(config['log_dir'], '{0}.log'.format(os.path.basename(os.path.normpath(config['save_path']))))

	# if log file already exists, make a new version by adding a random number (to avoid overwriting)
	if os.path.isfile(log_file):
		rand_num = round(random.random(),3)
		log_file = log_file.rstrip('.log') + str(rand_num) + '.log'

	fout = file(log_file,'w',0)
	# write both to standard output and log file
	sys.stdout = writer(sys.stdout, fout)

	print('configuration:')
	for par,value in config.iteritems():
		print('{0}\t{1}'.format(par, value))

	# read data in appropriate format + adapt configs if necessary
	config, eval_config, data, train_data, valid_data, test_data, (TRAIN, VALID, TEST) = read_data(
		config, eval_config, (TRAIN, VALID, TEST))

	with tf.Graph().as_default():

		if not 'random' in config:
			# use the same seed for random initialization (to better compare models)
			tf.set_random_seed(1)
		initializer = tf.random_uniform_initializer(minval=-config['init_scale'], maxval=config['init_scale'], seed=1)

		if TRAIN:

			reuseOrNot = True # valid and test models: reuse the graph

			print('Create training model...')
			with tf.name_scope("Train"):
				with tf.variable_scope("Model", reuse=None, initializer=initializer):

					train_lm = create_lm(config, is_training=True, reuse=False)
					merged = tf.summary.merge_all()
					train_writer = tf.summary.FileWriter(os.path.join(config['save_path'], 'train'))

				saver = tf.train.Saver()

		else:
			reuseOrNot = None

		if VALID:
			print('Create validation model...')
			with tf.name_scope("Valid"):
				with tf.variable_scope("Model", reuse=reuseOrNot, initializer=initializer):
					valid_lm = create_lm(config, is_training=False, reuse=reuseOrNot)

			if reuseOrNot == None:
				reuseOrNot = True


		if TEST:

			print('Create testing model...')
			with tf.name_scope("Test"):
				with tf.variable_scope("Model", reuse=reuseOrNot, initializer=initializer):
					test_lm = create_lm(eval_config, is_training=False, reuse=reuseOrNot, test=True)

		sv = tf.train.Supervisor(logdir=config['save_path'])

		# allow_soft_placement: automatically choose device
		with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as session:

			if TRAIN and VALID:

				train_writer.add_graph(session.graph)

				# create a trainer object based on config file
				class_name = 'trainer.{0}'.format(config['trainer'])
				train_obj = eval(class_name)(session, saver, config, train_lm, valid_lm, data, train_data, valid_data)

				# train + validate the model
				train_obj.train()

				train_writer.close()

			if VALID and not TRAIN:

				#validator = run_epoch.run_epoch(session, valid_lm, data, valid_data)
				validator = run_epoch.run_epoch(session, valid_lm, data, valid_data,
						eval_op=None, test=False, valid=True)

				valid_perplexity = validator()
				print('Valid Perplexity: {0}'.format(valid_perplexity))

			if TEST:

				# test the model
				print('Start testing...')

				tester = run_epoch.run_epoch(session, test_lm, data, test_data,
					eval_op=None, test=True)

				test_perplexity = tester()

				print('Test Perplexity: {0}'.format(test_perplexity))


if __name__ == "__main__":
	tf.app.run()
