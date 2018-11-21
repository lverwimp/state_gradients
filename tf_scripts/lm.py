#! /usr/bin/env python
# language model class

import tensorflow as tf
import numpy as np
import functools, collections
import IntermStateWrapper

class lm(object):
	'''
	Standard LSTM Language Model:
	Predicting the next word given the previous words.
	'''

	def __init__(self, config, is_training, reuse):
		'''
		Arguments:
			config: configuration dictionary
			is_training: boolean indicating whether we are training or not
			reuse: boolean indicating whether variables should be shared or not
		'''

		self.init_variables(config, reuse)

		# create model
		self.create_graph(is_training)
		self.create_output_weights()

		output = self.get_output(is_training)

		cost = self.calc_cost(output, is_training)

		# do not update weights if you are not training
		if not is_training:
			return
		else:
			self.update_model(cost)

	def init_variables(self, config, reuse):
		'''
		Initialize class variables.
		'''
		self.config = config

		self.batch_size = config['batch_size']
		self.num_steps = config['num_steps']
		self.size = config['size']
		self.vocab_size = config['vocab_size']
		self.reuse = reuse

		if 'embedding_size' in config:
			self.embedding_size = config['embedding_size']
		else:
			self.embedding_size = self.size

		self.inputs = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.num_steps], name='inputs')
		self.targets = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.num_steps], name='targets')

	def create_graph(self, is_training):
		'''
		Creates LSTM graph.
		'''
		with tf.variable_scope("lstm"):

			self.cell = tf.contrib.rnn.MultiRNNCell(
					[tf.contrib.rnn.BasicLSTMCell(self.size, forget_bias=self.config['forget_bias'],
					state_is_tuple=True, reuse=self.reuse) for _ in range(self.config['num_layers'])],
					state_is_tuple=True)

			if 'grad_interm' in self.config:
					# only for an LSTM cell we need a wrapper to retrieve the intermediate cell states
					# for a vanilla RNN or GRU, the output equals the (only) hidden state
					if 'basic_cell' in self.config and \
							self.config['basic_cell'].startswith('LSTM') and \
							self.config['grad_interm'] == 'cell':
						self.cell = IntermStateWrapper.IntermStateWrapper(self.cell, to_return='cell')

			if is_training and self.config['dropout'] < 1:
				self.cell = tf.contrib.rnn.DropoutWrapper(self.cell, output_keep_prob=self.config['dropout'])

			# for a network with multiple LSTM layers,
			# initial state = tuple (size = number of layers) of LSTMStateTuples,
			# each containing a zero Tensor for c and h (each batch_size x size)
			self._initial_state = self.cell.zero_state(self.batch_size, tf.float32)


	def create_output_weights(self):
		'''
		Creates output weight matrix and bias.
		'''

		output_size = self.size

		# output weight matrix and bias
		with tf.variable_scope("output_layer_weights"):
			self.softmax_w = tf.get_variable("softmax_w",
				[output_size, self.vocab_size], dtype=tf.float32)
			self.softmax_b = tf.get_variable("softmax_b",
				[self.vocab_size], dtype=tf.float32)

	def get_output(self, is_training):
		'''
		Feeds self.inputs to the graph and returns the output.
		'''
		with tf.variable_scope("get_output"):
			input_embeddings = self.get_input_embeddings(is_training)

			self.outputs, state = self.feed_to_lstm(input_embeddings)

			output = tf.reshape(tf.concat(self.outputs, 1), [-1, self.size], name="reshape_output")

		return output

	def get_input_embeddings(self, is_training):
		'''
		Creates embedding lookup table and returns the embeddings for self.inputs.
		'''

		with tf.name_scope("embedding_lookup"):
			self.embedding = tf.get_variable("embedding", [self.vocab_size, self.embedding_size], dtype=tf.float32)

			# returns Tensor of size [batch_size x num_steps x size]
			inputs = tf.nn.embedding_lookup(self.embedding, self.inputs, name="input_embeddings")

			# use droput on the input embeddings
			if is_training and self.config['dropout'] < 1:
				inputs = tf.nn.dropout(inputs, self.config['dropout'], name="dropout_inputs")

		return inputs

	def feed_to_lstm(self, inputs):
		'''
		Feeds input embeddings and returns the outputs and the hidden state.
		Input arguments:
			inputs: input embeddings; Tensor of size [batch_size x num_steps x size]
		Returns:
			outputs: outputs of the LSTM; Tensor of size [batch_size x num_steps x size]
			state: the hidden states of the LSTM after processing all inputs; Tensor of size [batch-size x size]
		'''

		state = self._initial_state

		# feed inputs to network: outputs = predictions, state = new hidden state
		outputs, state = tf.nn.dynamic_rnn(self.cell, inputs, sequence_length=None, initial_state=state)

		if 'grad_interm' in self.config:

			outputs, state = self.get_state_gradients(inputs, outputs, state)

		self._final_state = state

		return outputs, state

	def get_state_gradients(self, inputs, outputs, state):
		'''
		Calculates state gradients
		'''

		# if we want to have the gradients of the cell state/memory cell of an LSTM cell:
		# to get the intermediate cell states, we need to use the
		# IntermStateWrapper which returns the cell state at every time step
		# because by default, only the cell state at the last time step is returned
		# in this case, 'outputs' is a tuple containing c and h
		if 'basic_cell' in self.config and \
				self.config['basic_cell'].startswith('LSTM') and \
				self.config['grad_interm'] == 'cell':

			interm_states = outputs.c
			interm_hidden_states = outputs.h

			# redefine 'outputs' because we only use h for prediction
			outputs = interm_hidden_states

		# for the gradients of the output of the LSTM cell, or the hidden states of a GRU/RNN,
		# we only need the regular outputs because they already contain the state for every time step
		else:
			interm_states = outputs

		timestep = self.config['time_step']

		# cutoff = starting index
		# if timestep < max_delay, cutoff = 0
		# otherwise, cutoff = timestep - max_delay + 1
		# e.g. if timestep = 60 and max_delay = 50, cutoff = 60 - 50 + 1 = 11
		cutoff = timestep - self.config['max_delay'] + 1
		if cutoff < 0:
			cutoff = 0

		# number of states that we need, is minimum 1
		# e.g. length_slice = 60 - 11 + 1 = 50
		# --> take a slice starting at index 11 of length 50 = slice[11:60]
		length_slice = timestep - cutoff + 1
		if length_slice == 0:
			length_slice = 1

		# interm_states = Tensor of shape (batch_size, num_steps, hidden_size)
		# slice_state = Tensor of shape 1 x 1 x hidden_size
		slice_state = tf.slice(interm_states, [0, timestep, 0], [self.batch_size, 1, self.size])

		tmp_list = []
		# split per component in the hidden state
		for comp_el in tf.unstack(tf.squeeze(slice_state)):

			# gradient of a single component in a single time step in a single batch
			# w.r.t. all elements of the input
			# result = Tensor of shape 1 x num_steps x embedding_size
			grad_all_inputs = tf.gradients(comp_el, inputs)

			# Tensor of shape 1 x length_slice x embedding_size
			# which only contains the gradients with respect to
			# the current and previous inputs, up to max_delay (length_slice)
			slice_grads = tf.slice(tf.squeeze(grad_all_inputs, axis=0),
					[0, cutoff, 0], [self.batch_size, length_slice, self.embedding_size])

			tmp_list.append(slice_grads)

		# tmp_list = list of length hidden_size,
		# containing gradients for every component of the state
		# self.all_grads = Tensor of size hidden_size x length_slice x embedding_size
		self.all_grads = tf.squeeze(tf.stack(tmp_list), axis=1)

		return outputs, state


	def calc_cost(self, output, is_training):
		'''
		Calculates final predictions and the loss.
		'''

		with tf.name_scope("get_outputs"):
			# logits/scores = Tensor of size [batch_size*num_steps x vocab_size]
			self.logits = tf.matmul(output, self.softmax_w) + self.softmax_b

			self.softmax = tf.nn.softmax(self.logits, name="softmax")

			self.loss = self.get_loss(output, is_training)

			# sum all loss values
			self.reduce_loss = tf.reduce_sum(self.loss, name="reduce_loss")

			# cost = average loss per batch
			self.cost = self.reduce_loss / self.batch_size

		return self.cost

	def update_model(self, cost):
		with tf.name_scope("train_model"):

			self.lr = tf.Variable(float(self.config['learning_rate']), trainable=False, name="learning_rate")
			self.epoch = tf.Variable(0, trainable=False, name="epoch")

			# tvars = list of trainable variables
			tvars = tf.trainable_variables()

			# calculate gradients of cost with respect to variables in tvars
			# + clip the gradients by max_grad_norm:
			#	for each gradient: gradient * max_grad_norm / max (global_norm, max_grad_norm)
			# 	where global_norm = sqrt(sum([l2norm(x)**2 for x in gradient]))
			# 	l2norm = sqrt(sum(squared values))
			grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), self.config['max_grad_norm'], name="clipped_gradients")

			# get correct optimizer
			optimizer = self.get_optimizer()

			# apply gradients + increment global step
			self.train_op = optimizer.apply_gradients(
				zip(grads, tvars),
				tf.train.get_global_step(),
				name="train_op")

			self.new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
			self.lr_update = tf.assign(self.lr, self.new_lr)

			self.new_epoch = tf.placeholder(tf.int32, shape=[], name="new_epoch")
			self.epoch_update = tf.assign(self.epoch, self.new_epoch)


	def assign_lr(self, session, lr_value):
		'''
		Assign lr_value as learning rate.
		'''
		session.run(self.lr_update, feed_dict={self.new_lr: lr_value})

	def assign_epoch(self, session, epoch_value):
		'''
		Assign epoch_value as epoch.
		'''
		session.run(self.epoch_update, feed_dict={self.new_epoch: epoch_value})

	def get_loss(self, output, is_training):
		'''
		Calculates loss based on 'output'.
		Input:
			output: output of LSTM
			is_training: boolean that indicates whether we're training or not
		Returns:
			loss based on full or sampled softmax
		'''
		if self.config['softmax'] == 'full':

			targets = tf.reshape(self.targets, [-1])

			return tf.nn.sparse_softmax_cross_entropy_with_logits(
					labels=targets,
					logits=self.logits)

		elif self.config['softmax'] == 'sampled':
			# number of classes to randomly sample per batch
			NUM_SAMPLED = 32

			# sampled softmax is only for training
			if is_training:

				targets = tf.reshape(self.targets, [-1,1])

				return tf.nn.sampled_softmax_loss(
					weights=tf.transpose(self.softmax_w),
					biases=self.softmax_b,
					inputs=output,
					labels=targets,
					num_sampled=NUM_SAMPLED,
					num_classes=self.vocab_size)
			else:

				targets = tf.reshape(self.targets, [-1])

				return tf.nn.sparse_softmax_cross_entropy_with_logits(
					labels=targets,
					logits=self.logits)
		else:
			raise ValueError("Specify which softmax should be used: full or sampled.")


	def get_optimizer(self):
		'''
		Returns the optimizer asked for.
		'''
		if self.config['optimizer'] == 'sgd':
			return tf.train.GradientDescentOptimizer(self.lr)
		elif self.config['optimizer'] == 'adam':
			# default learning rate = 1e-3
			return tf.train.AdamOptimizer(self.config['learning_rate'])
		elif self.config['optimizer'] == 'adagrad':
			return tf.train.AdagradOptimizer(self.config['learning_rate'])
		else:
			raise ValueError("Specify an optimizer: stochastic gradient descent (sgd), adagrad or adam.")


	@property
	def initial_state(self):
		return self._initial_state

	@property
	def initial_state_bw(self):
		return self._initial_state_bw

	@property
	def input_sample(self):
		return self.inputs

	@property
	def target_sample(self):
		return self.targets

	@property
	def final_state(self):
		return self._final_state

	@property
	def final_state_bw(self):
		return self._final_state_bw
