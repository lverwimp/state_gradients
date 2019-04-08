#! /usr/bin/env python

from __future__ import print_function
import tensorflow as tf
import numpy as np
import math, os, sys, time, io

PRINT_SAMPLES = False # if PRINT_SAMPLES is True, all input and target batches are printed
PRINT_INTERMEDIATE = False # if PRINT_INTERMEDIATE is True, ppl and time after every 100 batches are printed

class run_epoch(object):
	'''
	Runs one epoch (one pass over all data) of training, or calculates the validation or test perplexity.
	'''
	def __init__(self, session, model, data_object, data_set, eval_op=None, test=False, valid=False):

		self.session = session
		self.model = model
		self.data_object = data_object
		self.data_set = data_set
		self.eval_op = eval_op
		self.test = test
		self.valid = valid

		if 'save_embedding' in self.model.config:
			self.dict = open('{0}.dict'.format(self.model.config['save_embedding']), 'w')
			self.emb_counter = 0

			if not os.path.exists(self.model.config['save_embedding']):
				os.makedirs(self.model.config['save_embedding'])

		if 'grad_interm' in self.model.config and self.valid:
			self.counter_files = 0

			# make directories for gradient matrices and mapping files
			if not os.path.exists(self.model.config['grad_dir']):
				os.makedirs(self.model.config['grad_dir'])

			if self.model.config['max_delay'] > self.model.config['time_step']:
				num_delays = self.model.config['time_step'] + 1
			else:
				num_delays = self.model.config['max_delay']

			self.mappings = []
			for i in range(num_delays):
				if not os.path.exists(os.path.join(self.model.config['grad_dir'], 'delay_{0}'.format(i))):
					os.makedirs(os.path.join(self.model.config['grad_dir'], 'delay_{0}'.format(i)))

				self.mappings.append(open(os.path.join(self.model.config['grad_dir'],
											'delay_{0}'.format(i),
											'mapping.txt'), 'w'))

	def __call__(self):
		costs = 0.0 # cross entropy based on normal (LM only) probabilities
		iters = 0
		# state = initial state of the model
		state = self.get_init_state()

		# create fetches dictionary = what we want the graph to return
		fetches = self.create_fetches()

		if 'save_embedding' in self.model.config:
			fetches, fetches_save = fetches

		end_reached = False

		# initialize batching
		self.init_batching()
		data_file = None

		# start iterating over data
		while True:
			start_time = time.time()

			# get batch of data
			x, y, end_reached = self.get_batch()

			# if end of data file is reached, end of epoch is reached
			if end_reached:
				break

			# create feed_dict = what we feed into the graph
			feed_dict = self.create_feed_dict(x, y, state)

			# run the model
			# vals contains the values for the variables specified in fetches after feeding feed_dict to the model
			vals = self.session.run(fetches, feed_dict)

			# print gradients for validation set
			if 'grad_interm' in self.model.config and self.valid:
				self.print_gradients(vals, x)

			# debugging: print every sample (input + target) that is fed to the model
			if PRINT_SAMPLES:
				self.print_samples(vals['input_sample'], vals['target_sample'])

			# determine new state: emtpy state or state of previous time step
			state = self.get_new_state(vals)

			# cost = -ln(prob_target_word) / batch_size
			costs += vals["cost"]
			iters += self.model.num_steps

			ppl = np.exp(costs / iters)

			if 'save_embedding' in self.model.config and self.test:
				vals_save = self.session.run(fetches_save, feed_dict)
				# save embedding of current input word
				np.save(os.path.join(self.model.config['save_embedding'], str(self.emb_counter)),
						vals_save['embedding'][vals['input_sample'][0][0]])
				self.emb_counter += 1

			# if PRINT_INTERMEDIATE is True, ppl and time after each batch is printed
			# can be changed to only printing after processing a certain amount of data
			if PRINT_INTERMEDIATE and (iters % (self.model.num_steps*100) == 0):
				print('ppl {0} ({1} seconds)'.format(ppl, time.time() - start_time))

		return ppl

	def get_init_state(self):
		'''
		Initialize to empty state.
		'''

		state = self.session.run(self.model.initial_state)

		return state

	def get_final_state(self, vals):
		'''
		Initialize to state of previous time step.
		'''
		state = vals["final_state"]

		return state

	def get_new_state(self, vals):
		'''
		Determine to what the state of the next time step should be initialized.
		'''

		# use final state of previous state as initial state for next batch
		state = self.get_final_state(vals)

		return state

	def create_fetches(self):
		'''
		Creates a dictionary containing model variables for which we want the new values.
		'''

		fetches = {
			"cost": self.model.cost,
			"final_state": self.model.final_state, # c and h of previous time step (for each hidden layer)
			"input_sample": self.model.input_sample,
			"target_sample": self.model.target_sample,
			}

		# _train_op in training phase
		if self.eval_op is not None:
			fetches["eval_op"] = self.eval_op

		if 'save_embedding' in self.model.config:
			fetches_save = {"embedding": self.model.embedding}

			fetches = (fetches, fetches_save)

		if 'grad_interm' in self.model.config and self.valid:
			fetches["all_grads"] = self.model.all_grads

		return fetches

	def create_feed_dict(self, x, y, state):
		'''
		Creates a dictionary containing the data that will be fed to the placeholders of the model.
		'''

		feed_dict = {self.model.inputs: x, self.model.targets: y,
					self.model.initial_state: state}

		return feed_dict

	def init_batching(self):
		'''
		Initialize batching.
		'''

		if self.test:
			self.data_object.init_batching(self.data_set, test=True)
		else:
			self.data_object.init_batching(self.data_set)

	def get_batch(self):
		'''
		Get a single batch.
		Returns:
			x: input
			y: target
			end_reached: boolean indicating whether the end of the file is reached or not
		'''

		x, y, end_reached = self.data_object.get_batch()

		return x, y, end_reached

	def print_gradients(self, vals, x):

		# vals['all_grads'] contains all gradients for a specific state,
		# namely the state at time step = delay
		# with respect to all input elements before that state

		# if the time_step is larger than the maximum delay that we want gradients for,
		# cutoff will be larger than 0
		cutoff = self.model.config['time_step']-self.model.config['max_delay'] + 1
		if cutoff < 0:
			cutoff = 0
		length_slice = self.model.config['time_step'] - cutoff + 1
		if length_slice == 0:
			length_slice = 1

		relevant_inputs = x[0][cutoff:cutoff+length_slice]

		y = ''
		for i in range(length_slice, 0, -1):
			delay = length_slice - i
			x = self.data_object.id_to_item[relevant_inputs[i-1]]
			y = self.data_object.id_to_item[relevant_inputs[i-1]] + '-' + y

			# write mapping
			self.mappings[delay].write(u'{0}\t{1}\t{2}\n'.format(self.counter_files, x, y).encode(self.data_object.encoding))

			# write gradient matrix
			np.save(os.path.join(self.model.config['grad_dir'],
					'delay_{0}'.format(delay),
					str(self.counter_files)),
					vals['all_grads'][:,i-1,:])

		self.counter_files += 1


	def print_samples(self, input_sample, target_sample):
		'''
		For debugging purposes: if PRINT_SAMPLES = True, print each sample that is given to the model.
		'''

		print('input_sample:', end="")
		for row in input_sample:
			for col in row:

				print(u'{0} '.format(self.data_object.id_to_item[col]).encode('utf-8'), end="")

			print('')

		print('target_sample:', end="")
		for row in target_sample:
			for col in row:

				print(u'{0} '.format(self.data_object.id_to_item[col]).encode('utf-8'), end="")

			print('')
