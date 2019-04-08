#! /usr/bin/env python

from __future__ import print_function

import tensorflow as tf
import os, collections, sys, subprocess, io
from abc import abstractmethod
import numpy as np

def save_item_to_id(item_to_id, file, encoding):
	'''
	Saves a item_to_id mapping to file.
	'''
	out = io.open(file, 'w', encoding=encoding)
	for item, id_ in item_to_id.iteritems():
		if item == '':
			print('EMPTY ELEMENT')
		if item == ' ':
			print('SPACE')
		out.write(u'{0}\t{1}\n'.format(item, id_))
	out.close()

def load_item_to_id(file, encoding):
	'''
	Loads an item_to_id mapping and corresponding id_to_item mapping from file.
	'''

	item_to_id = {}
	id_to_item = {}

	for line in io.open(file, 'r', encoding=encoding):
		l = line.strip().split()
		item_to_id[l[0]] = int(l[1])
		id_to_item[int(l[1])] = l[0]

	return item_to_id, id_to_item

class LMData(object):
	'''
	The input data: words, batches across sentence boundaries.
	'''

	def __init__(self, config, eval_config, TRAIN, VALID, TEST):
		'''
		Arguments:
			config: configuration dictionary, specifying all parameters used for training
			eval_config: configuration dictionary, specifying all parameters used for testing
			TRAIN: boolean indicating whether we want to train or not
			VALID: boolean indicating whether we want to validate or not
			TEST: boolean indicating whether we want to test or not
		'''

		self.config = config
		self.eval_config = eval_config
		self.TRAIN = TRAIN
		self.VALID = VALID
		self.TEST = TEST

		# if we want to train with a limited vocabulary, words not in the vocabulary
		# should already be mapped to UNK
		# data files should be of format train_50k-unk.txt etc. for a 50k vocabulary
		if config['vocab']:
			train_file = "train_" + str(config['vocab']) + "k-unk.txt"
			valid_file = "valid_" + str(config['vocab']) + "k-unk.txt"
			test_file = "test_" + str(config['vocab']) + "k-unk.txt"
			self.train_path = os.path.join(config['data_path'], train_file)
			self.valid_path = os.path.join(config['data_path'], valid_file)
			self.test_path = os.path.join(config['data_path'], test_file)
		else:
			self.train_path = os.path.join(config['data_path'], "train.txt")
			self.valid_path = os.path.join(config['data_path'], "valid.txt")
			self.test_path = os.path.join(config['data_path'], "test.txt")

		self.batch_size = config['batch_size']
		self.num_steps = config['num_steps']

		self.eval_batch_size = eval_config['batch_size']
		self.eval_num_steps = eval_config['num_steps']

		self.iterator = 0
		self.end_reached = False

		# default encoding = utf-8, specify in config file if otherwise
		if 'encoding' in self.config:
			self.encoding = self.config['encoding']
		else:
			self.encoding = "utf-8"

		self.id_to_item = {}
		self.item_to_id = {}

		# by default, unknown words are represented with <unk>
		# if this is not the case for a certain dataset, add it here
		if 'CGN' in self.config['data_path']:
			self.unk = '<UNK>'
			self.replace_unk = '<unk>'
		else:
			self.unk = '<unk>'
			self.replace_unk = '<UNK>'

		if 'other_test' in self.config:
			self.test_path = self.config['other_test']
		if 'valid_as_test' in self.config:
			self.test_path = self.valid_path

	def read_items(self, filename):
		'''
		Returns a list of all WORDS in filename.
		'''

		with tf.gfile.GFile(filename, "r") as f:
			# Wikitext: more than 1 sentence per line, also introduce <eos> at ' . '
			# add here other datasets that contain more than 1 sentence per line
			if "WikiText" in self.config['data_path']:
				data = f.read().decode(self.encoding).replace("\n", " <eos> ").replace(" . "," <eos> ").split()
			elif 'no_eos' in self.config:
				data = f.read().decode(self.encoding).replace("\n", " ").split()
			else:
				data = f.read().decode(self.encoding).replace("\n", " <eos> ").split()

			# make sure there is only 1 symbol for unknown words
			data = [self.unk if word==self.replace_unk else word for word in data]

			return data

	def build_vocab(self, filename):
		'''
		Returns an item-to-id and id-to-item mapping for all words (or characters) in filename.
		Arguments:
			filename: name of file for which the mapping will be built
		Returns:
			item_to_id mapping and id_to_item mapping
		'''

		data = self.read_items(filename)

		counter = collections.Counter(data)

		# counter.items() = list of the words in data + their frequencies, then sorted according to decreasing frequency
		count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

		# words = list of all the words (in decreasing frequency)
		items, _ = list(zip(*count_pairs))

		# make a dictionary with a mapping from each word to an id; word with highest frequency gets lowest id etc.
		item_to_id = dict(zip(items, range(len(items))))

		# remove empty element and space
		if '' in item_to_id:
			item_to_id.pop('')
		if ' ' in item_to_id and not 'char' in self.config:
			item_to_id.pop(' ')

		# reverse dictionary
		id_to_item = dict(zip(range(len(items)), items))

		# make sure there is a special token for unknown words
		if not self.unk in item_to_id:
			item_to_id[self.unk] = len(item_to_id)
			id_to_item[len(id_to_item)] = self.unk

		# add <bos>: used for sentence-level batches, or
		# for discourse-level models that are use for e.g. rescoring
		item_to_id['<bos>'] = len(item_to_id)
		id_to_item[len(id_to_item)] = '<bos>'

		return item_to_id, id_to_item


	def extend_vocab(self, filename):
		'''
		If there already is a vocabulary, this function extends the vocabulary with words
		found in the data file 'filename'.
		'''

		data = self.read_items(filename)
		vocab_curr = set(data)

		for word in vocab_curr:
			if word not in self.item_to_id:
				print(u'word {0} not yet seen'.format(word).encode(self.encoding))
				self.item_to_id[word] = len(self.item_to_id)
				self.id_to_item[len(self.id_to_item)] = word

	def file_to_item_ids(self, filename, item_to_id=None):
		'''
		Returns list of all words/characters (mapped to their ids) in the file,
		either one long list or a list of lists per sentence.
		Arguments:
			filename: name of file for which the words should be mapped to their ids
			Optional:
			item_to_id: dictionary that should be used for the mapping (otherwise self.item_to_id is used)
		'''

		if item_to_id == None:
			item_to_id = self.item_to_id

		data = self.read_items(filename)
		tmp_l = []
		for w in data:
			if w in item_to_id:
				tmp_l.append(item_to_id[w])

		return [item_to_id[item] if item in item_to_id else item_to_id[self.unk] for item in data]

	def read_data(self):
		'''
		Makes sure there is a vocabulary and reads all necessary data.
		Returns:
			all_data: tuple of three lists : train_data, valid_data and test_data
		'''
		if 'pretrained_embeddings' in self.config:
                	self.item_to_id, self.id_to_item = load_item_to_id(
				os.path.join(self.config['pretrained_embeddings'], 'dict'), self.encoding)
		
		if 'read_vocab_from_file' in self.config:
			# read vocabulary mapping from file
			self.item_to_id, self.id_to_item = load_item_to_id(self.config['read_vocab_from_file'], self.encoding)

			# check whether the data file contains words that are not yet in the vocabulary mapping
			self.extend_vocab(self.train_path)

		else:
			# if the vocabulary mapping is not saved on disk, make one based on the training data
			self.item_to_id, self.id_to_item = self.build_vocab(self.train_path)

			# save the item_to_id mapping such that it can be re-used
			if 'save_dict' in self.config:
				save_item_to_id(self.item_to_id, self.config['save_dict'], self.encoding)

		# make a label file to visualize the embeddings
		# with the correct labels (= words instead of ids) in tensorboard
		self.label_file = os.path.join(self.config['save_path'], "labels.tsv")

		# Write label file
		with io.open(self.label_file, "w", encoding=self.encoding) as f:
			for i in range(len(self.id_to_item)):
				f.write(u'{0}\n'.format(self.id_to_item[i]))

		# list of all words in training data converted to their ids
		if self.TRAIN:
			train_data = self.file_to_item_ids(self.train_path)
		else:
			train_data = []

		# list of all words in validation data converted to their ids
		if self.VALID:
			valid_data = self.file_to_item_ids(self.valid_path)
		else:
			valid_data = []

		# list of all words in test data converted to their ids
		if self.TEST:
			test_data = self.file_to_item_ids(self.test_path)
		else:
			test_data = []

		all_data = (train_data, valid_data, test_data)

		return all_data

	def get_data(self):
		'''
		Retrieve the necessary data and vocabulary size.
		'''
		all_data = self.read_data()
		return all_data, len(self.id_to_item), 0

	def init_batching(self, data, test=False):
		'''
		Prepare for batching.
		'''

		if test:
			batch_size = self.eval_batch_size
			self.num_steps = self.eval_num_steps
		else:
			batch_size = self.batch_size

		# beginning of data set: set self.end_reached to False (was set to True if another data set is already processed)
		if self.iterator == 0:
			self.end_reached = False

		data_len = len(data)

		# to divide data in batch_size batches, each of length batch_len
		batch_len = data_len // batch_size

		# number of samples that can be taken from the batch_len slices
		self.num_samples = (batch_len // self.num_steps) - 1

		# remove last part of the data that doesn't fit in the batch_size x num_steps samples
		data = data[:batch_size * batch_len]

		# convert to numpy array: batch_size x batch_len
		self.data_array = np.array(data).reshape(batch_size, batch_len)

	def get_batch(self):
		'''
		Gets a single batch.
		Returns:
			x: input data
			y: target data
			end_reached: boolean marking whether the end of the data file has been reached or not
		'''

		# take slice of batch_size x num_steps
		x = self.data_array[:, self.iterator * self.num_steps :
					(self.iterator * self.num_steps) + self.num_steps]
		# targets = same slice but shifted one step to the right
		y = self.data_array[:, (self.iterator * self.num_steps) +1 :
					(self.iterator * self.num_steps) + self.num_steps + 1]

		# if iterated over the whole dataset, set iterator to 0 to start again
		if self.iterator >= self.num_samples:
			self.iterator = 0
			self.end_reached = True
		# otherwise, increase count
		else:
			self.iterator += 1

		return x, y, self.end_reached
