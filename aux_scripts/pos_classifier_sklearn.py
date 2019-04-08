#! /usr/bin/env python
# train classifier that takes as input embeddings and predict POS

from __future__ import print_function
import sys, subprocess, os, itertools, pca, tsne, argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneOut
from scipy import spatial
from utils import read_pos_tags, read_mapping, make_emb_dict, split_per_pos_tag
from matplotlib import pyplot

class emb_data(object):

	def __init__(self, emb_dict, pos_tags):

		self.counter_batches = 0

		self.x_array = np.array(emb_dict.values())

		self.y_array = np.array(pos_tags)

		# split in train, validation and test data
		self.size_train = int((float(self.x_array.shape[0]) / 100.0) * 80.0)
		self.size_valid = int((float(self.x_array.shape[0]) / 100.0) * 10.0)
		self.size_test = self.x_array.shape[0] - self.size_train - self.size_valid

		self.x_train = self.x_array[:self.size_train, :]
		self.y_train = self.y_array[:self.size_train]

		self.x_valid = self.x_array[self.size_train:self.size_train+self.size_valid, :]
		self.y_valid = self.y_array[self.size_train:self.size_train+self.size_valid]

		self.x_test = self.x_array[self.size_train+self.size_valid:, :]
		self.y_test = self.y_array[self.size_train+self.size_valid:]



class linear_class(object):

	def __init__(self, multinomial= False, token_based= False):

		if multinomial and not token_based:
			logisticregressionparams = {"random_state": [2017, 1337],
								"penalty": ['l2'],
								"class_weight": ['balanced', None],
								"C": [0.1, 0.4, 0.6, 0.8, 1.0, 1.2, 2.],
								# ovr: binary problem is fit for each label
								# multinomial: multinomial loss
								"multi_class": ['ovr', 'multinomial'],
								# these are the possible solvers for multinomial loss:
								# - lbfgs: limited-memory Broyden-Fletcher-Goldfarb-Shannon algorithm
								# - sag: stochastic average gradient descent
								# - newton-cg
								"solver": ['lbfgs', 'sag', 'newton-cg']}
		else:
			logisticregressionparams = {"random_state": [2017, 1337],
								"penalty": ['l2'],
								"class_weight": ['balanced', None],
								"C": [0.6, 0.8, 1.0, 1.2, 2.],
								# this includes an extra solver: liblinear = coordinate descent algorithm
								"solver": ['liblinear', 'lbfgs', 'sag', 'newton-cg']}

		if token_based:
			# only 1 example for each class, so use leave-one-out cross-validation
			self.pos_logisticregression = GridSearchCV(LogisticRegression(),
									logisticregressionparams,
									cv = LeaveOneOut())
		else:
			self.pos_logisticregression = GridSearchCV(LogisticRegression(),
									logisticregressionparams)

	def train(self, name, x_train, y_train):

		print('Train classifier...')
		self.pos_logisticregression.fit(x_train, y_train)
		joblib.dump(self.pos_logisticregression.best_estimator_, "{0}.estimator".format(name))
		joblib.dump(self.pos_logisticregression.best_params_, "{0}.params".format(name))
		joblib.dump(self.pos_logisticregression.cv_results_, "{0}.results".format(name))
		with open("{0}.score".format(name), "w") as w:
			w.write(str(self.pos_logisticregression.best_score_))

	def test(self, name, x_test, y_test):

		print('Test classifier...')
		pos_classifier = joblib.load('{0}.estimator'.format(name))
		test_score = pos_classifier.score(x_test, y_test)
		print('Test score: {0}'.format(test_score))

	def results_gridsearch(self, name):

		results_gridsearch = joblib.load('{0}.results'.format(name))
		for param, results in results_gridsearch.iteritems():
			print(param, end=' ')
			for el in results:
				print(el, end=' ')
			print()

	def show_confusion_matrix(self, name, x_test, y_test, id_to_pos):

		pos_classifier = joblib.load('{0}.estimator'.format(name))
		label_ids = pos_classifier.classes_
		labels = [id_to_pos[label_id] for label_id in label_ids]
		y_pred = pos_classifier.predict(x_test)

		for l_id in label_ids:
			if l_id not in y_pred:
				print('label {0} is never predicted'.format(id_to_pos[l_id]))

		cm = confusion_matrix(y_test, y_pred)

		# normalise
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

		pyplot.figure(figsize=(15,15))

		# show an image (binary colourmap)
		pyplot.imshow(cm, interpolation='nearest', cmap='binary')
		pyplot.title("Confusion matrix")
		pyplot.colorbar()
		tick_marks = np.arange(len(labels)+1)
		# set labels of axes
		pyplot.xticks(tick_marks, labels, rotation=90)
		pyplot.yticks(tick_marks, labels)
		thresh = 0.5
		for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
			if str(round(cm[i, j], 2))[0] in ["1"]:
				text = str(round(cm[i, j], 2))
			elif str(round(cm[i, j], 2))[0] in ["n"]:
				text = ".0"
			else:
				text = str(round(cm[i, j], 2))[1:]
			if text == ".0":
				text = ""
			pyplot.text(j, i, text,
				horizontalalignment="center",
				color="white" if cm[i, j] > thresh else "black")
		pyplot.tight_layout()
		pyplot.ylabel('True label')
		pyplot.xlabel('Predicted label')
		pyplot.show()
		pyplot.savefig('{0}_confusion_matrix_norm_all.png'.format(name))

	def plot_pca(self, name, x, y, id_to_pos):
		'''
		Makes a PCA plot of the data.
		'''

		labels = []
		for l in np.nditer(y):
			labels.append(id_to_pos[int(l)])

		pca.pca_main(x, labels, '{0}_pca.png'.format(name))

	def plot_tsne(self, name, x, y, id_to_pos):
		'''
		Makes a T-SNE plot of the data.
		'''

		labels = []
		for l in np.nditer(y):
			labels.append(id_to_pos[int(l)])

		unique_ints = range(len(set(labels)))
		colors = [pyplot.cm.jet(float(i)/max(unique_ints)) for i in unique_ints]

		y = tsne.tsne(x)

		for i, label in enumerate(set(labels)):
			indices = [idx for idx, x in enumerate(labels) if x == label]

			pyplot.scatter(np.take(y, indices, axis=0)[:,0], np.take(y, indices, axis=0)[:,1], s=10, c=colors[i], label=label)

		pyplot.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
				ncol=2, fontsize='x-small', mode="expand", borderaxespad=0.)

		pyplot.savefig('{0}_tsne.png'.format(name))
		pyplot.show()

	def plot_lda(self, name, x, y, id_to_pos):

		lda = LinearDiscriminantAnalysis(n_components=1)
		result_lda = lda.fit(x, y).transform(x)

		labels = []
		for l in np.nditer(y):
			labels.append(id_to_pos[int(l)])

		unique_ints = range(len(set(labels)))
		colors = [pyplot.cm.jet(float(i)/max(unique_ints)) for i in unique_ints]

		for color, i, target_name in zip(colors, unique_ints, set(labels)):
			pyplot.scatter(result_lda[y == i, 0], result_lda[y == i, 1], alpha=.8, color=color,
				label=target_name)

	def plot_coef(self, name):
		'''
		Plots the coefficients of a trained file.
		'''

		estimator = joblib.load('{0}.estimator'.format(name))
		coef = estimator.coef_
		label_ids = estimator.classes_
		labels = [id_to_pos[l] for l in label_ids]

		for class_i in xrange(coef.shape[0]):
			pyplot.plot(coef[class_i], label=labels[class_i])

		pyplot.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
		   ncol=2, mode="expand", borderaxespad=0.)

		pyplot.savefig('{0}_coef.png'.format(name))
		#pyplot.show()

	def retrieve_coef(self, name):
		'''
		Saves the coefficients of a trained model as numpy files.
		'''

		estimator = joblib.load('{0}.estimator'.format(name))
		coef = estimator.coef_
		label_ids = estimator.classes_
		labels = [id_to_pos[l] for l in label_ids]

		for i, l in enumerate(labels):
			np.save('{0}_coef_{1}.npy'.format(name, l), coef[i])

	def compare_coef(self, name, x, y, id_to_pos):
		'''
		Compares the saved coefficients (with retrieve_coef) for a certain label
		with all vectors that correspond to that label
		and prints the cosine distances.
		'''

		estimator = joblib.load('{0}.estimator'.format(naliblinearme))
		label_ids = estimator.classes_
		labels = [id_to_pos[l] for l in label_ids]

		all_coef = {}
		for i, l in enumerate(labels):
			print('label {0}'.format(l))
			curr_coef = np.load('{0}_coef_{1}.npy'.format(name, l))
			all_coef[l] = curr_coef

			for id_in_y in xrange(y.shape[0]):
				if id_to_pos[y[id_in_y]] == l:
					emb = x[id_in_y]
					cos_dist = spatial.distance.cosine(curr_coef, emb)
					print('cosine distance {0}'.format(cos_dist))

		for l, coef in all_coef.iteritems():
			for l2, coef2 in all_coef.iteritems():
				tmp_sum = 0.0
				num_occ = 0.0
				for id_in_y in xrange(y.shape[0]):
					if id_to_pos[y[id_in_y]] == l2:
						emb = x[id_in_y]
						cos_dist = spatial.distance.cosine(coef, emb)
						tmp_sum += cos_dist
						num_occ += 1
				avg_dist = tmp_sum / num_occ
				print('Average cosine distance between label {0} and occurrences of {1}: {2}'.format(l, l2, avg_dist))

	def avg_emb(self, name, x, y, id_to_pos):

		avg_dict = {}
		for pos_id in id_to_pos.iterkeys():
			tmp_sum = 0.0
			tmp_denom = 0
			for i in xrange(y.shape[0]):
				if y[i] == pos_id:
					tmp_sum += x[i]
					tmp_denom += 1
			avg_dict[id_to_pos[pos_id]] = tmp_sum / tmp_denom
			np.save('{0}_avg_{1}.npy'.format(name, id_to_pos[pos_id]), (tmp_sum/tmp_denom))


if __name__ == '__main__':

	## input arguments ##
	parser = argparse.ArgumentParser()
	parser.add_argument('emb_f', type=str, help='numpy file containing all embeddings')
	parser.add_argument('dict', type=str, help='dict file containing mapping of words to indices used in the emb_f')
	parser.add_argument('pos_f', type=str, help='file containing list of all words and their possible POS tags')
	parser.add_argument('name', type=str, help='name for the model')
	parser.add_argument('--pos_classes', type=str, help='file containing the POS classes for which we want to train a classifier')
	parser.add_argument('--type_data', type=str,
				help="'collapsed' (=default) if the training set contains 1 (average) embedding for each word, otherwise 'running_text'",
				choices=['collapsed', 'running_text'], default='collapsed')
	parser.add_argument('--freq_cutoff', type=int, help='remove POS tags with frequency < cutoff')
	parser.add_argument('--no_train', help='do not train the classifier (default = train)', action='store_true', default=False)
	parser.add_argument('--no_valid', help='do not calculate accuracy on validation set (default = validate)', action='store_true', default=False)
	parser.add_argument('--no_test', help='do not calculate accuracy on test set (default = test)', action='store_true', default=False)
	parser.add_argument('--grid_search', help='print results of grid search over hyperparameters', action='store_true')
	parser.add_argument('--confusion_matrix', help='plot confusion matrix', action='store_true')
	parser.add_argument('--results_dataset', help="for confusion matrix/pca/tsne/lda: use full dataset ('full') or test set ('test', = default)",
				choices=['full', 'test'])
	parser.add_argument('--pca', help='PCA visualization of embeddings', action='store_true')
	parser.add_argument('--tsne', help='T-SNE visualization of embeddings', action='store_true')
	parser.add_argument('--lda', help='LDA visualization of embeddings', action='store_true')
	parser.add_argument('--plot_coef', help='plot the coefficients/weights of a trained model', action='store_true')
	parser.add_argument('--avg_emb', help='make an average embedding for every class', action='store_true')
	args = parser.parse_args()

	# if a pos_classes argument is given,
	# we only train a classifier for the POS tags in this file
	if args.pos_classes != None:
		tmp = open(args.pos_classes).readlines()

		# mapping based on tokens: pos_classes file should start with 'token'
		if tmp[0] == 'token\n':
			pos_classes = {}
			for el in tmp[1:]:
				split_class = el.strip().split('\t')
				# pos_classes contains class name + set of all words belonging to the class
				pos_classes[split_class[0]] = set(split_class[1].split(' '))
			token_based = True
		else:
			# mapping based on POS classes
			# if the pos_classes file contains class names +
			# list of POS that belong to the class
			if len(tmp[0].strip().split()) > 1:
				pos_classes = {}
				for el in tmp:
					split_class = el.strip().split('\t')
					pos_classes[split_class[0]] = set(split_class[1].split(' '))
			# simple list of POS tags
			else:
				pos_classes = [pos.strip() for pos in tmp]
			token_based = False

		with_classes = True
	else:
		with_classes = False
		token_based = False

	multinomial = True

	# read mapping of words to indices
	mapping = read_mapping(args.dict)

	# read embeddings
	emb_dict, size_emb = make_emb_dict(args.emb_f, mapping)

	# read POS tags
	pos_tags, vocab_pos_tags = read_pos_tags(args.pos_f)

	# throw away embeddings for which we do not have POS tags
	for w in list(emb_dict.iterkeys()):
		if w not in pos_tags:
			#print('no POS tag for {0}'.format(w))
			del emb_dict[w]

	# throw away POS tags for which we do not have embeddings
	for w in list(pos_tags.iterkeys()):
		if w not in emb_dict:
			#print('no embedding for {0}'.format(w))
			del pos_tags[w]

	# if we have only 1 embedding for each word and we want to classify based on POS,
	# it is possible that a word has multiple embeddings
	# so we change the training set, such that multiple training instances are created
	# for every word + POS combination
	if args.type_data == 'collapsed' and not token_based:
		emb_dict, pos_tags = split_per_pos_tag(emb_dict, pos_tags)

	# remove infrequent tags if needed
	if isinstance(args.freq_cutoff, int):
		# first count frequency per tag
		freq_tags = {}
		for tag in pos_tags.values():
			if tag in freq_tags:
				freq_tags[tag] += 1
			else:

				freq_tags[tag] = 1
		# then remove tags with frequency < threshold
		for tag, tag_freq in freq_tags.iteritems():
			print('{0}\t{1}'.format(tag, tag_freq))
			if tag_freq < args.freq_cutoff:
				# remove from pos_tags
				words_to_delete = []
				for word in pos_tags.keys():
					if tag == pos_tags[word]:
						del pos_tags[word]
						words_to_delete.append(word)
				# remove from emb_dict
				for w in words_to_delete:
					del emb_dict[w]

	# remove all training instances that do not
	# belong to the classes that we want to classify
	if with_classes:
		if isinstance(pos_classes, list):
			vocab_pos_tags = pos_classes
			# remove all examples with tags not belonging to pos_classes
			for word in pos_tags.keys():
				if pos_tags[word] not in pos_classes:
					del pos_tags[word]
					del emb_dict[word]
		elif isinstance(pos_classes, dict):
			# first map pos tags in vocab_pos_tags to the right class
			vocab_pos_tags = pos_classes.keys()

			if token_based:
				for word in pos_tags.keys():
					in_training_set = False
					for c, tokens in pos_classes.iteritems():
						if word in tokens:
							# if word in list, map to correct class
							pos_tags[word] = c
							in_training_set = True
					if not in_training_set:
						# otherwise, delete the training example
						del pos_tags[word]
						del emb_dict[word]
			else:
				# map all pos tags in training data to right class
				for word in pos_tags.keys():
					in_training_set = False
					for pos_class in pos_classes.keys():
						if pos_tags[word] in pos_classes[pos_class]:
							# map to correct class
							pos_tags[word] = pos_class
							in_training_set = True
					if not in_training_set:
						# otherwise, delete the training example
						del pos_tags[word]
						del emb_dict[word]

	# map POS tags to POS tag ids
	pos_to_id = dict(zip(vocab_pos_tags, range(len(vocab_pos_tags))))
	id_to_pos = dict(zip(range(len(vocab_pos_tags)), vocab_pos_tags))

	pos_ids = [pos_to_id[pos] for pos in pos_tags.values()]

	# create data object
	data = emb_data(emb_dict, pos_ids)

	# create classifier
	model = linear_class(multinomial, token_based)

	# train classifier
	if not args.no_train:
		model.train(args.name, data.x_train, data.y_train)

	if args.grid_search:
		model.results_gridsearch(args.name)

	if not args.no_valid:
		model.test(args.name, data.x_valid, data.y_valid)

	if not args.no_test:
		model.test(args.name, data.x_test, data.y_test)

	if args.confusion_matrix or args.pca or args.tsne or args.lda:
		# plot for whole dataset
		if args.results_dataset == 'full':
			data_x = data.x_array
			data_y = data.y_array
		# plot for test set only
		else:
			data_x = data.x_test
			data_y = data.y_test

	if args.confusion_matrix:
		pyplot.figure()
		model.show_confusion_matrix(args.name, data_x, data_y, id_to_pos)

	if args.pca:
		model.plot_pca(args.name, data_x, data_y, id_to_pos)

	if args.tsne:
		model.plot_tsne(args.name, data_x, data_y, id_to_pos)

	if args.lda:
		model.plot_lda(args.name, data_x, data_y, id_to_pos)

	if args.plot_coef:
		model.plot_coef(args.name)
		#model.retrieve_coef(name)

		#model.compare_coef(name, data.x_array, data.y_array, id_to_pos)

	if args.avg_emb:
		model.avg_emb(args.name, data.x_array, data.y_array, id_to_pos)
