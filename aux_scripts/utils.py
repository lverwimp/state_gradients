#! /usr/bin/env python

import os
import numpy as np

def read_mapping(dict_f):
    '''
    Read mapping from dict file
    Arguments:
        dict_f: file with mapping of words to the indices used for the numpy files
    Returns:
        mapping: dictionary containing words and indices used for the numpy files
    '''
    mapping = {}
    for line in open(dict_f, 'r'):
        mapping[line.split()[0]] = line.strip().split()[1]
    return mapping

def make_emb_dict(emb_f, mapping):
    '''
    Make a dictionary with as keys the words and as values the embeddings
    Arguments:
        emb_f: npy file containing all embeddings
        mapping: result of read_mapping
    Returns:
        emb_dict: dictionary containing words + their embeddings
        size_emb: size of the embeddings
    '''
    emb_dict = {}
    all_emb = np.load(emb_f)

    for w, w_id in mapping.iteritems():
        emb_dict[w] = all_emb[int(w_id)]

    return emb_dict, all_emb[0].shape[0]

def read_pos_tags(pos_f):
    '''
    Read POS tags
    Arguments:
        pos_f: file containing mapping of words to possible POS tags
    Returns:
        pos_tags: dictionary containing words + list of their tags
        vocab_pos_tags: list containing all possible POS tags
    '''
    pos_tags = {}
    vocab_pos_tags = []
    for line in open(pos_f, 'r'):
        l = line.strip().split()
        # skip the tokens for which there is no POS tag
        if ' '.join(l[1:]) == 'not in list':
            continue

        list_tags = []
        for tag in l[1:]:
            tag = tag.lstrip("[").rstrip(",").rstrip("]").strip("''")
            if '|' in tag:
                split = tag.split('|')
                for t in split:
                    list_tags.append(t)
                    if t not in vocab_pos_tags:
                        vocab_pos_tags.append(t)
            else:
                list_tags.append(tag)
                if tag not in vocab_pos_tags:
                    vocab_pos_tags.append(tag)

        pos_tags[l[0]] = list_tags

    return pos_tags, vocab_pos_tags

def split_per_pos_tag(emb_dict, pos_tags):
    '''
    If words have multiple POS tags, make separate training example of each word + tag combination.
    Arguments:
        emb_dict: output of make_emb_dict, dictionary containing words + their embeddings
        pos_tags: output of read_pos_tags, dictionary containing words + list of their tags
    Returns:
        new_emb_dict: dictionary containing word_counter + embedding
        new_pos_tags: dictionary containing word_counter + tag (instead of list of tags)
    '''
    new_pos_tags = {}
    new_emb_dict = {}

    for w, tag_set in pos_tags.iteritems():
        if len(tag_set) > 1:
            counter = 0
            for tag in tag_set:
                # version of word with counter
                adap_w = '{0}_{1}'.format(w, counter)
                new_pos_tags[adap_w] = tag
                new_emb_dict[adap_w] = emb_dict[w]
                counter += 1
        else:
            new_pos_tags[w] = tag_set[0]
            new_emb_dict[w] = emb_dict[w]

    return new_emb_dict, new_pos_tags
