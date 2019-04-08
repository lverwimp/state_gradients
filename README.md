# State Gradients

This is code for calculating, printing and analyzing state gradients of neural language models. State gradients are the gradients of the hidden state of the network with respect to its input embedding. 

It is both a simplified version of [my other GitHub repo](https://github.com/lverwimp/tf-lm/), since it only allows you to train word-level language models, and an extension of it, since it allows you to calculate the state gradients and print them. 

More information can be found in this paper:

[1] Lyan Verwimp, Hugo Van hamme, Vincent Renkens and Patrick Wambacq. 2018. [State Gradients for RNN Memory Analysis](https://www.researchgate.net/publication/325818651_State_Gradients_for_RNN_Memory_Analysis). In: *Proceedings Interspeech*. Hyderabad, India, 2-6 September 2018, pp. 1467-1471.

We also submitted a paper to Computer Speech and Language with follow-up experiments:

[2] Lyan Verwimp, Hugo Van hamme and Patrick Wambacq. State Gradients for Analyzing Memory in LSTM Language Models. Submitted to *Computer Speech and Language* on April 3, 2019.

# Installation and setup

* Python version used: 2.7.5. 
* Install [TensorFlow](https://www.tensorflow.org/versions/0.6.0/get_started/os_setup.html#download-and-setup). These scripts are compatible with version 1.8.
* Modify the config files in config/: change the pathnames and optionally the parameters.

# Replicate experiments from the paper

## Data

The data that is used in the paper are Penn TreeBank and Wall Street Journal, normalized to make sure they contain the same vocabulary. WSJ can be obtained from LDC, the only difference with the version that we used is the fact that all numbers have been normalized to 'N'.

You can download our version of PTB here:

* [Training set](http://homes.esat.kuleuven.be/~lverwimp/data_state_gradients/ptb_train.txt)
* [Validation set](http://homes.esat.kuleuven.be/~lverwimp/data_state_gradients/ptb_valid.txt)
* [Test set](http://homes.esat.kuleuven.be/~lverwimp/data_state_gradients/ptb_test.txt)
* [Mapping from words to possible POS tags](http://homes.esat.kuleuven.be/~lverwimp/data_state_gradients/ptb_words+pos.txt)

We run additional experiments on a synthetic dataset (not yet published):
  
* [Training set](http://homes.esat.kuleuven.be/~lverwimp/data_state_gradients/synth_train.txt)
* [Validation set](http://homes.esat.kuleuven.be/~lverwimp/data_state_gradients/synth_valid.txt)
* [Test set](http://homes.esat.kuleuven.be/~lverwimp/data_state_gradients/synth_test.txt)
    
## Training and calculating gradients

All scripts can be found in tf_scripts/.

Make sure that you put the PTB files in the directory specified by **data_path** in the config file.

* Train the baseline language model with the following command:
  * python main.py --config ../config/ptb-norm-wsj_64e_256h_steps200.config
* Print gradients for time step 0 (can be done on CPU):
  * python main.py --config ../config/ptb-norm-wsj_64e_256h_steps200_grad_timestep0.config --notrain --notest (--device cpu)
* Make config files for all time steps (make a copy and change value of **time_step**), and run the above command for every time step

!!! Notice that you will need enough disk space, the gradient matrices can easily take up GB's. 

## Training a language model with pre-trained embeddings

You can pre-train word embedding with word2vec and use those embeddings as input to your language model, instead of training the embedding weights jointly with the language model.

Follow these steps:

* Train word2vec embeddings with the [standard toolkit](https://code.google.com/archive/p/word2vec/) and save them in text format.
* Make sure that you have a list of words for which you want to keep the embeddings. It is possible that you train the word2vec embeddings on a much larger dataset with a much larger vocabulary, so you only want to keep the words that are in the LM vocabulary.
* Convert the output of the word2vec script to a format that we will use in our TF scripts:
aux_scripts/convert_word2vec2tf.py <output_word2vec> <word_list> <new_emb_dir>
* Next, you can train a LM by adding the following to your .config file:
pretrained_embeddings <new_emb_dir>

The pre-trained embeddings used in [2] can be downloaded here:

* [Numpy file](http://homes.esat.kuleuven.be/~lverwimp/data_state_gradients/cbow_wsj_all.npy)
* [Dict file](http://homes.esat.kuleuven.be/~lverwimp/data_state_gradients/cbow_wsj_dict)

## Postprocessing

You can compress the gradient matrices with the following command:
* aux_scripts/np2npz.py <gradient_directory>

This will create one .npz file, 'all.npz'. Afterwards, you can delete all .npy files.

Embeddings of a trained model can be printed by added the following key - value pairs to the config file:
* **save_embedding**: directory in which the embeddings should be saved; additionally a <save_embedding>.dict file will be created in which the words for which embeddings have been printed are written
* **other_test**: file for which you want to write the embeddings, typically the training file

Then the embeddings can be printed with the following command:
* python main.py --config <config> --notrain --novalid (--device cpu)

## Analysis

All scripts can be found in aux_scripts/.

* Average gradient matrices over all time steps for specific delay:
  * python avg_gradient_matrix.py \<gradient_directory\> \<delay\> (npz)
  * add 'npz' if the matrices have been compressed
* Average over a specific class of words:
  * python avg_gradient_matrix.py \<gradient_directory\> \<delay\> \<pos_classes\> \<pos_mapping\> (npz)
  * \<pos_mapping\> can be downloaded from the link above
  * \<pos_classes\>: for example aux_scripts/pos_classes/nouns; a file containing all POS tags that belong to a certain class (in this example NN singular common noun, NNS plural common noun, NNP singular proper noun, NNPS plural proper noun)
* Decompose average gradient matrix with Singular Value Decomposition and print top 5 singular values and sum of all singular values (gradients should be averaged):
  * python calculate_svs.py \<gradient_directory\> (\<basename_pos_classes\>)
  * if you want to calculate the singular values for a specific class, add the base name of the \<pos_classes\> file (e.g. nouns)
* Normalization to make sure the embedding space has equal variance, to make sure that variance in the embedding space has no influence on the state gradients:
  * python normalize_equal_variance.py \<emb_file\>
  * \<emb_file\> is the result of printing the embeddings for a trained model (see the [README](config/README.md) for config files)
  * this will create several files in the same directory as where \<emb_file\> can be found:
    * \<emb_file\>_mean.npy: mean of the embeddings
    * \<emb_file\>_covar.npy: covariance matrix of the embeddings
    * \<emb_file\>_Z_norm_emb.npy: weights that can be used to normalize the embeddings
    * \<emb_file\>_D_norm_emb.npy: weights that can be used to normalize the state gradients
* ... I plan to add more scripts in the future.

# Run your own experiments

* First train your own language model (new config file)
  * python main.py --config \<your_own_config\>
* Then make a copy the config file for training, and add the following key - value pairs:
  * **grad_interm**: either 'cell' or 'hidden'
  * **time_step**: integer, time step for which you want to print the gradients
  * **max_delay**: integer, maximum delay that you want to consider
  * **grad_dir**: directory where you want to print the gradients
* Additionally, change **batch_size** in the config file to 1 to avoid memory problems
* Print gradients for the validation set:
  * python main.py --config \<config_for_gradients\> --notrain --notest

