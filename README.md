# State Gradients

This is code for calculating, printing and analyzing state gradients of neural language models. State gradients are the gradients of the hidden state of the network with respect to its input embedding. 

It is both a simplified version of [my other GitHub repo](https://github.com/lverwimp/tf-lm/), since it only allows you to train word-level language models, and an extension of it, since it allows you to calculate the state gradients and print them. 

More information can be found in this paper:
Lyan Verwimp, Hugo Van hamme, Vincent Renkens and Patrick Wambacq. 2018. [State Gradients for RNN Memory Analysis](https://www.researchgate.net/publication/325818651_State_Gradients_for_RNN_Memory_Analysis). In: *Proceedings Interspeech*. Hyderabad, Inda, 2-6 September 2018, pp. 1467-1471.

# Installation and setup

* Python version used: 2.7.5. 
* Install [TensorFlow](https://www.tensorflow.org/versions/0.6.0/get_started/os_setup.html#download-and-setup). These scripts are compatible with version 1.8.
* Modify the config files in config/: change the pathnames and optionally the parameters.

# Replicate experiments from the paper

## Data

The data that is used in the paper are Penn TreeBank and Wall Street Journal, normalized to make sure they contain the same vocabulary.

You can download them here:

* Penn TreeBank:
  * [Training set](http://homes.esat.kuleuven.be/~lverwimp/data_state_gradients/ptb_train.txt)
  * [Validation set](http://homes.esat.kuleuven.be/~lverwimp/data_state_gradients/ptb_valid.txt)
  * [Test set](http://homes.esat.kuleuven.be/~lverwimp/data_state_gradients/ptb_test.txt)
* Wall Street Journal:
  * [Training set](http://homes.esat.kuleuven.be/~lverwimp/data_state_gradients/wsj_train.txt)
  * [Validation set](http://homes.esat.kuleuven.be/~lverwimp/data_state_gradients/wsj_valid.txt)
  * [Test set](http://homes.esat.kuleuven.be/~lverwimp/data_state_gradients/wsj_test.txt)
  
## Training

# Your own experiments

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

