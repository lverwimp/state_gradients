# Config files used in papers

The config files used in the Interspeech paper is the following:

[ptb-norm-wsj_64e_256h_steps50.config](ptb-norm-wsj_64e_256h_steps50.config)

This trains a baseline PTB model with sequence length = 50.

The config files used in the paper submitted to Computer Speech and Language are the following:
* Penn TreeBank:
  * [ptb-norm-wsj_64e_256h_steps200.config](ptb-norm-wsj_64e_256h_steps200.config): This trains the same baseline model but with sequence length = 200.
  * [ptb-norm-wsj_pretr-wsj_64e_256h_steps200.config](ptb-norm-wsj_pretr-wsj_64e_256h_steps200.config): Train the baseline model but with pre-trained WSJ embeddings (see the main [README](../README.md) for more information and to download the pre-trained embeddings).
  * [ptb-norm-wsj_64e_128h_steps200.config](ptb-norm-wsj_64e_128h_steps200.config): Train LSTM with smaller hidden size, 128 instead of 256.
  * [ptb-norm-wsj_64e_512h_steps200.config](ptb-norm-wsj_64e_512h_steps200.config): Train LSTM with larger hidden size, 512 instead of 256.
  * [ptb-norm-wsj_GRU_64e_256h_steps200.config](ptb-norm-wsj_GRU_64e_256h_steps200.config): Train GRU with same hyperparameters as baseline LSTM.
  * [ptb-norm-wsj_RNN_64e_256h_steps200.config](ptb-norm-wsj_RNN_64e_256h_steps200.config): Train vanilla RNN with same hyperparameters as baseline LSTM.
* WikiText:
  * [wiki2_64e_256h_steps200.config](wiki2_64e_256h_steps200.config): Train baseline on Wiki-2 with same hyperparameters as PTB baseline.
  * [wiki2_64e_512h_steps200.config](wiki2_64e_512h_steps200.config): Train LSTM with larger hidden size on Wiki-2, 512 instead of 256.
  * [wiki103_64e_512h_steps200.config](wiki103_64e_512h_steps200.config): Train LSTM with size 512 on Wiki-103 (vocabulary is the same as for Wiki-2).
  
To print the gradients for a trained model, make a copy of the config file and add the following lines:
* **grad_interm**: either 'cell' or 'hidden'
* **time_step**: integer, time step for which you want to print the gradients
* **max_delay**: integer, maximum delay that you want to consider
* **grad_dir**: directory where you want to print the gradients
Additionally, change **batch_size** in the config file to 1 to avoid memory problems

[ptb-norm-wsj_64e_256h_steps200_grad_timestep0.config](ptb-norm-wsj_64e_256h_steps200_grad_timestep0.config): Example of a config file to print gradients for the baseline model.

Save the trained embedding weights (will be used to normalize the embedding space):
* **save_embedding**: filename for the numpy file that contains the trained embeddings
* optionally: **save_dict**: filename to write the word to ID mapping

[ptb-norm-wsj_64e_256h_steps200_save-emb.config](ptb-norm-wsj_64e_256h_steps200_save-emb.config): Example of a config file to print embeddings for the baseline model.

