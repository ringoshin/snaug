#
# Library of classes to encapsulate GPU usage, as well as for NLP using Tensorlow/Keras
# LSTM models.
#

import tensorflow as tf

from keras.layers import LSTM, CuDNNLSTM
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential
from keras.callbacks import LambdaCallback, ModelCheckpoint
from keras.models import load_model

import pickle
import matplotlib.pyplot as plt


# Create a class for NLP using LSTM models
class TFModelLSTM:
    """
    A simple abstract class to escapsulate usage of GPU when available.
    This will be part of classes for NLP using Tensorflow/Keras LSTM models.
    """
    # Class variable tracking GPU availablity
    gpu_test = tf.test.gpu_device_name()
    have_gpu = True if 'GPU' in gpu_test else False
    
   	# Initialize object with self and GPU usage options
    def __init__(self, use_gpu=True):
        self.use_tfgpu(use_gpu)
        
    # Explicit setting of GPU usage
    def use_tfgpu(self, use_gpu):
        self.use_cudadnn = use_gpu and TFModelLSTM.have_gpu
        
        # For future implementation, this will force to CPU even if GPU exists 
        #import os
        #os.environ["CUDA_VISIBLE_DEVICES"]="-1" 

    # Customized LSTM method to use cudaDNN if available
    def select_LSTM(self, *args, **kwargs):
        if self.use_cudadnn:
            return CuDNNLSTM(*args, **kwargs)
        else:
            return LSTM(*args, **kwargs)

    # Define dummy LSTM model
    def define(self, vocab_size, embedding_size):
        # define model
        self.model = Sequential()
        self.model.add(Embedding(input_dim=vocab_size, output_dim=embedding_size))
        self.model.add(self.select_LSTM(embedding_size, return_sequences=True))
        self.model.add(Dense(embedding_size, activation='relu'))
        self.model.add(Dense((vocab_size+1), activation='softmax'))

    # Compile LSTM model
    def compile(self, *args, **kwargs):
        self.model.compile(*args, **kwargs)
        #metrics=['categorical_accuracy']

    # Fit LSTM model
    def fit(self, *args, **kwargs):
        self.history = self.model.fit(*args, **kwargs)
        return self.history
        #metrics=['categorical_accuracy']

    def plot_training(self, starting_epoch=0):
        # acc = history.history['categorical_accuracy']
        acc = self.history.history['acc']
        loss = self.history.history['loss']
        
        nb_epochs = len(acc)
        starting_epoch = starting_epoch if starting_epoch < nb_epochs else 0
        epochs = range(starting_epoch, nb_epochs)

        # Plot the accuracy curves
        plt.figure(figsize=(6,6))
        plt.plot(epochs, acc[starting_epoch:],'b', linestyle='-', label='accuracy')
        plt.plot(epochs, loss[starting_epoch:],'r', linestyle='-', label='loss')
        plt.title('Training Accuracy/Loss per Epoch')
        plt.legend()

    # serialize model weights to HDF5 and save model training history
    def save_weights_and_history(self, fname_prefix='trained_model'):
        weights_fname = fname_prefix + '_weights.h5'
        history_fname = fname_prefix + '_history.pkl'
        self.model.save_weights(weights_fname)
        pickle.dump(self.history, open(history_fname, 'wb'))

    # load model weights
    def load_weights(self, fname_prefix='trained_model'):
        weights_fname = fname_prefix + '_weights.h5'
        self.model.load_weights(weights_fname)

    # save model
    def save(self, fname_prefix='trained_model'):
        model_fname = fname_prefix + '_model.h5'   
        self.model.save(model_fname)

    # load model
    def load(self, fname_prefix='trained_model'):
        model_fname = fname_prefix + '_model.h5'   
        self.model = load_model(model_fname)
    

class TFModelLSTMCharToken(TFModelLSTM):
    """
    A child class to escapsulate an LSTM model using character tokenization 
    to generate text
    """
   	# Initialize class with self and which model to use
    def __init__(self, use_gpu=True, model_name='Character Tokenisation'):
        self.model_name = model_name
        super().__init__(use_gpu)
        #TFModelLSTM.__init__(self, use_gpu)

    # Define LSTM model that trains using character tokenization
    def define(self, maxlen, num_unique_char):
        self.model = Sequential()

        self.model.add(self.select_LSTM(512, 
                       input_shape=(maxlen, num_unique_char),
                       return_sequences=True))
        self.model.add(Dropout(0.2))

        self.model.add(self.select_LSTM(512))
        self.model.add(Dropout(0.2))

        self.model.add(Dense(num_unique_char))

        self.model.add(Activation('softmax'))


class TFModelLSTMWordToken(TFModelLSTM):
    """
    A child class to escapsulate an LSTM model using word tokenization and
    word embedding to generate text
    """
   	# Initialize class with self and which model to use
    def __init__(self, use_gpu=True, model_name='Word Tokenisation'):
        self.model_name = model_name
        super().__init__(use_gpu)
        #TFModelLSTM.__init__(self, use_gpu)

    # Define LSTM model that trains from scratch using word embedding
    def define(self, vocab_size, embedding_size, seq_length):
        # define model
        self.model = Sequential()
        self.model.add(Embedding(input_dim=(vocab_size+1), 
                                 output_dim=embedding_size, 
                                 input_length=seq_length))

        self.model.add(self.select_LSTM(128, return_sequences=True))
        self.model.add(Dropout(0.2))

        self.model.add(self.select_LSTM(128))
        self.model.add(Dropout(0.2))

        self.model.add(Dense(128, activation='relu'))

        self.model.add(Dense((vocab_size+1), activation='softmax'))


class TFModelLSTMWord2vec(TFModelLSTM):
    """
    A child class to escapsulate an LSTM model using word-tokenisation and
    pre-trained Word2vec to generate text
    """    
   	# Initialize class with self and which model to use
    def __init__(self, use_gpu=True, model_name='Pre-trained Word2vec'):
        self.model_name = model_name
        super().__init__(use_gpu)
        #TFModelLSTM.__init__(self, use_gpu)

    # Define LSTM model using pre-trained Word2vec weights
    def define(self, vocab_size, embedding_size, pretrained_weights):
        # define model
        self.model = Sequential()
        self.model.add(Embedding(input_dim=vocab_size, 
                                 output_dim=embedding_size, 
                                 weights=[pretrained_weights]))

        self.model.add(self.select_LSTM(embedding_size, return_sequences=True))
        self.model.add(Dropout(0.2))

        self.model.add(self.select_LSTM(embedding_size))
        self.model.add(Dropout(0.2))

        self.model.add(Dense(embedding_size, activation='relu'))

        self.model.add(Dense((vocab_size+1), activation='softmax'))



if __name__ == '__main__':
    pass
