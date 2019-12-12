#for getting args from script call
import os
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Train LSTM models for all child transcripts in data_dir. Save models in model_dir and results for production scores in result_dir. If test, use random 60% of child utterances for training and test on 40% remaining. ')
    script_dir = os.path.dirname(os.path.realpath(__file__))

    parser.add_argument('-d', '--data_dir', dest='transcript_dir',
                        action='store', required=True,
                        default=script_dir,
                        help='directory where transcripts are stored')
    parser.add_argument('-w', '--word_list', dest='word_list',
                        action='store', required=True,
                        default=script_dir,
                        help='tsv containing word list for aoa prediction')
    parser.add_argument('-m', '--model_dir', dest='model_dir',
                        action='store', required=True,
                        default=script_dir,
                        help='directory where models are written')
    parser.add_argument('-r', '--result_dir', dest='result_dir',
                        action='store', required=True,
                        default=script_dir,
                        help='directory where results are written')
    parser.add_argument('-t', '--test', dest='train_all_data',
                        action='store_true',
                        help='include 60% of child utterances sentences in training')
    return parser.parse_args()

args = get_args()

#libs for helper functions
import re
import math
#import multiprocessing
import random
# Keras model functions and classes
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences

# My classes
from my_data_generator import DataGenerator
from my_decoder_generator import DecoderGenerator

print('TESTED ALL DEPENDENCIES')

tf.test.is_gpu_available

tf.test.gpu_device_name
