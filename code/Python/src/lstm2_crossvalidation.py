#for getting args from script call
import os
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Do crossvalidation for one child corpus. Gets previous test/train data slits from model_dir. Trains LSTM models for all parameter combinations and saves performance by utterance length for all models to one csv file in result_dir for comparison. ')
    script_dir = os.path.dirname(os.path.realpath(__file__))

    parser.add_argument('-m', '--model_dir', dest='model_dir',
                        action='store', required=True,
                        default=script_dir,
                        help='directory where models are written')
    parser.add_argument('-r', '--result_dir', dest='result_dir',
                        action='store', required=True,
                        default=script_dir,
                        help='directory where results are written')
    parser.add_argument('-c', '--corpus_name', dest='corpus_name',
                        action='store', required=True,
                        default=script_dir,
                        help='name of corpus to run crossvalidation on')
    return parser.parse_args()

args = get_args()

#libs for helper functions
import re
import math
#import multiprocessing
import random
# Keras model functions and classes
from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding, LSTM, Dense
from keras.models import Sequential
from keras.regularizers import L1L2
# My classes
from my_data_generator import DataGenerator
from my_decoder_generator import DecoderGenerator


#### GLOBAL VARIABLES ####

model_dir = args.model_dir
result_dir = args.result_dir
childname = args.corpus_name

#### HYPERPARAMETERS ####

#cpus = multiprocessing.cpu_count()
# Nb of epochs (iterations through whole train set)
epochs=15
# Mini-batch size necessary for initializing data generators
batch_size = 32
# Size of word vectors
output_sizes = [10,50,100]
# Nb of hidden neurons in 1 layer of LSTM
hidden_dims = [5,10,25,50]
# regularizations
regularizations = [L1L2(l1=0.0, l2=0.0), L1L2(l1=0.01, l2=0.0), L1L2(l1=0.0, l2=0.01), L1L2(l1=0.01, l2=0.01)]
# Generate sentences in order of transcript
shuffle = False
# Nb of beams for beam beam_search
k = 5


#### HELPER FUNCTIONS ####

def get_train_test():
    trainfile = model_dir + '/train/' + childname + '.train.txt'
    testfile = model_dir + '/test/' + childname + '.test.txt'
    with open(trainfile, 'r') as f:
        train = f.readlines()
    with open(testfile, 'r') as f:
        test = f.readlines()
    return train,test

#Crossvalidates all combinations of params
def cross_validation(hidden_dims, output_sizes, regularizations):
    with open(result_dir+'/'+childname+'.prod_result_crossvalidation.csv','w') as f :
        f.write('decoder,hidden_dim,output_size,L1,L2,utterance_length,nb_utterances,produced,production_score\n')

    for hidden_dim in hidden_dims:
        for output_size in output_sizes:
            for reg in regularizations:
                print("hidden dim: " + str(hidden_dim))
                print("output size: " + str(output_size))
                print("reg: l1-" + str(reg.l1)+ " l2-" + str(reg.l2))
                print('INITIALIZE DATA GENERATORS...\n')
                # Create data generators for train and test sequences
                train_generator = DataGenerator(seqs = train_seqs,
                                                   vocab = vocab,
                                                   vocab_size = vocab_size,
                                                   maxlen = maxlen,
                                                   batch_size = batch_size,
                                                   shuffle = shuffle)
                test_generator = DataGenerator(seqs = test_seqs,
                                                   vocab = vocab,
                                                   vocab_size = vocab_size,
                                                   maxlen = maxlen,
                                                   batch_size = batch_size,
                                                   shuffle = shuffle)

                print('TRAINING MODEL...\n')
                # initialize model
                model = Sequential()
                # add initial embedding layer
                model.add(Embedding(input_dim = vocab_size,  # vocabulary size
                                    output_dim = output_size,  # size of embeddings
                                    input_length = maxlen-1))  # length of the padded sequences minus the last output word



                #add LSTM layers (2 LAYERS)
                model.add(LSTM(hidden_dim, return_sequences=True))
                #second layer with weight regularization
                model.add(LSTM(hidden_dim, bias_regularizer=reg))
                # add layer regular densely connected layer to reshape to output size and use softmax activation for output layer
                model.add(Dense(vocab_size, activation='softmax'))
                # use RMSprop for optimization (could also use Adam or Adagrad) and cross entropy for loss function
                model.compile('rmsprop', 'categorical_crossentropy')

                # Train LSTM
                model.fit_generator(train_generator,
                                    steps_per_epoch = steps_per_epoch,
                                    epochs = epochs,
                                    verbose=2,
                                    max_queue_size=10,
                                    shuffle=False)

                # Initialize decoders and get production scores by utterance length using both the greedy and the beam search decoders
                decoders = DecoderGenerator(model,test_generator,k)
                print('CALCULATING PRODUCTION PERFORMANCE METRIC 1...\n')
                results_greedy = decoders.get_performance_bylength("greedy")
                print('CALCULATING PRODUCTION PERFORMANCE METRIC 2...\n')
                results_beam = decoders.get_performance_bylength("beam")

                # save all performance results
                with open(result_dir+'/'+childname+'.prod_result_crossvalidation.csv','a') as f :
                    for length in results_greedy:
                        f.write('greedy'+','+str(hidden_dim)+','+
                                    str(output_size)+','+
                                    str(reg.l1)+','+
                                    str(reg.l2)+','+
                                    str(length)+','+
                                    str(results_greedy[length][1])+','+
                                    str(results_greedy[length][0])+','+
                                    str(results_greedy[length][0]/results_greedy[length][1])+'\n')
                    for length in results_beam:
                        f.write('beam'+','+str(hidden_dim)+','+
                                    str(output_size)+','+
                                    str(reg.l1)+','+
                                    str(reg.l2)+','+
                                    str(length)+','+
                                    str(results_beam[length][1])+','+
                                    str(results_beam[length][0])+','+
                                    str(results_beam[length][0]/results_beam[length][1])+'\n')
                del model



#### MAIN METHOD ####
train,test = get_train_test()


print('PREPARE DATA FOR: '+childname+'\n')
# Get vocabulary
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train + test)
vocab = tokenizer.word_index
# vocabulary size is equal to the vocab size + the <PAD> character used for
# padding sequences during training
vocab_size = len(vocab) + 1
# transform text strings into sequences of int (representing the word's
# index in vocab)
train_seqs = tokenizer.texts_to_sequences(train)
test_seqs = tokenizer.texts_to_sequences(test)
# get the maximum length of sequences - this is needed for data generator
maxlen = max([len(seq) for seq in train_seqs])
# number of optimization iterations to see whole corpus (epoch)
steps_per_epoch = math.ceil(len(train_seqs)/ batch_size)

print('vocab_size = '+str(vocab_size))
print('train_maxlen = '+str(maxlen))

cross_validation(hidden_dims, output_sizes, regularizations)
