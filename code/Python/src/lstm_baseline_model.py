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
from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding, LSTM, Dense
from keras.models import Sequential
# My classes
from my_data_generator import DataGenerator
from my_decoder_generator import DecoderGenerator


#### HYPERPARAMETERS ####

#cpus = multiprocessing.cpu_count()
# Nb of epochs (iterations through whole train set)
epochs=15
# Mini-batch size necessary for initializing data generators
batch_size = 32
# Size of word vectors
output_size = 100
# Nb of hidden neurons in 1 layer of LSTM
hidden_size = 50
# Generate sentences in order of transcript
shuffle = False
# Nb of beams for beam beam_search
k = 5


#### GLOBAL VARIABLES ####

transcript_dir = args.transcript_dir
model_dir = args.model_dir
result_dir = args.result_dir
train_all_data = args.train_all_data


#### HELPER FUNCTIONS ####

# A biased (p) coin flip to determine whether a child utterance will be part of train or test set
def is_test_sent(p):
    return True if random.random() < p else False


# Retrieve train and test sets for all child transcripts
def get_data_from_files():
    data = []
    for subdir, dirs, files in os.walk(transcript_dir):
        for file in files:
            if ('.capp' in file):
                textfile = subdir+'/'+file
                with open(textfile,'r') as f :
                    lines = f.readlines()
                train = []
                test = []
                for sent in lines :
                    if '*CHI:' in sent :
                        sent = re.sub('\*[A-Z]+: ', '', sent)
                        # if training on random 60% of child utterances and testing on 40% remaining
                        if train_all_data:
                            if is_test_sent(0.4):
                                test.append(sent)
                            else:
                                train.append(sent)
                        # else train only on child-directed and test on all child utterances
                        else:
                            test.append(sent)
                    else :
                        sent = re.sub('\*[A-Z]+: ', '', sent)
                        train.append(sent)
                data.append((file,train,test))
                # save test and train split in case we need to rerun model
                with open(model_dir+'/train/'+file.split('.capp')[0]+'.train.txt','w') as f :
                    for line in train:
                        f.write(line)
                with open(model_dir+'/test/'+file.split('.capp')[0]+'.test.txt','w') as f :
                    for line in test:
                        f.write(line)

    return data


#### MAIN METHOD : MODEL TRAINING ####

data = get_data_from_files()

for file,train,test in data:
    print('PREPARE DATA FOR: '+file+'\n')
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
    #add LSTM layer
    model.add(LSTM(hidden_size))
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

    # Save trained model for future use
    model.save(str(model_dir+'/'+file.split('.capp')[0]+'_model.h5'))
    # Initialize decoders and get production scores by utterance length using both the greedy and the beam search decoders
    decoders = DecoderGenerator(model,test_generator,k)
    print('CALCULATING PRODUCTION PERFORMANCE METRIC 1...\n')
    results_greedy = decoders.get_performance_bylength("greedy")
    print('CALCULATING PRODUCTION PERFORMANCE METRIC 2...\n')
    results_beam = decoders.get_performance_bylength("beam")

    # save all performance results
    with open(result_dir+'/greedy/'+file.split('.capp')[0]+'.prod_result.csv','w') as f :
        f.write("iter,utterance_length,nb_utterances,produced,production_score"+'\n')
        for length in results_greedy:
            f.write('1,'+str(length)+','+
                            str(results_greedy[length][1])+','+
                            str(results_greedy[length][0])+','+
                            str(results_greedy[length][0]/results_greedy[length][1])+'\n')
        with open(result_dir+'/beam/'+file.split('.capp')[0]+'.prod_result.csv','w') as f :
            f.write("iter,utterance_length,nb_utterances,produced,production_score"+'\n')
            for length in results_beam:
                f.write('1,'+str(length)+','+
                                str(results_beam[length][1])+','+
                                str(results_beam[length][0])+','+
                                str(results_beam[length][0]/results_beam[length][1])+'\n')
    del model
