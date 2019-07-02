import os
import argparse

#### GET ARGUMENTS FROM PYTHON SCRIPT CALL
def get_args():
    parser = argparse.ArgumentParser(description='Given a set of pretrained models, will calculate the production score for each model for each utterance length (< 17 words) using beam search and store them in the result directory')
    script_dir = os.path.dirname(os.path.realpath(__file__))

    parser.add_argument('-m', '--model_dir', dest='model_dir',
                        action='store', required=True,
                        default=script_dir,
                        help='directory where models are stored')
    parser.add_argument('-r', '--result_dir', dest='result_dir',
                        action='store', required=True,
                        default=script_dir,
                        help='directory where results are written')
    return parser.parse_args()

args = get_args()

from keras.preprocessing.text import Tokenizer
from keras.models import load_model

from my_data_generator import DataGenerator
from my_decoder_generator import DecoderGenerator


#### HYPERPARAMETERS ####
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
model_dir = args.model_dir
result_dir = args.result_dir



#### HELPER FUNCTIONS ####

# returns model name (childname), model, and corresponding test and train
# data sets previously stored during model training for all available models in
# model_dir
def get_model_train_test():
    data = []
    for subdir, dirs, files in os.walk(model_dir):
        for file in files:
            if ('.h5' in file):
                model_file = subdir + '/' + file
                this_model = load_model(model_file)
                childname = file.split('_model.h5')[0]
                trainfile = '../train/' + childname + '.train.txt'
                testfile = '../test/' + childname + '.test.txt'
                with open(trainfile, 'r') as f:
                    train = f.readlines()
                with open(testfile, 'r') as f:
                    test = f.readlines()
                data.append((childname, this_model, train, test))
    return data


#### MAIN METHOD : GET PRODUCTION SCORES ####

# get models and test/train sets
data = get_model_train_test()

for childname, this_model, train, test in data:
    print('PREPARE DATA FOR: ' + childname + '\n')
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

    print('vocab_size = ' + str(vocab_size))
    print('train_maxlen = ' + str(maxlen))
    print('INITIALIZE DATA GENERATORS...\n')

    # Create data generator for test sequences
    test_generator = DataGenerator(seqs=test_seqs,
                                   vocab=vocab,
                                   vocab_size=vocab_size,
                                   maxlen=maxlen,
                                   batch_size=batch_size,
                                   shuffle=shuffle)

    print('CALCULATING PRODUCTION PERFORMANCE BEAMS...\n')
    # calculate performance using a "beam" search decoder instead of a "greedy"
    # initialize decoders
    decoders = DecoderGenerator(this_model,test_generator,k)
    results_beam = decoders.get_performance_bylength("beam")

    # write all results to the result_dir
    with open(result_dir + '/' + childname + '.prod_result.csv', 'w') as f:
        f.write("iter,utterance_length,nb_utterances,produced,production_score" + '\n')
        for length in results_beam:
            f.write('1,' + str(length) + ',' +
                    str(results_beam[length][1]) + ',' +
                    str(results_beam[length][0]) + ',' +
                    str(results_beam[length][0] / results_beam[length][1]) + '\n')
