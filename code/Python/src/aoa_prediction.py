import os
import argparse

#### GET ARGUMENTS FROM PYTHON SCRIPT CALL
def get_args():
    parser = argparse.ArgumentParser(description='Given a set of pretrained models, a list of words and the set of training sentences for the models, will calculate the average surprisal for each word from list accross trainning contexts for each model.')
    script_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument('-w', '--word_list', dest='word_list',
                        action='store', required=True,
                        default=script_dir,
                        help='tsv containing word list for aoa prediction')
    parser.add_argument('-m', '--model_dir', dest='model_dir',
                        action='store', required=True,
                        default=script_dir,
                        help='directory where models and train sets are stored')
    parser.add_argument('-r', '--result_dir', dest='result_dir',
                        action='store', required=True,
                        default=script_dir,
                        help='directory where result matrix will be stored')
    return parser.parse_args()

import csv
from keras.preprocessing.text import Tokenizer
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import numpy as np


args = get_args()



#### GLOBAL VARIABLES ####
word_list = args.word_list
model_dir = args.model_dir
result_dir = args.result_dir

class AoAWord:
    def __init__(self, word, uni_lemma, maxlen, vocab):
        self.word = word
        self.uni_lemma = uni_lemma
        self.id = -1
        self.maxlen = maxlen
        self.contexts = []
        self.surprisals = []
        if word in vocab:
            self.id = vocab[word]

    def get_contexts_surprisals(self, sequences, model):
        if not self.id == -1:
            contexts = []
            for seq in sequences:
                if self.id in seq:
                    context = []
                    for w in seq:
                        if w == self.id:
                            break
                        context.append(w)
                    context.append(self.id)
                    contexts.append(context)
            contexts = pad_sequences(contexts, maxlen=self.maxlen, padding='pre')
            self.contexts= np.array(contexts)
            X, y = self.contexts[:,:-1],self.contexts[:,-1]
            p_pred = model.predict(X)
            for i, prob in enumerate(p_pred):
                self.surprisals.append(-np.log(prob[y[i]]))


    def get_avg_surprisal(self, sequences, model):
        score = 0.0
        if not self.contexts:
            self.get_contexts_surprisals(sequences, model)
        if len(self.surprisals) == 0:
            return "NA"
        else:
            for surprisal in self.surprisals:
                score += surprisal
            score = score/len(self.surprisals)
            return score

def get_model_train_test():
    data = []
    for subdir, dirs, files in os.walk(model_dir):
        for file in files:
            if ('.h5' in file):
                model_file = subdir + '/' + file
                this_model = load_model(model_file)
                childname = file.split('_model.h5')[0]
                trainfile = subdir +'/train/' + childname + '.train.txt'
                testfile = subdir + '/test/' + childname + '.test.txt'
                with open(trainfile, 'r') as f:
                    train = f.readlines()
                with open(testfile, 'r') as f:
                    test = f.readlines()
                data.append((childname, this_model, train, test))
    return data


### MAIN METHOD ###
words = []
with open(word_list) as file:
    reader = csv.reader(file, delimiter='\t')
    reader.__next__()
    for row in reader:
        words.append([row[0], row[1]])

data = get_model_train_test()
aoa_corpus = dict()
for childname, model, train, test in data:
    print('PREPARE DATA FOR: ' + childname + '\n')
    # Get vocabulary
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train + test)
    vocab = tokenizer.word_index
    #print(vocab)
    seqs = tokenizer.texts_to_sequences(train)
    maxlen = max([len(seq) for seq in seqs])
    aoa_words = list()
    for word,uni_lemma in words:
        aoa_words.append(AoAWord(word, uni_lemma, maxlen, vocab))
    aoa_corpus[childname] = aoa_words

        # write all results to the result_dir
    with open(result_dir + '/' + childname + '.aoa_result.csv', 'w') as f:
        f.write("word, uni_lemma, avg_surprisal" + '\n')
        for w in aoa_words:
            f.write(w.word + ',' +
                    w.uni_lemma + ',' +
                    str(w.get_avg_surprisal(seqs, model)) + '\n')
