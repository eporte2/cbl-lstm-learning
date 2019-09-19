### DISCLAIMER : The following code is a slightly modified version of McCauley & Christiansen's (2019) openly available code for their Chunk-based Learner (CBL) (see https://github.com/StewartMcCauley/CBL). These modifications have been made in order to keep track of production scores by utterance length. A new flag, --incremental_results, has been implemented in order to optionally test child utterances on the production task incrementally throughout processing when a child sentence is encountered (as in original McCauley & Christiansen production task implementation) or test on a random 40% of child utterances not seen during training/learning (60% remaining percent are seen during training) in order to compare to other model architectures.


import random
import os
import argparse
import copy

# MOD : Added argparse to make code more reusable, regardless of directory architechture
# MOD : Added optional --incremental_results flag
def get_args():
    parser = argparse.ArgumentParser(description='My script')
    script_dir = os.path.dirname(os.path.realpath(__file__))

    parser.add_argument('-d', '--data_dir', dest='data_dir',
                        action='store', required=True,
                        default=script_dir,
                        help='directory where data is stored')
    parser.add_argument('-r', '--result_dir', dest='result_dir',
                        action='store', required=True,
                        default=script_dir,
                        help='directory where results will be stored')
    parser.add_argument('-i', '--incremental_results', dest='incremental_results',
                        action='store_true',
                        help='Test production results incrementally through model training')
    return parser.parse_args()

args = get_args()
incremental_results = args.incremental_results
rootdir = args.data_dir
result_dir = args.result_dir


# MOD : Added helper function to randomly sample from a biased beta distribution. This is used when NOT collecting incremental_results in order to ramdomly select 40% of child utterances during processing for test at the end.
def is_test_sent(p):
    return True if random.random() < p else False

class MeanDict(dict):
    """Dictionary designed to allow easy calculation
    of the running average of its values."""
    def __init__(self):
        self._total = 0.0
        self._count = 0

    def __setitem__(self, k, v):
        if k in self:
            self._total -= self[k]
            self._count -= 1
        dict.__setitem__(self, k, v)
        self._total += v
        self._count += 1

    def __delitem__(self, k):
        v = self[k]
        dict.__delitem__(self, k)
        self._total -= v
        self._count -= 1

    def average(self):
        if self._count:
            return self._total/self._count

class CBL_Model:
    """Chunk-based Learner (CBL) model. McCauley &
    Christiansen (2011, 2014, in press)"""
    def __init__(self):
        self.avg_tp = 0
        self.Unigrams = {}
        self.Bigrams = {}
        self.TPRunAvg = MeanDict()
        self.UniChunks = {}
        self.BiChunks = {}
        # MOD : made spa into dict which collects prod scores for different sent lengths separately
        self.spa = dict()
        # MOD : made spa into dict which collects prod attempts for different sent lengths separately
        self.prod_attempts = dict()
        self.ChunkWordPairs = {}
        self.shallow_parses = []
        # MOD : added new property which saves random 40% of child utterances for test at the end of training if NOT incremental_results, else not used
        self.test_set = []

    def add_unigram(self, word):
        """Updates low-level frequency info for unigrams."""
        if word in self.Unigrams:
            self.Unigrams[word] += 1
        else:
            self.Unigrams[word] = 1

    def add_bigram(self, w1, w2):
        """Updates low-level frequency info for bigrams."""
        if w1+' '+w2 in self.Bigrams:
            self.Bigrams[w1+' '+w2] += 1
        else:
            self.Bigrams[w1+' '+w2] = 1

    def add_unichunk(self, chunk):
        """Updates chunkatory for a given chunk."""
        if chunk in self.UniChunks:
            self.UniChunks[chunk] += 1
        else:
            self.UniChunks[chunk] = 1

    def add_bichunk(self, chunk1, chunk2):
        """Updates frequency info for adjacent chunks,
        allowing the model to calculate TP between chunks."""
        if (chunk1, chunk2) in self.BiChunks:
            self.BiChunks[chunk1, chunk2] += 1
        else:
            self.BiChunks[chunk1, chunk2] = 1

    def add_chunkWordPair(self, w1, w2):
        """Updates frequency info for adjacent words occurring as part
        (or all) of a chunk, supporting discovery of new chunks."""
        if (w1, w2) in self.ChunkWordPairs:
            self.ChunkWordPairs[(w1,w2)] += 1
        else:
            self.ChunkWordPairs[(w1,w2)] = 1

    def calc_btp(self, w1, w2):
        """Calculates transition probabilities between words."""
        return float(self.Bigrams[w1+' '+w2])/float(self.Unigrams[w2])

    def update_chunks(self, linelist):
        """Update the chunkatory on-line."""
        line = ' '.join(linelist)
        chunks = line.split(' || ')
        self.add_unichunk(chunks[-1])

        if (len(chunks) > 1):
            self.add_bichunk(chunks[-2],chunks[-1])

    def calc_btp_chunks(self, chunk1, chunk2):
        """Calculates the TP between two chunks."""
        if (chunk1, chunk2) in self.BiChunks:
            return float(self.BiChunks[(chunk1, chunk2)])/float(self.UniChunks[chunk2])
        else:
            return 0.0

    def end_of_utterance(self,linelist):
        """End-of-line housekeeping. Adds a chunk-to-chunk
        frequency count for the start-of-utterance marker
        leading into the initial chunk in the utterance.
        This is done last for simplicity -- the result
        would be the same if it were done at the beginning."""
        line = ' '.join(linelist)
        chunks = line.split(' || ')
        self.add_bichunk('#',chunks[0])

    def bag_o_chunks(self, line):
        """Yields a bag-of-chunks for the production task."""
        bag = []
        while line:
            for i in range(len(line),0,-1):
                if i == 1:
                    bag.append(line[0])
                    del line[0]
                    break
                elif ' '.join(line[0:i]) in self.UniChunks:
                    bag.append(' '.join(line[0:i]))
                    del line[0:i]
                    break
        return bag

    def upd_run_avg(self, tp, prev_word, item):
        """Update the running average TP."""
        if self.Unigrams[item] > 1:
            if prev_word != '#':
                self.TPRunAvg[prev_word, item] = tp
                self.avg_tp = self.TPRunAvg.average()

    def parse(self, tp, shal_pars, prev_word, item):
            """Shallow parsing operations for this timestep."""
            if tp < self.avg_tp:
                if (prev_word, item) not in self.ChunkWordPairs:
                    self.update_chunks(shal_pars)
                    if prev_word != '#':
                        shal_pars.append('||')
                elif self.ChunkWordPairs[prev_word, item] < 2:
                    self.update_chunks(shal_pars)
                    if prev_word != '#':
                        shal_pars.append('||')
            else:
                self.add_chunkWordPair(prev_word,item)

            return shal_pars

    def big_spa(self, utterance):
        """Implements our modified version of the bag-of-words
        incremental sentence generation task of Chang et al. 2008"""
        # MOD : replaced prod_attempts counter with counter by utterance length below
        line = utterance.split()
        del line[0] #delete speaker tag (*CHI:)
        del line[-1] #delete punctuation

        # MOD : added utterance_length variable to keep track of utterance lengths
        utterance_length = len(line)
        # MOD : added prod_attempts counter by utterance length
        if(utterance_length in self.prod_attempts):
            self.prod_attempts[utterance_length] += 1
        else:
            self.prod_attempts[utterance_length] = 1
            self.spa[utterance_length] = 0

        bag = self.bag_o_chunks(line[:])

        prev_chunk = '#' #set start of utterance marker as first chunk

        produced = []

        while bag: #incrementally produce new utterance chunk-by-chunk
            highest = 0.0
            candidates = []
            for item in bag:
                tp = self.calc_btp_chunks(prev_chunk, item)
                if tp > highest:
                    candidates = [item]
                    highest = tp
                elif tp == highest:
                    candidates.append(item)
            output = candidates[random.randint(0,len(candidates)-1)]

            produced.append(output)
            bag.remove(output)
            prev_chunk = output

        if ' '.join(produced) == ' '.join(line):
            # MOD : increment count at key utterance length in dict
                self.spa[utterance_length] += 1

# MOD: Added new function which uses beam search decoder instead of greedy decoder (as in original code) for big_spa
    def big_spa_beam(self, utterance, k):
        line = utterance.split()
        del line[0] #delete speaker tag (*CHI:)
        del line[-1] #delete punctuation

        utterance_length = len(line)
        if(utterance_length in self.prod_attempts):
            self.prod_attempts[utterance_length] += 1
        else:
            self.prod_attempts[utterance_length] = 1
            self.spa[utterance_length] = 0

        bag = self.bag_o_chunks(line[:])

        beams = [[['#'], bag, 1.0]]

        for i in range(len(bag)):
            candidates = []
            for (context, vocab, score) in beams:
                for v in range(len(vocab)):
                    prev_chunk = context[-1]
                    new_score = score * self.calc_btp_chunks(prev_chunk, vocab[v])
                    new_vocab = vocab[:v]+vocab[(v+1):]
                    new_context = copy.deepcopy(context)
                    new_context.append(vocab[v])
                    candidates.append([new_context, new_vocab, new_score])
            ordered = sorted(candidates, key=lambda prob:prob[2])
            if k < len(ordered):
                beams = ordered[:k]
            else:
                beams = ordered

        result = 0
        for context, vocab, score in beams:
            if ' '.join(context[1:]) == ' '.join(line):
                result=1
        if result == 1:
            self.spa[utterance_length] += 1


    def line_process(self, line):
        del line[0] #remove speaker tag
        del line[-1] #remove punctuation
        prev_chunk = '#'#set start of utterance marker
        prev_word = '#'#set start of utterance marker
        self.add_unigram(prev_word)
        self.add_unichunk(prev_chunk)

        shal_pars = []

        for item in line:
            """Process incrementally, on-line --
            each word represents a time-step."""
            #update low-level freq info
            self.add_unigram(item)
            self.add_bigram(prev_word, item)

            #calculate running average TP
            tp = self.calc_btp(prev_word, item)
            self.upd_run_avg(tp, prev_word, item)

            #shallow parse the utterance
            shal_pars = self.parse(tp, shal_pars, prev_word, item)

            shal_pars.append(item)
            prev_word = item

        #end of utterance has been encountered, so the final
        #chunk can now be updated online
        self.update_chunks(shal_pars)

        #house-keeping operation
        self.end_of_utterance(shal_pars)

        #Add the on-line shallow parse to a list --
        #this list can then be compared to the gold standard
        #shallow parses.
        self.shallow_parses.append(shal_pars)


    def process(self, utterance):
        """Process an utterance on-line, treating each
        word as a separate timestep."""
        line = utterance.split()
        #if list has less than 3 elements, it is empty b/c
        #auto-cleaning removed a non-speech sound, etc.
        if len(line) < 3:
            return

        #if it's a child utterance, run it through BIG-SPA
        if line[0] == '*CHI:':
            if len(line) > 3:
                # MOD : Given new flag, if not incremental_results,
                if not incremental_results:
                    if is_test_sent(0.4):
                        self.test_set.append(utterance)
                    else:
                        self.line_process(line)
                else:
                    self.big_spa(utterance)
                    self.line_process(line)

        else:
            self.line_process(line)


for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        if ('.capp' in file):
            textfile = subdir+'/'+file
            f= open(textfile,'r')
            lines = f.readlines()
            f.close()

            production_scores = []

            for i in range(0,10):#define number of iterations (for averaging)

                model = CBL_Model()

                for line in lines:
                    #Ensure age-markers aren't treated as utterances
                    #Ex. "*AGEIS: 25 .\n" <-- child at 25 months
                    if '*AGEIS:' in line:
                        continue
                    #Process the utterance incrementally and on-line
                    model.process(line)

                #production_scores.append(float(model.spa)/float(model.prod_attempts))
                # MOD: production_scores  now keeps track of score for each utterance length
                if not incremental_results:
                    for utterance in model.test_set:
                        #model.big_spa(utterance)
                        model.big_spa_beam(utterance, 5)
                results = dict()
                for utterance_length in model.spa:
                    results[utterance_length] = [str(model.prod_attempts[utterance_length]), str(model.spa[utterance_length]), str(float(model.spa[utterance_length])/float(model.prod_attempts[utterance_length]))]
                production_scores.append(results)

            #Print out the model's average score on the production task
            #MOD : Removed print statement and instead write all iteration results to csv file
            print(textfile)

            #output record of the on-line shallow parses to a new file for scoring
            #against gold-standard shallow parses
            #MOD : changed outputfile to save all performance of production task results for each iteration for each sentence length
            i = 1
            outputfile = open(result_dir+'/'+file.split('.capp')[0]+'.beam_prod_result.csv','w')
            outputfile.write("iter,utterance_length,nb_utterances,produced,production_score"+'\n')
            for results in production_scores:
                for utterance_length in results:
                    outputfile.write(str(i)+','+str(utterance_length)+','+results[utterance_length][0]+','+results[utterance_length][1]+','+results[utterance_length][2]+'\n')
                i+=1
