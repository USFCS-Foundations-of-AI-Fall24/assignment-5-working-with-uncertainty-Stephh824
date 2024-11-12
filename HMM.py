

import random
import argparse
import codecs
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from decimal import Decimal

import numpy

# Sequence - represents a sequence of hidden states and corresponding
# output variables.

class Sequence:
    def __init__(self, stateseq, outputseq):
        self.stateseq  = stateseq   # sequence of states
        self.outputseq = outputseq  # sequence of outputs
    def __str__(self):
        return ' '.join(self.stateseq)+'\n'+' '.join(self.outputseq)+'\n'
    def __repr__(self):
        return self.__str__()
    def __len__(self):
        return len(self.outputseq)

# HMM model
class HMM:
    def __init__(self, transitions={}, emissions={}):
        """creates a model from transition and emission probabilities
        e.g. {'happy': {'silent': '0.2', 'meow': '0.3', 'purr': '0.5'},
              'grumpy': {'silent': '0.5', 'meow': '0.4', 'purr': '0.1'},
              'hungry': {'silent': '0.2', 'meow': '0.6', 'purr': '0.2'}}"""
        self.transitions = transitions
        self.emissions = emissions

    # Loading the contents of the basename to add to the proper attribute
    def load(self, basename):
        types = (".trans", ".emit")
        for ftype in types :
            tdict = defaultdict(dict)
            path = Path(basename + ftype)
            # Validating that the path exists
            if not path.is_file() :
                print("Please enter a valid basename for your .emit and .trans files")
                sys.exit()
            # Opening the file and iterating through the lines
            with path.open() as f :
                for line in f :
                    line = line.split(" ")
                    tdict[line[0]][line[1]] = line[2].rstrip('\n')
                # Adding to the proper attribute
                match ftype:
                    case ".trans" :
                        self.transitions = tdict
                    case ".emit" :
                        self.emissions = tdict

    def generate(self, n):
        """return an n-length Sequence by randomly sampling from this HMM."""
        obs = list()
        curr_state = random.choices(list(self.transitions["#"].keys()), [float(item) for item in self.transitions["#"].values()])[0]
        for i in range(0, n) :
            curr_emit = random.choices(list(self.emissions[curr_state].keys()), [float(item) for item in self.emissions[curr_state].values()])[0]
            obs.append(curr_emit)
            curr_state = random.choices(list(self.transitions[curr_state].keys()), [float(item) for item in self.transitions[curr_state].values()])[0]
        return obs

    def forward(self, sequence):
        # Initializing a matrix with sequence + 1 columns to account for sequence and #
        # and with transitions["#"].keys() + 1 rows to account for starting states and "#"
        matrix = [[float(0) for j in range(len(sequence) + 1)] for i in range(len(self.transitions["#"].keys()) + 1)]
        matrix[0][0] = 1
        # Iterating through columns -> sequence
        for i in range(1, len(sequence) + 1):
            if i == 1 :
                # For initial step since it's not reliant on previous state, only probability of start state
                for idx, state in enumerate(self.transitions["#"].keys()):
                    # Probability = P(observation | state) -> the emission probability *
                    #               P(state | "#") -> the start state probability *
                    #               P("#") -> always 1.0
                    prob = float(self.emissions[state][sequence[i - 1]]) * float(self.transitions["#"][state])
                    matrix[idx + 1][i] = prob
            else :
                # Iterating through rows -> "#", "happy", "grumpy", "hungry"
                for idx1, state in enumerate(self.transitions["#"].keys()) :
                    prob = 0
                    # Iterating through states again for prev_state accessibility
                    for idx2, state2 in enumerate(self.transitions["#"].keys()) :
                        # Probability = P(observation | state) -> the emission probability *
                        #               P(state | prev_state) -> the transition probability *
                        #               P(prev_state) -> the previous row entry in the matrix
                        prob += matrix[idx2 + 1][i-1] * float(self.transitions[state2][state]) * float(self.emissions[state][sequence[i-1]])
                    matrix[idx1 + 1][i] = prob
        return matrix

    def viterbi(self, sequence):
        pass
    ## You do this. Given a sequence with a list of emissions, fill in the most likely
    ## hidden states using the Viterbi algorithm.



if __name__ == "__main__" :
    # when moving to submission.py it'll look like HMM.HMM()
    parser = argparse.ArgumentParser(
                    prog = 'HMM.py',
                    description = 'Performs Hidden Markov Models on a sequence of states')
    parser.add_argument('basename', help = "The basename of the .emit and .trans file to process")
    parser.add_argument('--generate', metavar = "N", help = "Generates a random sequence with N random observations")
    parser.add_argument('--forward', metavar = "outfile", help = "Runs the forward algorithm on the observations in this file, if given a file that doesn't exist, will create one with default 30 observations")
    args = parser.parse_args()

# If both generate and forward, do generate write to a file
# and then do forward on that file
# If just forward, auto-generate a 20 n sequence
# If just generate -> generate
    h = HMM()
    if args.generate :
        with open(args.basename + "_sequence.obs", 'w') as f :
            h.load(args.basename)
            toks = " ".join(h.generate(int(args.generate)))
            f.write(toks)
    if args.forward :
        if Path(args.forward).is_file() :
            h.load(args.basename)
            with open(args.forward) as f :
                lines = f.readlines()
                lines = " ".join(lines).split(' ')
            print(h.forward(lines))
        else :
            with open(args.forward, 'w') as f :
                h.load(args.basename)
                toks = " ".join(h.generate(30))
                f.write(toks)
            h.forward(args.forward)
