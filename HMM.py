import random
import argparse
import codecs
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy
from torch.distributed.tensor import empty


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
        matrix[0][0] = 1.0
        # Iterating through columns -> sequence
        for i in range(1, len(sequence) + 1):
            if i == 1 :
                # For initial step since it's not reliant on previous state, only probability of start state
                for idx, state in enumerate(self.transitions["#"].keys()):
                    # Probability = P(observation | state) -> the emission probability *
                    #               P(state | "#") -> the start state probability *
                    #               P("#") -> always 1.0
                    try:
                        prob = float(self.emissions[state][sequence[i - 1]]) * float(self.transitions["#"][state])
                        # if KeyError, means transition or emission is not reachable therefore is 0
                    except KeyError:
                        # Means no transitions or emission exists
                        prob = 0
                    matrix[idx + 1][i] = prob
            else :
                # Iterating through rows (states) -> "#", "happy", "grumpy", "hungry"
                for idx1, state in enumerate(self.transitions["#"].keys()) :
                    prob = 0
                    # Iterating through states again for prev_state accessibility
                    for idx2, state2 in enumerate(self.transitions["#"].keys()) :
                        # Probability = P(observation | state) -> the emission probability *
                        #               P(state | prev_state) -> the transition probability *
                        #               P(prev_state) -> the previous row entry in the matrix
                        try:
                            prob += matrix[idx2 + 1][i-1] * float(self.transitions[state2][state]) * float(self.emissions[state][sequence[i-1]])
                        # if KeyError, means transition or emission is not reachable therefore is 0
                        except KeyError:
                            prob += 0
                    matrix[idx1 + 1][i] = prob
        max_idx = 0
        max_val = 0
        for idx, row in enumerate(matrix):
            if row[len(row) - 1] > max_val:
                max_val = row[len(row) - 1]
                max_idx = idx
        return list(self.transitions["#"].keys())[max_idx - 1]

    def viterbi(self, sequence):
        # Initializing a matrix with sequence + 1 columns to account for sequence and #
        # and with transitions["#"].keys() + 1 rows to account for starting states and "#"
        matrix = [[float(0) for j in range(len(sequence) + 1)] for i in range(len(self.transitions["#"].keys()) + 1)]
        backpointers = [[float(0) for j in range(len(sequence) + 1)] for i in range(len(self.transitions["#"].keys()) + 1)]
        matrix[0][0] = 1.0
        # Iterating through columns -> sequence
        for i in range(1, len(sequence) + 1):
            if i == 1:
                # For initial step since it's not reliant on previous state, only probability of start state
                for idx, state in enumerate(self.transitions["#"].keys()):
                    # Probability = P(observation | state) -> the emission probability *
                    #               P(state | "#") -> the start state probability *
                    #               P("#") -> always 1.0
                    try:
                        prob = float(self.emissions[state][sequence[i - 1]]) * float(self.transitions["#"][state])
                    # if KeyError, means transition or emission is not reachable therefore is 0
                    except KeyError:
                        prob = 0
                    matrix[idx + 1][i] = prob
            else:
            # Iterating through rows (states) -> "#", "happy", "grumpy", "hungry"
                for idx1, state in enumerate(self.transitions["#"].keys()):
                    max_prob = 0
                    # Iterating through states again for prev_state accessibility
                    for idx2, state2 in enumerate(self.transitions["#"].keys()):
                        try :
                            curr_prob = matrix[idx2 + 1][i - 1] * float(self.transitions[state2][state]) * float(self.emissions[state][sequence[i - 1]])
                        # if KeyError, means transition or emission is not reachable therefore is 0
                        except KeyError:
                            curr_prob = 0
                        if curr_prob > max_prob:
                            max_prob = curr_prob
                            backpointers[idx1+1][i] = idx2 + 1
                    matrix[idx1 + 1][i] = max_prob
        states = []
        min_value = float("inf")
        min_row = -1
        last_column = len(matrix[0]) - 1
        # Finds the initial minimum value from the last column
        for idx, row in enumerate(backpointers[1:]):
            if (row[last_column] != 0) and row[last_column] < min_value:
                min_value = row[last_column]
                min_row = idx + 1
        # Both value and row get added because value doesn't give us which row the last value occurred at
        states.append(min_row)
        states.append(int(min_value))
        # Iterates backwards using the min_value we found to backtrace the indices
        while last_column > 1:
            last_column -= 1
            min_value = int(backpointers[int(min_value)][last_column])
            states.append(min_value)
        states.reverse()
        emits = []
        # Ignoring the 0 as it tells us nothing, the entire first (second(?)) row is all zeros
        for state in states[1:] :
            emits.append(list(self.transitions["#"].keys())[state - 1])
        return emits

def validate(basename, file, type) :
    h = HMM()
    h.load(basename)
    if not Path(file).is_file() :
        with open(file, 'w') as f:
            toks = " ".join(h.generate(20))
            f.write(toks)
    with open(file) as f :
        for line in f :
            if len(line) != 1 :
                lines = line.split(" ")
                tokens = [item.rstrip('\n') for item in lines if item != ('' or '\n')]
                match type :
                    case "forward" :
                        print(f"The most likely current state is %s" % h.forward(tokens))
                    case "viterbi" :
                        print(f"The most likely sequence of hidden states for the sequence of observations \"%s\" is \"%s\"" % (" ".join(tokens), " ".join(h.viterbi(tokens))))


if __name__ == "__main__" :
    # when moving to submission.py it'll look like HMM.HMM()
    parser = argparse.ArgumentParser(
                    prog = 'HMM.py',
                    description = 'Performs Hidden Markov Models on a sequence of states')
    parser.add_argument('basename', help = "The basename of the .emit and .trans file to process")
    parser.add_argument('--generate', metavar = "N", help = "Generates a random sequence with N random observations")
    parser.add_argument('--forward', metavar = "outfile", help = "Runs the Forward Algorithm on the observations in this file, if given a file that doesn't exist, will create one with default 20 observations")
    parser.add_argument('--viterbi', metavar = "outfile", help = "Runs the Viterbi Algorithm on the observations in this file, if given a file that doesn't exist, will create one with default 20 observations")
    args = parser.parse_args()


    if args.forward :
        validate(args.basename, args.forward, "forward")
    if args.viterbi :
        validate(args.basename, args.viterbi, "viterbi")
    # if args.generate :
    #     with open(args.basename + "_sequence.obs", 'w') as f :
    #         h.load(args.basename)
    #         toks = " ".join(h.generate(int(args.generate)))
    #         f.write(toks)
    # if args.forward:
    #     if Path(args.forward).is_file() :
    #         h.load(args.basename)
    #         with open(args.forward) as f :
    #             for line in f :
    #                 if len(line) != 1:
    #                     lines = line.split(" ")
    #                     tokens = [item.rstrip('\n') for item in lines if item != ('' or '\n')]
    #                     print(f"The most likely current state is %s" % h.forward(tokens))
    #     else :
    #         with open(args.forward, 'w') as f :
    #             h.load(args.basename)
    #             toks = " ".join(h.generate(30))
    #             f.write(toks)
    #         h.forward(args.forward)
    # if args.viterbi :
    #     if Path(args.viterbi).is_file() :
    #         h.load(args.basename)
    #         with open(args.viterbi) as f :
    #             for line in f :
    #                 if len(line) != 1:
    #                     lines = line.split(" ")
    #                     tokens = [item.rstrip('\n') for item in lines if item != ('' or '\n')]
    #                     print(tokens)
    #                     h.viterbi(tokens)
