

import random
import argparse
import codecs
import os
import re
from collections import defaultdict
from pathlib import Path

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
                return
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
        pass

    def forward(self, sequence):
        pass
    ## you do this: Implement the Viterbi algorithm. Given a Sequence with a list of emissions,
    ## determine the most likely sequence of states.

    def viterbi(self, sequence):
        pass
    ## You do this. Given a sequence with a list of emissions, fill in the most likely
    ## hidden states using the Viterbi algorithm.



if __name__ == "__main__" :
    # when moving to submission.py it'll look like HMM.HMM()
    h = HMM()
    h.load('cat')
    print(h.transitions)




