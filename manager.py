#!/usr/bin/python3

import subprocess
import itertools


def findsubsets(lst, size):
    return [list(a) for a in itertools.combinations(lst, size)]

ctypes = ['BLCA', 'BRCA', 'COAD', 'ESCA', 'HNSC', 'KIRC', 'KIRP', 'LC', 'LIHC', 'LUAD', 'LUSC', 'PRAD', 'THCA', 'UCEC']

def run(ctypes):

    # for i in range(len(ctypes)):
    for i in range(3, 4):
        subsets = findsubsets(ctypes, size=i)
        for subset in subsets:

            c_line = []
            for cancer in subset:
                c_line += ["-c"] + [cancer]

            depth = 7 if i > 3 else 3
            subprocess.check_call(["tbc", "-ver", "-vis", "-d", str(depth)] + c_line)



run(ctypes)
