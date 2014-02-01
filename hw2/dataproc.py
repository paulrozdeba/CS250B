"""
This module contains functions that take in the sentence and label data from
.txt files and returns the data as lists of lists.  The top level list will is
a list of the sentences or corresponding label sequences, and each entry in such
list is a list of the words or labels in that sentence.
"""

def import_sentences(path_to_file):
    """
    path_to_file should be the path and filename of the sentences data.  Returns
    a list of lists.
    """
    sentences = []
    f = open(path_to_file,'r')
    for line in f:
        a = line.replace('\n','')
        b = a.split()
        b += ['LASTWORD']
        b[:0] = ['FIRSTWORD']
        sentences.append(b)
    f.close()
    return sentences

def import_labels(path_to_file):
    """
    path_to_file should be the path and filename of the labels data.  Returns
    a list of lists.
    """
    labels = []
    f = open(path_to_file,'r')
    for line in f:
        a = line.replace('\n','')
        b = a.split()
        b += ['STOP']
        b[:0] = ['START']
        labels.append(b)
    f.close()
    return labels