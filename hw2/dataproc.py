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
    
def labels_as_ints(path_to_file):
    """
    path_to_file should be the path and filename of the labels data.  Returns
    a list of lists with corresponding tags as integer equivalents.
    """
    key_dict = dict([('SPACE',2),('COMMA',3),('PERIOD',4),('COLON',5),('EXCLAMATION_POINT',6),('QUESTION_MARK',7)])
    labels = []
    f = open(path_to_file,'r')
    for line in f:
        a = line.replace('\n','')
        b = a.split()
        c = [key_dict.get(item,item) for item in b]
        c += [1]
        c[:0] = [0]
        labels.append(c)
    f.close()
    return labels