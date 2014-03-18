import numpy as np
import tree_maker as tm
import math

def word_ranks(vocab,Wlabel):
    v_size = np.shape(vocab)[0]
    score_list = []
    meaning_list = []
    word_list = []
    for i in range(1,v_size):
        label = tm.make_d(vocab[i,:],Wlabel)
        score_list.append(math.log(label[0]/label[1]))
        meaning_list.append(vocab[i,:])
        word_list.append([i])
    return word_list,score_list,meaning_list



def phrase_ranking(neg_examples,pos_examples,vocab,W1,W2,b1,b2,Wlabel,normalized):
    #first loop over neg_examples
    phrase_list = []
    score_list = []
    meaning_list = []
    label = np.array([0.0,1.0])
    for example in neg_examples:
        sub_phrase_list = []
        sub_score_list = []
        sub_meaning_list = []
        N = len(example)
        num_nodes = 2*N - 1
        tree_stuff = tm.build_tree(example,label,vocab,W1,W2,b1,b2,Wlabel,normalized)
        tree_info = tree_stuff[0]
        tree_meanings = tree_stuff[1]
        for i in range(N):
            sub_phrase_list.append([example[i]])
        for k in range(N,num_nodes):
            left = tree_info[0,k]
            right = tree_info[1,k]
            phrase = sub_phrase_list[left] + sub_phrase_list[right]
            sub_phrase_list.append(phrase)
        for j in range(num_nodes):
            predicted = tm.make_d(tree_meanings[:,j],Wlabel)
            score = math.log(predicted[0]/predicted[1])
            sub_score_list.append(score)
            sub_meaning_list.append(tree_meanings[:,j])
        score_list += sub_score_list
        meaning_list += sub_meaning_list
        phrase_list += sub_phrase_list
    label = np.array([1.0,0.0])
    for example in pos_examples:
        sub_phrase_list = []
        sub_score_list = []
        sub_meaning_list = []
        N = len(example)
        num_nodes = 2*N - 1
        tree_stuff = tm.build_tree(example,label,vocab,W1,W2,b1,b2,Wlabel,normalized)
        tree_info = tree_stuff[0]
        tree_meanings = tree_stuff[1]
        for i in range(N):
            sub_phrase_list.append([example[i]])
        for k in range(N,num_nodes):
            left = tree_info[0,k]
            right = tree_info[1,k]
            phrase = sub_phrase_list[left] + sub_phrase_list[right]
            sub_phrase_list.append(phrase)
        for j in range(num_nodes):
            predicted = tm.make_d(tree_meanings[:,j],Wlabel)
            score = math.log(predicted[0]/predicted[1])
            sub_score_list.append(score)
            sub_meaning_list.append(tree_meanings[:,j])
        score_list += sub_score_list
        meaning_list += sub_meaning_list
        phrase_list += sub_phrase_list
    return phrase_list,score_list,meaning_list

def remove_singles(phrase_list,score_list,meaning_list):
    reduced_phrases = []
    reduced_scores = []
    reduced_meanings = []
    for i in range(len(phrase_list)):
        if (len(phrase_list[i]) > 1):
            reduced_phrases.append(phrase_list[i])
            reduced_scores.append(score_list[i])
            reduced_meanings.append(meaning_list[i])
    return reduced_phrases,reduced_scores,reduced_meanings

def no_duplicates(phrase_list,score_list,meaning_list):
    filtered_phrases = []
    filtered_scores = []
    filtered_meanings = []
    for i in range(len(phrase_list)):
        phrase = phrase_list[i]
        if phrase not in filtered_phrases:
            filtered_phrases.append(phrase_list[i])
            filtered_scores.append(score_list[i])
            filtered_meanings.append(meaning_list[i])
    return filtered_phrases,filtered_scores,filtered_meanings
    
def num_to_word(fullvocab,phrase):
    inwords = []
    for i in range(len(phrase)):
        index = phrase[i]
        word = str(fullvocab['words'][0][index - 1][0])
        inwords.append(word)
    return inwords
    

