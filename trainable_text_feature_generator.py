import numpy as np
import multiprocessing
from eval_tools import load_wordvec
import cPickle

def text_feature_generation(user_feature_sequence):
    text_vec = load_wordvec()
    useful_vec = {}
    print ("useful data length",len(user_feature_sequence))
    count = 0
    for u in user_feature_sequence.keys():
        features = user_feature_sequence[u]
        for traj_fea in range(len(features)):
            useful_word_sample = []
            for i in range(len(features[traj_fea][2])):
                text = features[traj_fea][2][i]
                words_key = []
                if not text == 0:
                    words = text.split(' ')
                    for w in words:
                        if (text_vec.has_key(w)) & (not useful_vec.has_key(w)):
                            useful_vec[w] = text_vec[w]
                        if useful_vec.has_key(w):
                            words_key.append(w)
                else: print "Text == 0"
                useful_word_sample.append(words_key)
                if len(words_key) ==0 : print ("record empty useful words")
            user_feature_sequence[u][traj_fea].append(useful_word_sample)
        # if count % 20 ==0:
        #     print user_feature_sequence[u]

    return user_feature_sequence,useful_vec

if __name__ == "__main__":
    text_feature_generation()
