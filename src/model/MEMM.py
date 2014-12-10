'''
@author: liorab
input -
    history:4-tuple - t_i-2,t_i-1,sentence of length n
    and the current position i;
    y_i = current word tag; 
    v vector (feature score vector);
    
'''
from collections import namedtuple
import string
import features
import numpy as np
import sys
import math
from scipy.optimize import fmin_l_bfgs_b
import scipy
import time

class MEMM():

    
    def __init__(self):
        self.sentences_dict = {}
        self.POS_tags_dict = {}
        self.word_tuples_dict = {} #keep all the tuples created for each word in the train set. key- word number, value - tuple 
        self.num_of_feature_functions = 2
        self.features_res_vector_for_curr_tuple_and_tag = {}
        self.tuple_probabilities_dict = {}
        self.data_path = r"C:\study\technion\MSc\3rd_semester\NLP\hw1\sec2-21"
        self.input_sentence_file = self.data_path+r"\sec2-21.words" 
        self.input_tags_file = self.data_path+r"\sec2-21.pos" 
        self.POS_set = []
        self.sum_of_expected_count = np.zeros(self.num_of_feature_functions)
        self.sum_of_empirical_count = np.zeros(self.num_of_feature_functions)
        self.regularization_lambda = 0.9
        self.v_optimal = np.zeros(self.num_of_feature_functions) ;
     
    def read_input_sentences_for_train(self):
        sentences_file = open(self.input_sentence_file,"rb").read().strip()
        for i, line in enumerate(sentences_file.split('\n')):
            if i < 2001:   
                self.sentences_dict[i] = line.split()
   
    def read_input_sentences_for_test(self,input_file):
        sentences_file = open(self.input_sentence_file,"rb").read().strip()
        for i, line in enumerate(sentences_file.split('\n')):
            if i > 5000 and i< 10001:   
                self.sentences_dict[i] = line.split()
    
    def product(self,vec1,vec2):
        return sum(p*q for p,q in zip(vec1, vec2))
    
    def remove_punctuation(self,sentence_word_list):
        exclude = set(string.punctuation)
        clean_sentence = []
        clean_word = ""
        for word in sentence_word_list:
            for ch in exclude:
                if ch in word:
                    clean_word = word.replace(ch,"")
            if not clean_word is "":
                clean_sentence.append(clean_word)
            else:
                clean_sentence.append(word)
            clean_word = ""
        return clean_sentence
                    
    def read_input_POS_tags_for_train(self):
        tags_file = open(self.input_tags_file,"rb").read().strip()
        POS_set = set()
        for i, line in enumerate(tags_file.split('\n')):
            if i < 2001:     
                POS_list = line.split()
                self.POS_tags_dict[i] = POS_list
                for POS in POS_list:
                    POS_set.add(POS)
        for tag in POS_set:   
            self.POS_set.append(tag)
              
    def read_input_POS_tags_for_test(self):
        tags_file = open(self.input_tags_file,"rb").read().strip()
        for i, line in enumerate(tags_file.split('\n')):
            if i > 5001 and i< 10001:   
                self.POS_tags_dict[i] = line.split()
        
    def create_sentence_word_tuple(self,sentence,sentence_index,word_index):
        sentence_tuple = namedtuple('sentence_tuple','tag_position_2,tag_position_1, sentence_index, curr_pos, curr_tag')
        if word_index == 0:
            curr_sentence_word_tuple = sentence_tuple("*","*",sentence_index,word_index,self.POS_tags_dict[sentence_index][word_index])
        elif word_index == 1:
            curr_sentence_word_tuple = sentence_tuple("*",self.POS_tags_dict[sentence_index][word_index-1],sentence_index,word_index,self.POS_tags_dict[sentence_index][word_index])
        else:
            curr_sentence_word_tuple = sentence_tuple(self.POS_tags_dict[sentence_index][word_index-2],self.POS_tags_dict[sentence_index][word_index-1],sentence_index,word_index,self.POS_tags_dict[sentence_index][word_index])
        return curr_sentence_word_tuple 

    def compute_Likelihood(self):
        def likelihood_func(*args):
            print "    computing L"
            try:
                t1 = time.clock();
                features_functions = features.feature_functions(self.num_of_feature_functions)
                likelihood = 0
                for (tuple_index,tuple_feature_vec) in self.features_res_vector_for_curr_tuple_and_tag.items():
                    try:
                        likelihood = likelihood + self.product(args[0], tuple_feature_vec)
                        # calculate the inner sum of the second term
                        expected_count = 0
                        for tag in self.POS_set:
                            feature_vec = features_functions.apply_feature_functions(self.word_tuples_dict[tuple_index], tag, self.sentences_dict[self.word_tuples_dict[tuple_index].sentence_index])
                            expected_count += math.exp(self.product(feature_vec, args[0]))
                        likelihood = likelihood - math.log(expected_count)
                    except Exception as err: 
                        sys.stderr.write("problem in tuple_index"+str(tuple_index))     
                        print err.args      
                        print err
                v_vector_square = sum(math.pow(v_k, 2) for v_k in args[0])
                likelihood = likelihood - ((self.regularization_lambda / 2) * v_vector_square)
                t2 = time.clock()
                print "    v-norm:", v_vector_square
                print "    time to calc L:" , t2 - t1
            except Exception as err: 
                sys.stderr.write("problem likelihood_func")     
                print err.args      
                print err
            return -(likelihood)
        return likelihood_func
    
    def compute_likelihood_gradient(self):
        def gradient_likelihood(*args):
            print "computing gradientL"
            t1 = time.clock();
            try:
                features_functions = features.feature_functions(self.num_of_feature_functions)
                likelihood_derivative = [0] * self.num_of_feature_functions
                for (tuple_index,tuple_feature_vec) in self.features_res_vector_for_curr_tuple_and_tag.items():
                    #calc the probabilities first
                    P = [0] * len(self.POS_set)
                    for tag in range(0,len(self.POS_set)):
                        feature_vec = features_functions.apply_feature_functions(self.word_tuples_dict[tuple_index],tag,self.sentences_dict[self.word_tuples_dict[tuple_index].sentence_index])
                        inner_product = sum(self.product(feature_vec, args))
                        P[tag] = math.exp(inner_product)
                    normalization = sum(P);
                    for k in range(0,self.num_of_feature_functions):
                        # empirical counts 
                        likelihood_derivative[k] = likelihood_derivative[k] + tuple_feature_vec[k]
                        # expected count
                        expected_count = 0
                        for tag in self.POS_set:
                            features_all_tags_res = features_functions.apply_feature_functions(self.word_tuples_dict[tuple_index],tag,self.sentences_dict[self.word_tuples_dict[tuple_index].sentence_index])
                            f_k = features_all_tags_res[k]
                            expected_count = expected_count + (f_k * P[self.POS_set.index(tag)]/normalization)
                        likelihood_derivative[k] = likelihood_derivative[k] - expected_count
                        likelihood_derivative[k] = likelihood_derivative[k] - (self.regularization_lambda*args[0][k])
                max_likelihood_derivative = map(lambda x: -1 * x, likelihood_derivative) 
                t2 = time.clock();
                print "    time to calc gradient:" , t2 - t1
            except Exception as err: 
                sys.stderr.write("problem gradient_likelihood")     
                print err.args      
                print err
            return scipy.array(max_likelihood_derivative)
        return gradient_likelihood;
                       
    
    def optimize_v(self):
        print "optimizing v "
        try:
            x0 = [0]*self.num_of_feature_functions
            self.v_optimal = fmin_l_bfgs_b(self.compute_Likelihood(), x0, self.compute_likelihood_gradient())
            print self.v_optimal
        except Exception as err: 
            sys.stderr.write("problem optimize_v")     
            print err.args      
            print err
            
    def compute_features_on_all_words(self): 
        try:
            t1 = time.clock();
            features_functions = features.feature_functions(self.num_of_feature_functions)
            for (index,curr_sentence_word_tuple) in self.word_tuples_dict.items():
                self.features_res_vector_for_curr_tuple_and_tag[index] = features_functions.apply_feature_functions(curr_sentence_word_tuple,curr_sentence_word_tuple.curr_tag,self.sentences_dict[curr_sentence_word_tuple.sentence_index])
            t2 = time.clock()
            print "finished compute_features_on_all_words in", t2 - t1
        except Exception as err: 
            sys.stderr.write("fell in compute_features_on_all_words")     
            print err.args      
            print err               
    
    def create_tuples_from_all_words(self):
        #create a tuple from all the words in the sentences, keep in memory, and train on each word in a pipeline
        try:
            word_counter = 0
            for sentence_index in range(0,len(self.sentences_dict.values())):
                sentence = self.sentences_dict[sentence_index]
                for word_index in range(0,len(sentence)):
                    try:
                        curr_sentence_word_tuple = self.create_sentence_word_tuple(sentence,sentence_index,word_index)
                        self.word_tuples_dict[word_counter] = curr_sentence_word_tuple
                        word_counter +=1
                    except Exception as err: 
                        sys.stderr.write("problem in creating word tuple")     
                        print err.args      
                        print err         
        except Exception as err: 
            sys.stderr.write("problem in create_tuples_from_all_words")     
            print err.args      
            print err                     
        
        print "finished create_tuples_from_all_words"
        
    def train_memm(self):
        """go over all the 5000 WSJ sentence
           create x - tuple (history), and the i'th word tag, and compute likelihood
           the output is the v_vector - weights for the features
        """
        self.read_input_sentences_for_train()
        self.read_input_POS_tags_for_train()
        self.create_tuples_from_all_words()
        self.compute_features_on_all_words()
        self.optimize_v()    
    
    def calculate_p_with_optimal_weight_vector(self):
        features_functions = features.feature_functions()
        for (tuple_index,tuple_feature_vec) in self.features_res_vector_for_curr_tuple_and_tag.items():
            prob_nominator = math.exp(self.product(tuple_feature_vec,self.v_optimal))
            prob_normalization_factor = 0
            for tag in self.POS_set:
                prob_normalization_factor +=  math.exp(features_functions.apply_feature_functions(self.word_tuples_dict[tuple_index], tag, self.sentences_dict[self.word_tuples_dict[tuple_index].sentence_index]))
            self.tuple_probabilities_dict[tuple_index] = float(prob_nominator/prob_normalization_factor)
                
def main():
    try:
        memm_POS_tag = MEMM()
        memm_POS_tag.train_memm()
    except Exception as err: 
        sys.stderr.write("problem")     
        print err.args      
        print err
    
if __name__ == '__main__':
    main()     