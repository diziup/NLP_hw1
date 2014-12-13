'''
@author: liorab
    history:4-tuple - t_i-2,t_i-1,word    
'''
# import features
import numpy as np
import sys
import math
from scipy.optimize import fmin_l_bfgs_b
import scipy
import time
import sentence
import collections
import features

class MEMM():
    
    def __init__(self):
        
        self.num_of_feature_functions = 2
        self.tuple_probabilities_dict = {}
        self.data_path = r"C:\study\technion\MSc\3rd_semester\NLP\hw1\sec2-21"
        self.input_sentence_file = self.data_path+r"\sec2-21.words" 
        self.input_tags_file = self.data_path+r"\sec2-21.pos" 
        self.seen_tags_set = []
        self.regularization_lambda = 0.9
        self.num_of_sentences = 1000
        self.sentences_list = []
        self.words_feature_vectors = []
        self.frequent_word_tag_pairs_dict = {} #key -word, val - tag with count larger than threshold 
        self.word_tag_threshold = 4
        self.word_tag_threshold_counter = 0
        self.morphological_set_lambda = 1
        self.contextual_set_lambda = 1
        self.word_tag_set_lambda = 1
        self.basic_setup = "word_tag"
        self.medium_setup = "morphological"
        self.advanced_setup = "contextual"
        self.setup = "word_tag"
        self.feature_function = None
        
    def read_input_sentences_and_tags(self):
        temp_word_tag_dict = {}
        sentences_file = open(self.input_sentence_file,"rb").readlines()
        tags_file = open(self.input_tags_file,"rb").readlines()
        seen_tags_set = set()
        for i in range(0,self.num_of_sentences):
            words = sentences_file[i].split()
            tags = tags_file[i].split()
            s = sentence.sentence(words,tags)
            self.sentences_list.append(s)
            #create the word-tag counts dict
            for index in range(0,len(words)):
                if (words[index],tags[index]) in temp_word_tag_dict.keys():
                    temp_word_tag_dict[words[index],tags[index]] += 1    
                else:
                    temp_word_tag_dict[words[index],tags[index]] = 1
                seen_tags_set.add(tags[index])
        #rank temp_word_tag_dict according to counts/take top k features
        temp_word_tag_dict_sorted = collections.OrderedDict(sorted(temp_word_tag_dict.items(), key= lambda x: (x[1]),reverse=True)) 
        for ((word,tag),count) in temp_word_tag_dict_sorted.items():
            if count >= self.word_tag_threshold:
                self.word_tag_threshold_counter += 1
                if word in self.frequent_word_tag_pairs_dict.keys():
                    self.frequent_word_tag_pairs_dict[word].append(tag)
                else:
                    self.frequent_word_tag_pairs_dict[word] = [tag]
        print "self.word_tag_threshold_counter",self.word_tag_threshold_counter 
        #append the tag set to the member list
        for tag in seen_tags_set:
            self.seen_tags_set.append(tag)
        
#         features = features_hagar.feature_functions()
#         features.extract_word_tag_features(self.frequent_word_tag_pairs_dict)
                   
    def compute_Likelihood(self):
        def likelihood_func(*args):
            print "    computing L"
            try:
                t1 = time.clock()
#                 features_functions = features_hagar.feature_functions()
                likelihood = 0
                v= args[0]
                feature_vec_indices = []
                all_tags_feature_vec_indices = []
                for sentence in self.sentences_list:
                    for word_index in range(0,sentence.length):                    
                        try:
                            if self.setup == self.basic_setup: #word-tag
                                self.v_word_tag_set = np.zeros(self.features_functions.get_num_of_word_tag_features())
                                feature_vec_indices = self.features_functions.get_word_tag_feature_vec_indices(sentence.get_tag(word_index-2),sentence.get_tag(word_index-1),sentence.get_tag(word_index),sentence.get_word(word_index))
                            elif self.setup == self.medium_setup: #morpho':
                                self.v_morphological_contextual_set = np.zeros(self.features_functions.get_num_of_morphological_features())
                                feature_vec_indices = self.features_functions.get_morphological_feature_vec_indices(sentence.get_tag(word_index-2),sentence.get_tag(word_index-1),sentence.get_tag(word_index),sentence.get_word(word_index))
                            elif self.setup == self.advanced_setup:#contextual:
                                self.v_contextual_set = np.zeros(self.features_functions.get_num_of_cotextual_features())
                                feature_vec_indices = self.features_functions.get_contextual_feature_vec_indices(sentence.get_tag(word_index-2),sentence.get_tag(word_index-1),sentence.get_tag(word_index),sentence.get_word(word_index))
                            
                            for feature_index in feature_vec_indices:
                                likelihood += v[feature_index]*1
#                             likelihood += np.inner(args[0], self.words_feature_vectors[word_index])
                            # expected counts
                            expected_counts = 0
                            if self.setup == self.basic_setup:#word-tag
                                relevant_set_of_tags = self.frequent_word_tag_pairs_dict[sentence.get_word[word_index]]
                                for relevant_tag in relevant_set_of_tags:
                                    all_tags_feature_vec_indices.extend(self.features_functions.get_word_tag_feature_vec_indices(sentence.get_tag(word_index-2),sentence.get_tag(word_index-1),relevant_tag,sentence.get_word(word_index)))
                            elif self.setup == self.medium_setup:#morpho'
                                # go over only the tag it is appearing with
                                relevant_set_of_tags = self.frequent_word_tag_pairs_dict[sentence.get_word[word_index]]
                                for relevant_tag in relevant_set_of_tags:
                                    all_tags_feature_vec_indices.extend(self.features_functions.get_morphological_feature_vec_indices(sentence.get_tag(word_index-2),sentence.get_tag(word_index-1),relevant_tag,sentence.get_word(word_index)))                            
                            elif self.setup == self.advanced_setup: #contextual:
                                for tag in self.seen_tags_set:
                                    all_tags_feature_vec_indices.extend(self.features_functions.get_contextual_feature_vec_indices(sentence.get_tag(word_index-2),sentence.get_tag(word_index-1),tag,sentence.get_word(word_index)))
                            #sum up the addition to expected_counts 
                            for feature_index in all_tags_feature_vec_indices: 
                                expected_counts += v[feature_index]*1
                            #subtract from the total likelihood
                            likelihood -= math.log(expected_counts)
                        except Exception as err: 
                            sys.stderr.write("problem calculating likelihood_func")     
                            print err.args      
                            print err
                v_square = sum(math.pow(v_k, 2) for v_k in v)
                likelihood -= ((self.regularization_lambda / 2) * v_square)
                t2 = time.clock()
                print "    v-norm:", v_square
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
            t1 = time.clock()
            v = args[0]
            try:
                features_functions = self.feature_functions()
                all_tags_feature_vec = {} #temp dict for the feature vec result across all POS tags
                curr_relevant_set_of_tags = []
                
                if self.setup is self.basic_setup:
                    likelihood_derivative = [0] * features_functions.get_num_of_word_tag_features()
                elif self.setup is self.medium_setup:
                    likelihood_derivative = [0] * features_functions.get_num_of_morphological_features()                
                elif self.setup is self.advanced_setup:
                    likelihood_derivative = [0] * features_functions.get_num_of_contextual_features()
     
                for sentence in self.sentences_list:
                    for word_index in range(0,sentence.length):                   
                        #calc the probabilities first
                        if self.setup is self.basic_setup:
                            relevant_set_of_tags = self.frequent_word_tag_pairs_dict[sentence.get_word[word_index]]
                            curr_relevant_set_of_tags = relevant_set_of_tags
                            prob = [0] * len(relevant_set_of_tags)
                            inner_product = 0
                            all_tag_feature_vec_indices = {} #key is a tag, value is a list of feature indices "on"
                            for relevant_tag in relevant_set_of_tags:
                                all_tag_feature_vec_indices[relevant_tag] = features_functions.get_word_tag_feature_vec_indices(sentence.get_tag(word_index-2),sentence.get_tag(word_index-1),relevant_tag,sentence.get_word(word_index))
                                for feature_index in all_tag_feature_vec_indices[relevant_tag]:                                                  
                                    inner_product += v[feature_index]*1
                                prob[relevant_tag] += math.exp(inner_product)
                        elif self.setup is self.medium_setup :
                            relevant_set_of_tags = self.frequent_word_tag_pairs_dict[sentence.get_word[word_index]]
                            curr_relevant_set_of_tags = relevant_set_of_tags
                            prob = [0] * len(relevant_set_of_tags)
                            for relevant_tag in relevant_set_of_tags:
                                all_tag_feature_vec_indices[relevant_tag] = features_functions.get_morphological_feature_vec_indices(sentence.get_tag(word_index-2),sentence.get_tag(word_index-1),relevant_tag,sentence.get_word(word_index))
                                for feature_index in all_tag_feature_vec_indices[relevant_tag]:                                                  
                                    inner_product += v[feature_index]*1
                                prob[relevant_tag] += math.exp(inner_product)           
                        elif self.setup is self.advanced_setup:
                            prob = [0] * len(self.seen_tags_set)
                            curr_relevant_set_of_tags = self.seen_tags_set
                            for seen_tag in range(0,len(self.seen_tags_set)):
                                all_tag_feature_vec_indices[seen_tag] = features_functions.get_contextual_feature_vec_indices(sentence.get_tag(word_index-2),sentence.get_tag(word_index-1),seen_tag,sentence.get_word(word_index))
                                for feature_index in all_tag_feature_vec_indices[seen_tag]:                                                  
                                        inner_product += v[feature_index]*1
                                prob[seen_tag] = math.exp(inner_product)
                        normalization = sum(prob)
                        #go over all the features of the current setup
                        for k in range(0,len(likelihood_derivative)):
                            # empirical counts 
                            if k in all_tag_feature_vec_indices[sentence.get_tag[word_index]] :
                                likelihood_derivative[k] += 1
                            # expected counts
                            expected_count = 0
                            for tag in curr_relevant_set_of_tags:
                                f_k = all_tag_feature_vec_indices[tag][k]
                                expected_count +=  (f_k * prob[curr_relevant_set_of_tags.index(tag)]/normalization)
                            likelihood_derivative[k] -= expected_count
                            likelihood_derivative[k] -= (self.regularization_lambda*v[k])
                max_likelihood_derivative = map(lambda x: -1 * x, likelihood_derivative) 
                t2 = time.clock()
                print "    time to calc gradient:" , t2 - t1
            except Exception as err: 
                sys.stderr.write("problem gradient_likelihood")     
                print err.args      
                print err
            return scipy.array(max_likelihood_derivative)
        return gradient_likelihood
                       
    
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
            t1 = time.clock()
#             features_functions = features_hagar.feature_functions()
            if self.setup == self.basic_setup:
                self.features_functions.apply_word_tag_features(self.frequent_word_tag_pairs_dict)
            elif self.setup == self.medium_setup:
                self.features_functions.apply_morphological_features()
            elif self.setup == self.advanced_setup:
                self.features_functions.apply_contextual_features()    
#             for sentence in self.sentences_list:
#                 for word_index in range(0,sentence.length):
#                     #TODO: change to apply_{set_type}_feature!!!
#                     self.words_feature_vectors.append(features_functions.apply_feature_functions(sentence.get_tag(word_index-2),sentence.get_tag(word_index-1),sentence.get_tag(word_index),sentence.get_word(word_index)))
            t2 = time.clock()
            print "finished compute_features_on_all_words in", t2 - t1
        except Exception as err: 
            sys.stderr.write("problem in compute_features_on_all_words")     
            print err.args      
            print err                  
        
#     def calculate_prob_with_optimal_weight_vector(self):
#         features_functions = features.feature_functions()
#         for (tuple_index,tuple_feature_vec) in self.features_res_vector_for_curr_tuple_and_tag.items():
#             prob_nominator = math.exp(np.inner(tuple_feature_vec,self.v_optimal))
#             prob_normalization_factor = 0
#             for tag in self.POS_set:
#                 prob_normalization_factor +=  math.exp(features_functions.apply_feature_functions(self.word_tuples_dict[tuple_index], tag, self.sentences_dict[self.word_tuples_dict[tuple_index].sentence_index]))
#             self.tuple_probabilities_dict[tuple_index] = float(prob_nominator/prob_normalization_factor)
             
    def train_memm(self):
        """go over all the 5000 WSJ sentence
           create x - tuple (history), and the i'th word tag, and compute likelihood
           the output is the v_vector - weights for the features
        """
        self.read_input_sentences_and_tags()
        self.features_functions = features.feature_functions()
        self.compute_features_on_all_words()
        self.optimize_v() 
        
                   
def main():
    try:
        t_start = time.clock()
        memm_POS_tag = MEMM()
        memm_POS_tag.train_memm()
        t_end = time.clock()
        print " ---finished---",t_end-t_start
    except Exception as err: 
        sys.stderr.write("problem")     
        print err.args      
        print err
    
if __name__ == '__main__':
    main()     