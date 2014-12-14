'''
@author: liorab
    history:4-tuple - t_i-2,t_i-1,word    
'''
# import features
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
        
        self.num_of_feature_functions = 0
        self.tuple_probabilities_dict = {}
        self.data_path = r"C:\study\technion\MSc\3rd_semester\NLP\hw1\sec2-21"
        self.input_sentence_file = self.data_path+r"\sec2-21.words" 
        self.input_tags_file = self.data_path+r"\sec2-21.pos" 
        self.seen_tags_set = []
        self.regularization_lambda = 0.5
        self.num_of_sentences = 5000
        self.sentences_list = []
        self.words_feature_vectors = []
        self.word_tags_list_dict = {}  #key - word, value - list of tags the word got
        self.frequent_word_tags_list_dict = {} #key - word, value - list of tags with count larger than threshold 
        self.frequent_word_tag_pairs_dict = {}  #key - word, value - tag 
        self.word_tag_threshold = 5
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
            if word in self.word_tags_list_dict.keys():
                    self.word_tags_list_dict[word].append(tag)
            else:
                self.word_tags_list_dict[word] = [tag]
            if count >= self.word_tag_threshold:
                self.word_tag_threshold_counter += 1
                if word in self.frequent_word_tags_list_dict.keys():
                    self.frequent_word_tags_list_dict[word].append(tag)
                else:
                    self.frequent_word_tags_list_dict[word] = [tag]
        print "self.word_tag_threshold_counter",self.word_tag_threshold_counter 
        #append the tag set to the member list
        for tag in seen_tags_set:
            self.seen_tags_set.append(tag)
        self.create_word_tag_pairs_above_threshold_dict()
                
    def create_word_tag_pairs_above_threshold_dict(self):
        word_tag_pair_cnt = 0
        for (word,tags_list) in self.frequent_word_tags_list_dict.items():
            for tag in tags_list:
                self.frequent_word_tag_pairs_dict[(word,tag)] = word_tag_pair_cnt 
                word_tag_pair_cnt +=1
                   
    def compute_Likelihood(self):
        def likelihood_func(*args):
            print "    computing L"
            try:
                t1 = time.clock()
                likelihood = 0
                v= args[0]
                feature_vec_indices = []                
                for sentence in self.sentences_list:
                    for word_index in range(0,sentence.length):                    
                        all_tags_feature_vec_indices = []
                        try:
                            #empirical count
                            if self.setup == self.basic_setup: #word-tag
                                feature_vec_indices = self.features_functions.get_word_tag_features_index(sentence.get_tag(word_index+2),sentence.get_tag(word_index+1),sentence.get_tag(word_index),sentence.get_word(word_index))
                            elif self.setup == self.medium_setup: #morpho':
                                feature_vec_indices = self.features_functions.get_morphological_features_index(sentence.get_tag(word_index+2),sentence.get_tag(word_index+1),sentence.get_tag(word_index),sentence.get_word(word_index))
                            elif self.setup == self.advanced_setup:#contextual:
                                feature_vec_indices = self.features_functions.get_contextual_feature_vec_indices(sentence.get_tag(word_index+2),sentence.get_tag(word_index+1),sentence.get_tag(word_index),sentence.get_word(word_index))
                            
                            if len(feature_vec_indices) > 0:
                                for feature_index in feature_vec_indices:
                                    likelihood += v[feature_index]*1
#                             likelihood += np.inner(args[0], self.words_feature_vectors[word_index])
                            
                            #expected counts
                            expected_counts = 0
                            if self.setup == self.basic_setup:#word-tag
                                for tag in self.seen_tags_set:
                                    all_tags_feature_vec_indices.extend(self.features_functions.get_word_tag_features_index(tag,sentence.get_tag(word_index+1),sentence.get_tag(word_index),sentence.get_word(word_index)))
                            elif self.setup == self.medium_setup:#morpho'
                                for tag in self.seen_tags_set:
                                    all_tags_feature_vec_indices.extend(self.features_functions.get_morphological_feature_vec_indices(sentence.get_tag(word_index+1),sentence.get_tag(word_index),tag,sentence.get_word(word_index)))
                            elif self.setup == self.advanced_setup: #contextual:
                                for tag in self.seen_tags_set:
                                    all_tags_feature_vec_indices.extend(self.features_functions.get_contextual_feature_vec_indices(tag,sentence.get_tag(word_index+1),sentence.get_tag(word_index),sentence.get_word(word_index)))
                            #sum up the addition to expected_counts 
                            if len(all_tags_feature_vec_indices) > 0:
                                for feature_index in all_tags_feature_vec_indices: 
                                    expected_counts += math.exp(v[feature_index]*1)
                            else: #no "on" feature- so the exponent will be 0 
                                expected_counts += math.exp(0)
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
#                 print "     likelihood value:",likelihood
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
                curr_relevant_set_of_tags = []
                all_tag_feature_vec_indices = {} #key is a tag, value is a list of feature indices "on"
                likelihood_derivative = [0] * self.num_of_feature_functions                
                
                for sentence in self.sentences_list:
                    for word_index in range(0,sentence.length):                   
                        #calc the probability first - denominator in expected counts. go over all the tags in the data
                        prob = {}
                        inner_product = 0
                        if self.setup is self.basic_setup:
                            relevant_set_of_tags = self.word_tags_list_dict[sentence.get_word(word_index)]
                            curr_relevant_set_of_tags = relevant_set_of_tags 
                            for tag in self.seen_tags_set:
                                all_tag_feature_vec_indices[tag] = self.features_functions.get_word_tag_features_index(tag,sentence.get_tag(word_index+1),sentence.get_tag(word_index),sentence.get_word(word_index))
                                for feature_index in all_tag_feature_vec_indices[tag]:                                                  
                                    inner_product += v[feature_index]*1
                                prob[tag] = math.exp(inner_product)
                        elif self.setup is self.medium_setup :
                            relevant_set_of_tags = self.word_tags_list_dict[sentence.get_word(word_index)]
                            curr_relevant_set_of_tags = relevant_set_of_tags         
                            for tag in self.seen_tags_set:
                                all_tag_feature_vec_indices[tag] = self.features_functions.get_morphological_feature_vec_indices(tag,sentence.get_tag(word_index+1),sentence.get_tag(word_index),sentence.get_word(word_index))
                                for feature_index in all_tag_feature_vec_indices[tag]:                                                  
                                    inner_product += v[feature_index]*1
                                prob[tag] = math.exp(inner_product)                              
                        elif self.setup is self.advanced_setup:
                            curr_relevant_set_of_tags = self.seen_tags_set
                            for tag in self.seen_tags_set:
                                all_tag_feature_vec_indices[tag] = self.features_functions.get_contextual_feature_vec_indices(tag,sentence.get_tag(word_index+1),sentence.get_tag(word_index),sentence.get_word(word_index))
                                for feature_index in all_tag_feature_vec_indices[tag]:                                                  
                                    inner_product += v[feature_index]*1
                                prob[tag] = math.exp(inner_product)
                        normalization = sum(prob.values())
                        #go over all the features of the current setup - with the relevant tags
                        
                        for k in all_tag_feature_vec_indices[sentence.get_tag(word_index+2)] :
                            # empirical counts 
#                             if k in all_tag_feature_vec_indices[sentence.get_tag(word_index+2)] :
                            likelihood_derivative[k] += 1
#                             else:
#                                 continue                        
                            # expected counts
                            expected_count = 0
                            for tag in curr_relevant_set_of_tags:  
                                f_k = 1
                                expected_count +=  (f_k * prob[tag]/normalization)
                            likelihood_derivative[k] -= expected_count
                            likelihood_derivative[k] -= (self.regularization_lambda*v[k])
                max_likelihood_derivative = map(lambda x: -1 * x, likelihood_derivative) 
#                 print "    max_likelihood_derivative:",max_likelihood_derivative
                t2 = time.clock()
                print "    time to calc gradient:" , t2 - t1
            except Exception as err: 
                sys.stderr.write("    problem gradient_likelihood")     
                print err.args      
                print err
            return scipy.array(max_likelihood_derivative)
        return gradient_likelihood
       
    def optimize_v(self):
        print "optimizing v with features:", self.num_of_feature_functions
        try:
            x0 = [0]*self.num_of_feature_functions
            v_opt = fmin_l_bfgs_b(self.compute_Likelihood(), x0, self.compute_likelihood_gradient(), disp=False)
            self.v_optimal = v_opt[0]           
            print "v_optimal",self.v_optimal
        except Exception as err: 
            sys.stderr.write("problem optimize_v")     
            print err.args      
            print err
            
    def compute_features_on_all_words(self): 
        try:
            t1 = time.clock()
            if self.setup == self.basic_setup:
                self.features_functions.apply_word_tag_features(self.frequent_word_tag_pairs_dict)
            elif self.setup == self.advanced_setup:
                self.features_functions.apply_contextual_features()    
            self.num_of_feature_functions = self.features_functions.get_num_of_features(self.setup)
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