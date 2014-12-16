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
import cPickle

class MEMM():
    
    def __init__(self,setup,num_of_sentence,reg_lambda,threshold):
        
        self.num_of_feature_functions = 0
        self.tuple_probabilities_dict = {}
        self.data_path = r"C:\study\technion\MSc\3rd_semester\NLP\hw1\sec2-21"
        self.input_sentence_file = self.data_path+r"\sec2-21.words" 
        self.input_tags_file = self.data_path+r"\sec2-21.pos" 
        self.seen_tags_set = []
        self.regularization_lambda = reg_lambda
        self.num_of_sentences = num_of_sentence
        self.train_sentences_list = []
        self.test_sentences_list = []
        self.words_feature_vectors = []
        self.word_tags_list_dict = {}  #key - word, value - list of tags the word got
        self.frequent_word_tags_list_dict = {} #key - word, value - list of tags with count larger than threshold 
        self.frequent_word_tag_pairs_dict = {}  #key - word, value - tag 
        self.word_tag_threshold = threshold
        self.word_tag_threshold_counter = 0
        self.morphological_set_lambda = 1
        self.contextual_set_lambda = 1
        self.word_tag_set_lambda = 1
        self.morphological_setup_all = "morphological_all"
        self.morphological_setup_unigram = "morphological_unigram"
        self.morphological_setup_bigram = "morphological_bigram"
        self.morphological_setup_trigram = "morphological_trigram"
        self.contextual_setup_all = "contextual_all"
        self.contextual_setup_unigram = "contextual_unigram"
        self.contextual_setup_bigram = "contextual_bigram"
        self.contextual_setup_trigram = "contextual_trigram"
        self.setup = setup
        self.feature_functions = features.feature_functions(num_of_sentence,threshold)
        self.train_smoothing_setup = "train_smoothing_setup" #take the v result of the separate gram runs, and interpolate
        self.smoothing_lambda_unigram = 0.2
        self.smoothing_lambda_bigram = 0.5
        self.smoothing_lambda_trigram = 0.3
        self.test_tags_results = {} #key is a sentence index from the test, value is a list of the predicted tags
        self.get_indices_function_dict = {"contextual_all":[self.feature_functions.get_word_tags_unigram_index,\
                                                            self.feature_functions.get_word_tags_bigram_index, \
                                                            self.feature_functions.get_word_tags_trigram_index],\
                                          "contextual_unigram":[self.feature_functions.get_word_tags_unigram_index],\
                                          "contextual_bigram":[self.feature_functions.get_word_tags_bigram_index],\
                                          "contextual_trigram":[self.feature_functions.get_word_tags_trigram_index]
                                        }
        
        self.q = {}  #key will be the possible t_i-2,t_i-1,t_i, value - its prob
        self.pi = {} #key - word, t_i-1,t_i value -  the max prob over all the t_i-2
        self.bp = {} #back pointer: key - word, t_i-1,t_i value - the tag t_i-2 that got the max prob
        
        
    def read_input_sentences_and_tags_for_test(self):
        sentences_file = open(self.input_sentence_file,"rb").readlines()
        tags_file = open(self.input_tags_file,"rb").readlines()
#         seen_tags_set = set()
        for i in range(5000,self.num_of_sentences+2000):
            words = sentences_file[i].split()
            tags = tags_file[i].split()
            s = sentence.sentence(words,tags)
            self.test_sentences_list.append(s)
        
    def read_input_sentences_and_tags_for_train(self):
        print "reading input file...."
        temp_word_tag_dict = {}
        sentences_file = open(self.input_sentence_file,"rb").readlines()
        tags_file = open(self.input_tags_file,"rb").readlines()
        seen_tags_set = set()
        for i in range(0,self.num_of_sentences):
            words = sentences_file[i].split()
            tags = tags_file[i].split()
            s = sentence.sentence(words,tags)
            self.train_sentences_list.append(s)
            #create the word-tag counts dict
            for index in range(0,len(words)):
                try:
                    if index == 0:
                        if (words[index],tags[index],"*","*") in temp_word_tag_dict.keys():
                            temp_word_tag_dict[words[index],tags[index],"*","*"] += 1
                        else:
                            temp_word_tag_dict[words[index],tags[index],"*","*"] = 1
                    else:
                        if (words[index],tags[index],tags[index-1],tags[index-2]) in temp_word_tag_dict.keys():
                            temp_word_tag_dict[words[index],tags[index],tags[index-1],tags[index-2]] += 1
                        else:
                            temp_word_tag_dict[words[index],tags[index],tags[index-1],tags[index-2]] = 1
                    seen_tags_set.add(tags[index])
                except Exception as err: 
                    sys.stderr.write("problem")     
                    print err.args      
                    print err       
        #rank temp_word_tag_dict according to counts/take top k features
        print "finished reading input file, calculating tags frequency"
        self.temp_word_tag_dict_sorted = collections.OrderedDict(sorted(temp_word_tag_dict.items(), key= lambda x: (x[1]),reverse=True))
        for ((word,tag,tag_1,tag_2),count) in self.temp_word_tag_dict_sorted.items():
            if word in self.word_tags_list_dict.keys():
                    self.word_tags_list_dict[word].append((tag,tag_1,tag_2))
            else:
                self.word_tags_list_dict[word] = [(tag,tag_1,tag_2)]
            if count >= self.word_tag_threshold:
                self.word_tag_threshold_counter += 1
                if word in self.frequent_word_tags_list_dict.keys():
                    self.frequent_word_tags_list_dict[word].append((tag,tag_1,tag_2))
                else:
                    self.frequent_word_tags_list_dict[word] = [(tag,tag_1,tag_2)]
#         print "self.word_tag_threshold_counter",self.word_tag_threshold_counter 
        #append the tag set to the member list
        for tag in seen_tags_set:
            self.seen_tags_set.append(tag)
        print "finished ..."
#         self.create_word_tag_pairs_above_threshold_dict()
      
    def create_word_tag_pairs_above_threshold_dict(self):
        word_tag_pair_cnt = 0
        for (word,pairs_tags_list) in self.frequent_word_tags_list_dict.items():
            for tag,tag1 in pairs_tags_list:
                self.frequent_word_tag_pairs_dict[(word,tag,tag1)] = word_tag_pair_cnt
                word_tag_pair_cnt +=1
                   
    def compute_Likelihood(self):
        def likelihood_func(*args):
            print "    computing L"
            try:               
                t1 = time.clock()
                likelihood = 0
                v= args[0]
                sentence_counter = 0
#                 printing = int(self.num_of_sentences/5)
                for sentence in self.train_sentences_list:                   
                    for word_index in range(0,sentence.length):                    
                        feature_vec_indices = []
                        all_tags_feature_vec_indices = []
                        try:
                            #empirical count
                            for func in self.get_indices_function_dict[self.setup]:
                                feature_vec_indices.extend(func(sentence.get_tag(word_index+2),sentence.get_tag(word_index+1),sentence.get_tag(word_index),sentence.get_word(word_index)))
                            if len(feature_vec_indices) > 0:
                                for feature_index in feature_vec_indices:
                                    likelihood += v[feature_index]*1
                            #expected counts
                            expected_counts = 0
                            inner_product = 0
                            try:
                                for tag in self.seen_tags_set:
                                    for func in self.get_indices_function_dict[self.setup]:
                                        all_tags_feature_vec_indices.extend(func(tag,sentence.get_tag(word_index+1),sentence.get_tag(word_index),sentence.get_word(word_index)))
                                    if len(all_tags_feature_vec_indices) > 0:
                                        for feature_index in all_tags_feature_vec_indices: 
                                            inner_product += v[feature_index]*1
                                    expected_counts += math.exp(inner_product)
                            except Exception as err: 
                                sys.stderr.write("problem with tag", tag)     
                                print err.args      
                                print err
                            #sum up the addition to expected_counts 
#                             if len(all_tags_feature_vec_indices) > 0:
#                                 for feature_index in all_tags_feature_vec_indices: 
#                                     expected_counts += v[feature_index]*1
#                             else: #no "on" feature- so the exponent will be 0 
#                                 expected_counts += 0
                                #subtract from the total likelihood
#                             expected_counts = math.exp(expected_counts)
                            likelihood -= math.log(expected_counts)
                        except Exception as err: 
                            sys.stderr.write("problem calculating likelihood_func")     
                            print err.args      
                            print err
                    sentence_counter += 1
#                     if (sentence_counter % 5) == 1:
#                         print "done:",sentence_counter," in",(time.clock()- t1)/sentence_counter
                v_square = sum(math.pow(v_k, 2) for v_k in v)
                likelihood -= ((self.regularization_lambda / 2) * v_square)
                t2 = time.clock()
                print "L:",likelihood," v-norm:", math.sqrt(v_square)
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
                
                #key is a tag, value is a list of feature indices "on"
                likelihood_derivative = [0] * self.num_of_feature_functions                
                
                for sentence in self.train_sentences_list:
                    for word_index in range(0,sentence.length):
                        all_tag_feature_vec_indices = {}                   
                        #calc the probability first - denominator in expected counts. go over all the tags in the data
                        prob = {}
                        inner_product = 0
                        curr_relevant_set_of_tags = []
                        curr_relevant_set_of_tags = self.word_tags_list_dict[sentence.get_word(word_index)][0]
                        try:
                            for tag in self.seen_tags_set:
                                curr_indices = []
                                for func in self.get_indices_function_dict[self.setup]:
                                    curr_indices.extend(func(sentence.get_tag(word_index+2),sentence.get_tag(word_index+1),sentence.get_tag(word_index),sentence.get_word(word_index)))
                                all_tag_feature_vec_indices[tag] = curr_indices
                                for feature_index in all_tag_feature_vec_indices[tag]:                                                  
                                    inner_product += v[feature_index]*1
                                prob[tag] = math.exp(inner_product)
                        except OverflowError:
                            print "overflow error: exponent =", inner_product," in sentence index"\
                                , self.train_sentences_list.index(sentence) ,"in word index:", word_index," v[feature_index]:",v[feature_index]
                            raise
                        normalization = sum(prob.values())
                        #go over all the features of the current setup - with the relevant tags
                        
                        for k in all_tag_feature_vec_indices[sentence.get_tag(word_index+2)] :
                            # empirical counts 
                            likelihood_derivative[k] += 1
                            # expected counts
                            expected_count = 0
                            for tag in curr_relevant_set_of_tags:
                                if tag != "*":  
                                    f_k = 1
                                    expected_count +=  (f_k * prob[tag]/normalization)
                            likelihood_derivative[k] -= expected_count
                            likelihood_derivative[k] -= (self.regularization_lambda*v[k])
                max_likelihood_derivative = map(lambda x: -1 * x, likelihood_derivative)
                likelihood_derivative = sum(math.pow(likelihood_derivative_k, 2) for likelihood_derivative_k in likelihood_derivative) 
                print "   likelihood_derivative norm:",math.sqrt(likelihood_derivative)
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
            v_opt = fmin_l_bfgs_b(self.compute_Likelihood(), x0, self.compute_likelihood_gradient(), disp=True)
            self.v_optimal = v_opt[0]           
            print "v_optimal",self.v_optimal
            with open("v_opt_"+self.setup+"_reg_lambda_"+str(self.regularization_lambda), 'wb') as handle:
                cPickle.dump(self.v_optimal, handle)
            handle.close()    
#             cPickle.dump(self.v_optimal, "v_opt_"+self.setup+"_reg_lambda_"+str(self.regularization_lambda),"wb")
        except Exception as err: 
            sys.stderr.write("problem optimize_v")     
            print err.args      
            print err
            
    def compute_features_on_all_words(self): 
        print "applying features..."
        try:
            t1 = time.clock()
            if "contextual" in self.setup: 
                self.feature_functions.apply_word_tags_features(self.frequent_word_tags_list_dict)
                self.num_of_feature_functions = self.feature_functions.num_of_word_tags_features
            elif "morphological" in self.setup:
                self.feature_functions.apply_set2_features(self.frequent_word_tag_pairs_dict)
            
            t2 = time.clock()
            print "finished applying ",self.num_of_feature_functions," in", t2 - t1
        except Exception as err: 
            sys.stderr.write("problem in compute_features_on_all_words")     
            print err.args      
            print err                  
    
    def calc_viterbi_probabilities(self, sentence):
        for word_index in range(0,len(sentence)):
            #initialize the S_k - the possible tags a word can get
            tag_set_minus_1 = []
            tag_set_minus_two = []
            
            if (word_index == 0):
                tag_set_minus_1.append('*')
                tag_set_minus_two.append('*')
            elif (word_index == 1):
                tag_set_minus_1=self.seen_tags_set
                tag_set_minus_two.append('*')
            elif (word_index > 1):
                tag_set_minus_1 = self.seen_tags_set
                tag_set_minus_two = self.seen_tags_set
            # calc q - have a dict of dicts - for all the possible t_i_2,t_i_1 and t_i prob
            for tag_minus_2 in tag_set_minus_two:
                self.q[tag_minus_2] = {}
                for tag_minus_1 in tag_set_minus_1:
                    self.q[tag_minus_2][tag_minus_1] = {}
                    prob_denominator = 0
                    # for t_i,t_i-1, t_i-2  - calc the normalization- the denominator for the prob
                    for tag in self.seen_tags_set:
                        #according to the setup, get the indices of the "on" features for the current x.
                        feature_vec_indices = self.features_functions.get_contextual_feature_vec_indices(tag,sentence.get_tag(word_index+1),sentence.get_tag(word_index),sentence.get_word(word_index))
                        for feature_index in feature_vec_indices:
                            prob_denominator += self.v_optimal[feature_index]
                    prob_denominator = math.exp(prob_denominator)
                    #final normalized prob
                    for tag in self.seen_tags_set:
                        prob_nominator = 0
                        feature_vec_indices = self.feature_functions.get_contextual_feature_vec_indices(tag,sentence.get_tag(word_index+1),sentence.get_tag(word_index),sentence.get_word(word_index))                        
                        for feature_index in feature_vec_indices:
                            prob_nominator += self.v_optimal[feature_index]
                        self.q[tag_minus_2][tag_minus_1][tag] = float(math.exp(prob_nominator)/prob_denominator)
            #apply the recursion
            #for the first word -  need to calculate all the |tags| options:
            pi = {} # key (word_index,t_minus_1,t) ; value  the maximal prob 
            bp = {} # key (word_index,t_minus_1,t) ; value  the tag that got maximal prob 
            k_top_prob_tags = 5 #for the beam search -  replace the t_minus_2 and minus_1 with the top k
            
            pi[word_index] = {}
            bp[word_index] = {}
            for tag in self.seen_tags_set:
                pi[word_index][tag]={}
                bp[word_index][tag]={}
                if word_index == 0:
                    pi[word_index]["*"][tag] = self.q["*"]["*"][tag]
            #finished calc the first word, take the top k results from it - sort according to q
            for tag in pi[word_index]["*"].values():
                 
                 
                
                       
            
    def find_best_tag_sequence(self,sentence,word_index):
        #go over the possible tags for t_i - 
        pi = {}
        bp = {}
        pi[word_index] = {}
        bp[word_index] = {}
        
        if word_index == 0: #first word -  need to go over only on t_i t options
            for tag in self.seen_tags_set:
                pi[word_index][tag]
                                          
    def compute_viterbi(self,sentence):
        self.calc_viterbi_probabilities(sentence)
#         self.find_best_tag_sequence(sentence)                 
    
    def summarize(self):
        print "MODEL SUMMARY:"
        print   "\tnum sentences             =", self.num_of_sentences, \
                  "\n\tnum features              =", self.num_of_feature_functions, \
                  "\n\tnum tags                  =", len(self.seen_tags_set), \
                  "\n\tlamda                     =", self.regularization_lambda, \
    
    
    def train_memm(self):
        """go over all the 5000 WSJ sentence
           create x - tuple (history), and the i'th word tag, and compute likelihood
           the output is the v_vector - weights for the features
        """
        self.read_input_sentences_and_tags_for_train()
        self.compute_features_on_all_words()
        self.optimize_v()
        self.summarize()
    
#     def analyze_results(self):
#         #compute the precision/recall for each tag in the data.
#         #keep a running total for each tag: true_pos, false_neg, false_pos
#         results_per_tag = {} #key is a tag, value is a list of true_pos, false_neg, false_pos
#         for sentence_index in 
               
    def test_memm(self):
        self.read_input_sentences_and_tags_for_test()
        self.compute_features_on_all_words()
        for sentence in self.test_sentences_list:
            self.compute_viterbi(sentence)
#             self.test_tags_results[self.test_sentences_list.index(sentence)] = self.compute_viterbi(sentence)
   
        