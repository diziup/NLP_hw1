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
        self.word_seen_tags_list_dict = {}  #key - word, value - list of tags the word got
        self.frequent_word_tags_list_dict = {} #key - word, value - list of tags with count larger than threshold 
        self.frequent_word_tag_pairs_dict = {}  #key - word, value - tag 
        self.word_tag_threshold = threshold
        self.word_tag_threshold_counter = 0
        self.contextual_all_lambda = 0.5
        self.morphological_all_lambda = 0.5
        self.morphological_setup_all = "morphological_all"
        self.morphological_setup_unigram = "morphological_unigram"
        self.morphological_setup_bigram = "morphological_bigram"
        self.morphological_setup_trigram = "morphological_trigram"
        self.contextual_setup_all = "contextual_all"
        self.contextual_setup_unigram = "contextual_unigram"
        self.contextual_setup_bigram = "contextual_bigram"
        self.contextual_setup_trigram = "contextual_trigram"
        self.setup = setup
        self.feature_functions = features.feature_functions(setup,num_of_sentence,threshold,reg_lambda)
        self.train_smoothing_setup = "train_smoothing_setup" #take the v result of the separate gram runs, and interpolate
        self.smoothing_lambda_unigram = 0.2
        self.smoothing_lambda_bigram = 0.5
        self.smoothing_lambda_trigram = 0.3
        self.test_tags_results = [] #key is a sentence index from the test, value is a list of the predicted tags
        self.get_indices_function_dict = {"contextual_all":[self.feature_functions.get_word_tags_unigram_index,\
                                                            self.feature_functions.get_word_tags_bigram_index, \
                                                            self.feature_functions.get_word_tags_trigram_index],\
                                          "contextual_unigram":[self.feature_functions.get_word_tags_unigram_index],\
                                          "contextual_bigram":[self.feature_functions.get_word_tags_bigram_index],\
                                          "contextual_trigram":[self.feature_functions.get_word_tags_trigram_index],\
                                          "morphological_all":[self.feature_functions.get_morphological_unigram_index,\
                                                               self.feature_functions.get_morphological_bigram_index,\
                                                               self.feature_functions.get_morphological_trigram_index],\
                                          "morphological_unigram":[self.feature_functions.get_morphological_unigram_index],\
                                          "morphological_bigram":[self.feature_functions.get_morphological_bigram_index],\
                                          "morphological_trigram":[self.feature_functions.get_morphological_trigram_index],\
                                          "smoothing_contextual":[self.feature_functions.get_word_tags_unigram_index,\
                                                                  self.feature_functions.get_word_tags_bigram_index, \
                                                                  self.feature_functions.get_word_tags_trigram_index],\
                                          "smoothing_morphological":[self.feature_functions.get_morphological_unigram_index,\
                                                                    self.feature_functions.get_morphological_bigram_index,\
                                                                    self.feature_functions.get_morphological_trigram_index]
                                          
                                        }        
        self.q = {}  #key will be the possible t_i-2,t_i-1,t_i, value - its prob
        self.pi = {} #key - word, t_i-1,t_i value -  the max prob over all the t_i-2
        self.bp = {} #back pointer: key - word, t_i-1,t_i value - the tag t_i-2 that got the max prob
        self.lambda_smoothing_contextual_all = 0.5
        self.lambda_smoothing_morphological = 0.5
        self.lambda_smoothing_unigram = 0.333
        self.lambda_smoothing_bigram = 0.333
        self.lambda_smoothing_trigram = 0.333
        self.morphological_all_v_opt = None
        self.morphological_unigram_v_opt = None
        self.morphological_bigram_v_opt = None
        self.morphological_bigram_v_opt = None
        self.contextual_all_v_opt = None
        self.contextual_unigram_v_opt = None
        self.contextual_bigram_v_opt = None
        self.contextual_trigram_v_opt = None
        self.total_TP = 0.0
        self.total_TN = 0.0
        self.total_FP = 0.0
        self.total_FN = 0.0
        self.accuracy = 0.00
        self.recall = 0.00
        self.precision = 0.00
        self.F_measure= 0.00
        self.tags_results = {}
        
        
    def read_input_sentences_and_tags_for_test(self):
        sentences_file = open(self.input_sentence_file,"rb").readlines()
        tags_file = open(self.input_tags_file,"rb").readlines()
#         seen_tags_set = set()
        for i in range(self.num_of_sentences,self.num_of_sentences+1000):
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
                        if (s.words[index],s.POS_tags[index+2],"*","*") in temp_word_tag_dict.keys():
                            temp_word_tag_dict[s.words[index],s.POS_tags[index+2],"*","*"] += 1
                        else:
                            temp_word_tag_dict[s.words[index],s.POS_tags[index+2],"*","*"] = 1
                    else:
                        if (s.words[index],s.POS_tags[index+2],s.POS_tags[index+1],s.POS_tags[index]) in temp_word_tag_dict.keys():
                            temp_word_tag_dict[s.words[index],s.POS_tags[index+2],s.POS_tags[index+1],s.POS_tags[index]] += 1
                        else:
                            temp_word_tag_dict[s.words[index],s.POS_tags[index+2],s.POS_tags[index+1],s.POS_tags[index]] = 1
                    seen_tags_set.add(s.POS_tags[index+2])
                except Exception as err: 
                    sys.stderr.write("problem")     
                    print err.args      
                    print err       
        #rank temp_word_tag_dict according to counts/take top k features
        print "finished reading input file, calculating tags frequency"
        self.temp_word_tag_dict_sorted = collections.OrderedDict(sorted(temp_word_tag_dict.items(), key= lambda x: (x[1]),reverse=True))
        for ((word,tag,tag_1,tag_2),count) in self.temp_word_tag_dict_sorted.items():
            if word in self.word_seen_tags_list_dict.keys():
                if not tag in self.word_seen_tags_list_dict[word]:
                    self.word_seen_tags_list_dict[word].append(tag)
            else:
                self.word_seen_tags_list_dict[word] = [tag]
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
        self.save_to_pickle(self.word_seen_tags_list_dict, "word_seen_tags_list_dict")
        self.save_to_pickle(self.seen_tags_set, "seen_tags_set")
        print "finished ..."
#         self.create_word_tag_pairs_above_threshold_dict()
    
    def create_word_tag_pairs_above_threshold_dict(self):
        word_tag_pair_cnt = 0
        for (word,pairs_tags_list) in self.frequent_word_tags_list_dict.items():
            for tag,tag1 in pairs_tags_list:
                self.frequent_word_tag_pairs_dict[(word,tag,tag1)] = word_tag_pair_cnt
                word_tag_pair_cnt += 1
                   
    def compute_Likelihood(self):
        def likelihood_func(*args):
            print "    computing L"
            try:               
                t1 = time.clock()
                likelihood = 0
                v = args[0]
                sentence_counter = 0
#                 printing = int(self.num_of_sentences/5)
                for sentence in self.train_sentences_list:                   
                    for word_index in range(0,sentence.length):                    
                        feature_vec_indices = []              
                        try:
                            #empirical count
                            try:
                                for func in self.get_indices_function_dict[self.setup]:
                                    feature_vec_indices.extend(func(sentence.get_tag(word_index+2),sentence.get_tag(word_index+1),sentence.get_tag(word_index),sentence.get_word(word_index),self.setup))
#                                 print feature_vec_indices
                                for feature_ind in feature_vec_indices:
                                    likelihood += v[feature_ind]*1
                                #expected counts -  go over all possible tags
                            except Exception as err: 
                                sys.stderr.write("problem empirical count")     
                                print err.args      
                                print err
                            try:
                                expected_counts = 0                   
                                for tag in self.seen_tags_set:
                                    inner_product = 0
                                    curr_tag_feature_vec_indices = []
                                    for func in self.get_indices_function_dict[self.setup]:
                                        curr_tag_feature_vec_indices.extend(func(tag,sentence.get_tag(word_index+1),sentence.get_tag(word_index),sentence.get_word(word_index),self.setup))
                                    for feature_ind in curr_tag_feature_vec_indices: 
                                        inner_product += v[feature_ind]*1
                                    expected_counts += math.exp(inner_product)
                            except Exception as err: 
                                sys.stderr.write("problem expected counts with tag", tag)     
                                print err.args      
                                print err                                
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
                        curr_relevant_set_of_tags = self.word_seen_tags_list_dict[sentence.get_word(word_index)]
                        try:
                            for tag in self.seen_tags_set:
                                inner_product = 0
                                curr_indices = []
                                for func in self.get_indices_function_dict[self.setup]:
                                    curr_indices.extend(func(tag,sentence.get_tag(word_index+1),sentence.get_tag(word_index),sentence.get_word(word_index),self.setup))
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
            with open("v_opt_"+self.setup+"_reg_lambda_"+str(self.regularization_lambda)+"_sen_num_"+str(self.num_of_sentences)+"_threshold_"+str(self.word_tag_threshold), 'wb') as handle:
                cPickle.dump(self.v_optimal, handle)
            handle.close()    
#             cPickle.dump(self.v_optimal, "v_opt_"+self.setup+"_reg_lambda_"+str(self.regularization_lambda),"wb")
        except Exception as err: 
            sys.stderr.write("problem optimize_v")     
            print err.args      
            print err
    
    def save_to_pickle(self,d,filename):
        with open(filename, 'wb') as handle:
                cPickle.dump(d, handle)
        handle.close()
          
    def compute_features_on_all_words(self): 
        print "applying features..."
        try:
            t1 = time.clock()
            if "contextual" in self.setup: 
#                 self.feature_functions.apply_word_tags_features(self.frequent_word_tags_list_dict)
                self.feature_functions.apply_word_tags_features(self.frequent_word_tags_list_dict,self.train_sentences_list)
                self.num_of_feature_functions = self.feature_functions.num_of_feature_func
            elif "morphological" in self.setup:
                self.feature_functions.apply_morphological_word_tags_features(self.frequent_word_tags_list_dict)
                self.num_of_feature_functions = self.feature_functions.num_of_feature_func 
                
            t2 = time.clock()
            print "finished applying ",self.num_of_feature_functions," in", t2 - t1
        except Exception as err: 
            sys.stderr.write("problem in compute_features_on_all_words")     
            print err.args      
            print err                  
    
    def compute_q_prob_for_viterbi_single_features_set_setup(self,sentence,word_index):
        #initialize the S_k - the possible tags a word can get
            tag_set_minus_1 = []
            tag_set_minus_two = []
            if (word_index == 0):
                tag_set_minus_1.append('*')
                tag_set_minus_two.append('*')
            elif (word_index == 1):
                tag_set_minus_1 = self.seen_tags_set #maybe here 
                tag_set_minus_two.append('*')
            elif (word_index > 1):
                tag_set_minus_1 = self.seen_tags_set #need to change according to the beam search
                tag_set_minus_two = self.seen_tags_set
            # calc q - have a dict of dicts - for all the possible t_i_2,t_i_1 and t_i prob
            for tag_minus_2 in tag_set_minus_two:
                self.q[tag_minus_2] = {}
                for tag_minus_1 in tag_set_minus_1:
                    self.q[tag_minus_2][tag_minus_1] = {}
                    all_tag_feature_vec_indices = {}      
                    prob = {}       
                    # for t_i,t_i-1, t_i-2  - calc the normalization- the denominator for the prob
                    #get the tags that the current word was seen from, if it were in the train
                    for tag in self.seen_tags_set:
                        #according to the setup, get the indices of the "on" features for the current x.
                        inner_product = 0
                        curr_indices = [] = []
                        for func in self.get_indices_function_dict[self.setup]:
                            curr_indices.extend(func(tag,tag_minus_1,tag_minus_2,sentence.get_word(word_index),self.setup))
                        all_tag_feature_vec_indices[tag] = curr_indices
                        for feature_index in all_tag_feature_vec_indices[tag]:                                                  
                            inner_product += self.v_optimal[feature_index]*1
                        prob[tag] = math.exp(inner_product)
                    #final normalized prob
                    prob_denominator = sum(prob.values())
                    for tag in self.seen_tags_set:
                        self.q[tag_minus_2][tag_minus_1][tag] = float(prob[tag]/prob_denominator)       
    
    def compute_q_prob_for_viterbi_linear_interpolation_setup(self,sentence,word_index):
        #have 2 prob rountines for the contextual_all and morphological_all, when the final q is their sum
        #initialize the S_k - the possible tags a word can get
            tag_set_minus_1 = []
            tag_set_minus_two = []
            if (word_index == 0):
                tag_set_minus_1.append('*')
                tag_set_minus_two.append('*')
            elif (word_index == 1):
                tag_set_minus_1 = self.seen_tags_set #maybe here 
                tag_set_minus_two.append('*')
            elif (word_index > 1):
                tag_set_minus_1 = self.seen_tags_set #need to change according to the beam search
                tag_set_minus_two = self.seen_tags_set
            # calc q - have a dict of dicts - for all the possible t_i_2,t_i_1 and t_i prob
            for tag_minus_2 in tag_set_minus_two:
                self.q[tag_minus_2] = {}
                for tag_minus_1 in tag_set_minus_1:
                    self.q[tag_minus_2][tag_minus_1] = {}
                    all_tag_feature_vec_indices_contextual_all = {}    
                    all_tag_feature_vec_indices_morphological_all = {}     
                    prob_contextual_all = {}
                    prob_morphological_all = {}    
                    # for t_i,t_i-1, t_i-2  - calc the normalization- the denominator for the prob
                    #get the tags that the current word was seen from, if it were in the train
                    for tag in self.seen_tags_set:
                        #according to the setup, get the indices of the "on" features for the current x.
                        inner_product = 0
                        curr_indices = [] = []
                        for func in self.get_indices_function_dict["contextual_all"]:
                            curr_indices.extend(func(tag,tag_minus_1,tag_minus_2,sentence.get_word(word_index),self.setup))
                        all_tag_feature_vec_indices_contextual_all[tag] = curr_indices
                        for feature_index in all_tag_feature_vec_indices_contextual_all[tag]:                                                  
                            inner_product += self.contextual_all_v_opt[feature_index]*1
                        prob_contextual_all[tag] = math.exp(inner_product)
                    #final normalized prob
                    prob_denominator_contextuall = sum(prob_contextual_all)
                    for tag in self.seen_tags_set:
                        self.q[tag_minus_2][tag_minus_1][tag] = self.contextual_all_lambda*float(prob_contextual_all[tag]/prob_denominator_contextuall)       
                    
                    for tag in self.seen_tags_set:
                        #according to the setup, get the indices of the "on" features for the current x.
                        inner_product = 0
                        curr_indices = [] = []
                        for func in self.get_indices_function_dict["morphological_all"]:
                            curr_indices.extend(func(tag,tag_minus_1,tag_minus_2,sentence.get_word(word_index),self.setup))
                        all_tag_feature_vec_indices_morphological_all[tag] = curr_indices
                        for feature_index in all_tag_feature_vec_indices_morphological_all[tag]:                                                  
                            inner_product += self.morphological_all_v_opt[feature_index]*1
                        prob_morphological_all[tag] = math.exp(inner_product)
                    #final normalized prob
                    prob_denominator_morphological = sum(prob_morphological_all)
                    for tag in self.seen_tags_set:
                        self.q[tag_minus_2][tag_minus_1][tag] += self.morphological_all_lambda*float(prob_morphological_all[tag]/prob_denominator_morphological) 
    
    def compute_q_prob_for_viterbi_smoothing_setup(self,sentence,word_index):
        #have 3 prob rountines for the unigram,bigram and trigram 
        #initialize the S_k - the possible tags a word can get
            tag_set_minus_1 = []
            tag_set_minus_two = []
            if (word_index == 0):
                tag_set_minus_1.append('*')
                tag_set_minus_two.append('*')
            elif (word_index == 1):
                tag_set_minus_1 = self.seen_tags_set #maybe here 
                tag_set_minus_two.append('*')
            elif (word_index > 1):
                tag_set_minus_1 = self.seen_tags_set #need to change according to the beam search
                tag_set_minus_two = self.seen_tags_set
            # calc q - have a dict of dicts - for all the possible t_i_2,t_i_1 and t_i prob
            for tag_minus_2 in tag_set_minus_two:
                self.q[tag_minus_2] = {}
                for tag_minus_1 in tag_set_minus_1:
                    self.q[tag_minus_2][tag_minus_1] = {}
                    all_tag_feature_vec_indices_unigram = {}    
                    all_tag_feature_vec_indices_bigram = {}   
                    all_tag_feature_vec_indices_trigram = {}       
                    prob_unigram = {}
                    prob_bigram = {}
                    prob_trigram = {}       
                    # for t_i,t_i-1, t_i-2  - calc the normalization- the denominator for the prob
                    #get the tags that the current word was seen from, if it were in the train - 
                    #unigram
                    for tag in self.seen_tags_set:
                        #according to the setup, get the indices of the "on" features for the current x.
                        inner_product = 0
                        curr_indices = [] 
                        if self.setup == "smoothing_contextual":
                            for func in self.get_indices_function_dict["contextual_unigram"]:
                                curr_indices.extend(func(tag,tag_minus_1,tag_minus_2,sentence.get_word(word_index),self.setup))
                        elif self.setup == "smoothing_morphological":
                            for func in self.get_indices_function_dict["morphological_unigram"]:
                                curr_indices.extend(func(tag,tag_minus_1,tag_minus_2,sentence.get_word(word_index),self.setup))
                        
                        all_tag_feature_vec_indices_unigram[tag] = curr_indices
                        for feature_index in all_tag_feature_vec_indices_unigram[tag]:                                                  
                            if self.setup == "smoothing_contextual": 
                                inner_product += self.contextual_unigram_v_opt[feature_index]*1
                            elif self.setup == "smoothing_morphological":
                                inner_product += self.morphological_unigram_v_opt[feature_index]*1
                        prob_unigram[tag] = math.exp(inner_product)
                    #final normalized prob
                    prob_denominator_unigram = sum(prob_unigram)
                    for tag in self.seen_tags_set:
                        self.q[tag_minus_2][tag_minus_1][tag] = self.smoothing_lambda_unigram*float(prob_unigram[tag]/prob_denominator_unigram)       
                    #bigram
                    for tag in self.seen_tags_set:
                        #according to the setup, get the indices of the "on" features for the current x.
                        inner_product = 0
                        curr_indices = [] = []
                        if self.setup == "smoothing_contextual":
                            for func in self.get_indices_function_dict["contextual_bigram"]:
                                curr_indices.extend(func(tag,tag_minus_1,tag_minus_2,sentence.get_word(word_index),self.setup))
                        elif self.setup == "smoothing_morphological":
                            for func in self.get_indices_function_dict["morphological_bigram"]:    
                                all_tag_feature_vec_indices_bigram[tag] = curr_indices
                        for feature_index in all_tag_feature_vec_indices_bigram[tag]:                                                  
                            if self.setup == "smoothing_contextual": 
                                inner_product += self.contextual_bigram_v_opt[feature_index]*1
                            elif self.setup == "smoothing_morphological":
                                inner_product += self.morphological_bigram_v_opt[feature_index]*1
                        prob_bigram[tag] = math.exp(inner_product)
                    #final normalized prob
                    prob_denominator_bigram = sum(prob_bigram)
                    for tag in self.seen_tags_set:
                        self.q[tag_minus_2][tag_minus_1][tag] += self.smoothing_lambda_bigram*float(prob_bigram[tag]/prob_denominator_bigram)
                    #trigram
                    for tag in self.seen_tags_set:
                        #according to the setup, get the indices of the "on" features for the current x.
                        inner_product = 0
                        curr_indices = [] = []
                        if self.setup == "smoothing_contextual":
                            for func in self.get_indices_function_dict["contextual_trigram"]:
                                curr_indices.extend(func(tag,tag_minus_1,tag_minus_2,sentence.get_word(word_index),self.setup))
                        elif self.setup == "smoothing_morphological":
                            for func in self.get_indices_function_dict["morphological_trigram"]:
                                curr_indices.extend(func(tag,tag_minus_1,tag_minus_2,sentence.get_word(word_index),self.setup))
                        all_tag_feature_vec_indices_trigram[tag] = curr_indices
                        for feature_index in all_tag_feature_vec_indices_trigram[tag]:                                                  
                            if self.setup == "smoothing_contextual": 
                                inner_product += self.contextual_trigram_v_opt[feature_index]*1
                            elif self.setup == "smoothing_morphological":
                                inner_product += self.morphological_trigram_v_opt[feature_index]*1
                        prob_trigram[tag] = math.exp(inner_product)
                    #final normalized prob
                    prob_denominator_trigram = sum(prob_trigram)
                    for tag in self.seen_tags_set:
                        self.q[tag_minus_2][tag_minus_1][tag] += self.smoothing_lambda_trigram*float(prob_trigram[tag]/prob_denominator_trigram)                    
       
    def compute_viterbi(self, sentence):
        pi = {} #  value  the maximal prob 
        bp = {} #  ; value  the tag that got maximal prob 
        n = sentence.length
        best_sentence_tags = [None] * n
        print "viterbi: in sentence" , self.test_sentences_list.index(sentence)
        for word_index in range(0,n):
            if self.setup == "linear_inter":
                self.compute_q_prob_for_viterbi_linear_interpolation_setup(sentence,word_index)
            elif self.setup == "smoothing_contextual":
                self.compute_q_prob_for_viterbi_smoothing_setup(sentence,word_index)
            elif self.setup == "smoothing_morphological":
                self.compute_q_prob_for_viterbi_smoothing_setup(sentence,word_index)
            else:
                self.compute_q_prob_for_viterbi_single_features_set_setup(sentence,word_index)            
            try:#apply the recursion 
            #for the first word -  need to calculate all the |tags| options:
                pi[word_index] = {} #pword index] [ti][t_minus_1]
                bp[word_index] = {}
                if word_index == 0: 
                    pi[word_index]["*"] = {}
                    bp[word_index]["*"] = {}
                    for tag in self.seen_tags_set: # as t_i
                        pi[word_index]["*"][tag] = self.q["*"]["*"][tag]
                        bp[word_index]["*"][tag] = "*"
                    #get the max prob tag for word 1                   
                elif word_index == 1: #second word                            
                    for tag_minus_1 in self.seen_tags_set:
                        pi[word_index][tag_minus_1] = {}
                        bp[word_index][tag_minus_1] = {} 
                        for tag in self.seen_tags_set: # as t_i
                            pi[word_index][tag_minus_1][tag] = pi[word_index-1]["*"][tag_minus_1]*self.q["*"][tag_minus_1][tag]
                            bp[word_index][tag_minus_1][tag] = "*"  
                else: #word 3 and above
                    try:
                        for tag_minus_1 in self.seen_tags_set:
                            pi[word_index][tag_minus_1] = {}
                            bp[word_index][tag_minus_1] = {}   
                            for tag in self.seen_tags_set: 
                                pi_with_tag_minus_2 = []
                                for tag_minus_2 in self.seen_tags_set: #go over all the t_i_2 options
                                    pi_with_tag_minus_2.append((pi[word_index-1][tag_minus_2][tag_minus_1]*self.q[tag_minus_2][tag_minus_1][tag],tag_minus_2))    
                                max_prob , max_tag = self.find_max_prob_and_tag(pi_with_tag_minus_2)         
                                pi[word_index][tag_minus_1][tag] = max_prob
                                bp[word_index][tag_minus_1][tag] = max_tag
                    except Exception as err: 
                        sys.stderr.write("problem in viterbi with word index",word_index ,"with tag:",tag," tag minus 1:",tag_minus_1)     
                        print err.args      
                        print err   
            except Exception as err: 
                        sys.stderr.write("problem in viterbi with word index",word_index)     
                        print err.args      
                        print err
        
        #finished calc the words, now find the best sequence - from the end to the beginning of the sentence
        #need to find the best u,v from pi[n]
        max_tn_prob = 0
        best_tag_minus_1 = 0
        best_tag = 0
        for tag_minus_1 in pi[n-1].keys():
            for tag in pi[n-1][tag_minus_1].keys():
                if pi[n-1][tag_minus_1][tag] > max_tn_prob:
                    max_tn_prob = pi[n-1][tag_minus_1][tag]
                    best_tag_minus_1 = tag_minus_1
                    best_tag = tag 
        best_sentence_tags[n-1] = best_tag
        best_sentence_tags[n-2] = (best_tag_minus_1)
        #now calc the best sequence on the rest of the sentence - from k = (n-2),...,1:
        for k in range(n-3,-1,-1):
            best_sentence_tags[k] = bp[k+2][best_sentence_tags[k+1]][best_sentence_tags[k+2]]
            
        return best_sentence_tags 

    def find_max_prob_and_tag(self,t_minus_2_and_prob_list):
        prob , tags = zip(*t_minus_2_and_prob_list) 
        max_prob = max(prob)
        max_tag = tags[prob.index(max_prob)]
        return (max_prob,max_tag)           
#     def find_max_prob_and_tag(self,pi_dict):
#         max_prob = max(pi_dict.iteritems(), key=operator.itemgetter(1))[1]
#         max_tag = [k for k,v in pi_dict.items() if v==max_prob]
#         return [max_prob, max_tag]    
                                                                             
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
    def read_setup_data(self):
        if self.setup == "linear_inter": #need to read the contex_all v_opt and the morph_all v_opt
            self.contextual_all_v_opt =  cPickle.load( open("v_opt_contextual_all_reg_lambda_"+str(self.regularization_lambda)+"_sen_num_"+str(self.num_of_sentences)+"_threshold_"+str(self.word_tag_threshold), "rb" ) )
            self.morphological_all_v_opt = cPickle.load( open("v_opt_morphological_all_reg_lambda_"+str(self.regularization_lambda)+"_sen_num_"+str(self.num_of_sentences)+"_threshold_"+str(self.word_tag_threshold), "rb" ) )
        elif self.setup == "smoothing_contextual": #read the uni,bi,trigram v
            self.contextual_unigram_v_opt = cPickle.load( open("v_opt_contextual_unigram_reg_lambda_"+str(self.regularization_lambda)+"_sen_num_"+str(self.num_of_sentences)+"_threshold_"+str(self.word_tag_threshold), "rb" ) )
            self.contextual_bigram_v_opt = cPickle.load( open("v_opt_contextual_bigram_reg_lambda_"+str(self.regularization_lambda)+"_sen_num_"+str(self.num_of_sentences)+"_threshold_"+str(self.word_tag_threshold), "rb" ) )
            self.contextual_trigram_v_opt = cPickle.load( open("v_opt_contextual_trigram_reg_lambda_"+str(self.regularization_lambda)+"_sen_num_"+str(self.num_of_sentences)+"_threshold_"+str(self.word_tag_threshold), "rb" ) )            
        elif self.setup == "smoothing_morphological": #read the uni,bi,trigram v
            self.morphological_unigram_v_opt = cPickle.load( open("v_opt_morphological_unigram_reg_lambda_"+str(self.regularization_lambda)+"_sen_num_"+str(self.num_of_sentences)+"_threshold_"+str(self.word_tag_threshold), "rb" ) )
            self.morphological_bigram_v_opt = cPickle.load( open("v_opt_morphological_bigram_reg_lambda_"+str(self.regularization_lambda)+"_sen_num_"+str(self.num_of_sentences)+"_threshold_"+str(self.word_tag_threshold), "rb" ) )
            self.morphological_trigram_v_opt = cPickle.load( open("v_opt_morphological_trigram_reg_lambda_"+str(self.regularization_lambda)+"_sen_num_"+str(self.num_of_sentences)+"_threshold_"+str(self.word_tag_threshold), "rb" ) )            
        else:
            self.v_optimal =  cPickle.load( open("v_opt_"+self.setup+"_reg_lambda_"+str(self.regularization_lambda)+"_sen_num_"+str(self.num_of_sentences)+"_threshold_"+str(self.word_tag_threshold), "rb" ) )
        self.word_seen_tags_list_dict = cPickle.load( open("word_seen_tags_list_dict","rb"))
        self.seen_tags_set = cPickle.load( open("seen_tags_set","rb"))
        
    def test_memm(self):
        self.read_input_sentences_and_tags_for_test()
        self.feature_functions.read_features_dict_for_test()
        self.read_setup_data()
        for sentence in self.test_sentences_list:
            self.test_tags_results.append(self.compute_viterbi(sentence))
        self.save_to_pickle(self.test_tags_results, "test_tags_results_"+self.setup+"_reg_lambda_"+str(self.regularization_lambda)+"_threshold_"+str(self.word_tag_threshold))
        
    def read_test_tags_results(self):
        self.test_tags_results = cPickle.load( open("test_tags_results_"+self.setup+"_reg_lambda_"+str(self.regularization_lambda)+"_threshold_"+str(self.word_tag_threshold), "rb" ) )
        self.read_input_sentences_and_tags_for_test()
        #compare each sentence length of tags -  gold and predicted
#         diff_len    _sen_index = []
#         for sentence_index in range(0,len(self.test_tags_results)):
#             if len(self.test_tags_results[sentence_index]) != self.test_sentences_list[sentence_index].length:
#                 diff_len_sen_index.append(sentence_index)
#         if len(diff_len_sen_index) > 0:
#             print diff_len_sen_index     
#         else:
#             print "no diff len sentence"
    
    def analyze_results(self):
        self.read_input_sentences_and_tags_for_test()
        num_of_tags = 0
        tags_results = {}
        real_tag_list = []
        pred_tag_list = []
        #init confusion matrix for each tag from results and actual vector of tags
        for sentence in self.test_sentences_list:
            for j in range(0,len(sentence.POS_tags)):
                real_tag = sentence.get_tag(j)
                if real_tag!='*':
                    if(tags_results.has_key(real_tag)==False):
                        tags_results[real_tag]  = {"TP": 0,"TN":0,"FP":0,"FN":0}
                    num_of_tags +=1
                    real_tag_list.append(real_tag)
        for tag_list in range(0,len(self.test_tags_results)):
            for tag in range(0,len(self.test_tags_results[tag_list])):
                predict_tag = self.test_tags_results[tag_list][tag]
                if(tags_results.has_key(predict_tag)==False):
                    tags_results[predict_tag]  = {"TP": 0,"TN":0,"FP":0,"FN":0}
                num_of_tags +=1
                pred_tag_list.append(predict_tag)
        #calc TP and FN
        try:
            for real_tag in range(0,len(real_tag_list)):
                if(real_tag_list[real_tag] == pred_tag_list[real_tag]):
                    tags_results[real_tag_list[real_tag]]["TP"]+=1
                else:
                    tags_results[real_tag_list[real_tag]]["FN"]+=1
        except Exception as err: 
            sys.stderr.write("problem analyze_results, real_tag:",real_tag)     
            print err.args      
            print err
        #calc FP
        for pred_tag in range(0,len(pred_tag_list)):
            if(real_tag_list[pred_tag]!=pred_tag_list[pred_tag]):
                tags_results[pred_tag_list[pred_tag]]["FP"]+=1
        #calc total measures
        for tag in tags_results.keys():
            tags_results[tag]["TN"] = num_of_tags-tags_results[tag]["TP"]-tags_results[tag]["FP"]-tags_results[tag]["FN"]
        for tag in tags_results.keys():
            self.total_TP +=tags_results[tag]["TP"]
            self.total_TN +=tags_results[tag]["TN"]
            self.total_FP +=tags_results[tag]["FP"]
            self.total_FN +=tags_results[tag]["FN"]
        self.accuracy =(self.total_TP+self.total_TN)/(self.total_TP+self.total_TN+self.total_FP+self.total_FN)
        self.recall =(self.total_TP)/(self.total_TP+self.total_FN)
        self.precision =(self.total_TP)/(self.total_TP+self.total_FP)
        self.F_measure = 2*(self.recall*self.precision)/(self.recall+self.precision)
        self.tags_results = tags_results
        print "accuracy:", self.accuracy, "recall:",self.recall,"precision:",self.precision,"F:",self.F_measure