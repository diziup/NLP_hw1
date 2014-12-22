'''
@author: liorab
Feature functions for the POS tagging task
'''
from collections import OrderedDict
import sys
import csv
import cPickle
import re

class feature_functions():
    feature_func_list = []

    def __init__(self,setup,num_of_sentences,threshold,reg_lambda):
#         self.num_of_feature_functions = num_of_feature_functions
#         self.features_res_vector = np.zeros(num_of_feature_functions)
        self.feature_func_list = []
#         self.create_feature_functions_set_and_weight_vector()
        
        self.sorted_unigram_features_freq = {}
        self.sorted_bigram_features_freq = {}
        self.sorted_trigram_features_freq = {}
        self.feature_tag_bigram = {}
        self.feature_tag_trigram = {}
        self.feature_tag_unigram = {}
        self.sorted_word_tag_freq = {}
        self.sorted_morphological_features = {}
        self.num_of_contextual_unigram_features = 0
        self.num_of_contextual_bigram_features = 0
        self.num_of_contextual_trigram_features = 0
        self.num_of_morphological_tag_features = 0
        self.num_of_morphological_features = 2
        self.num_of_word_tag_features = 0
        self.data_path = r"C:\study\technion\MSc\3rd_semester\NLP\hw1\sec2-21"
        self.input_tags_file = self.data_path+r"\sec2-21.pos"
        self.input_sentence_file = self.data_path+r"\sec2-21.words"

#         self.input_tags_file = r"../data/sec2-21/sec2-21.pos"
#         self.input_sentence_file = r"../data/sec2-21/sec2-21.words"
        self.tags_list = []
        self.num_of_sentences = num_of_sentences
        self.word_tag_threshold = threshold
        self.word_tag_threshold_counter = 0
        self.frequent_word_tags_3gram_dict = {}
        self.frequent_word_tags_2gram_dict = {}
        self.frequent_word_tags_1gram_dict = {}
        self.num_of_word_tag_1gram_features = 0
        self.num_of_word_tag_2gram_features = 0
        self.num_of_word_tag_3gram_features = 0
        
        self.setup = setup
        self.regularization_lambda = reg_lambda
        self.num_of_feature_func = 0
        self.apply_feature_function_dict = {"contextual_all":[self.extract_contextual_unigram_tag_with_word,\
                                                            self.extract_contextual_bigram_tag_with_word, \
                                                            self.extract_trigram_tag_with_word],\
                                          "contextual_unigram":[self.extract_contextual_unigram_tag_with_word],\
                                          "contextual_bigram":[self.extract_contextual_bigram_tag_with_word],\
                                          "contextual_trigram":[self.extract_trigram_tag_with_word]
                                        }
        self.morphological_unigram_features = {}
        self.morphological_bigram_features = {}
        self.morphological_trigram_features = {}
        self.prefix_list = ['re','co','in','pr','de','st','pre','con','di','pro']
        self.suffix_list = ['s','e','ed','y','ing','n','t','es','l','er','ly','ion','ted','ers','ent','ons','ies',]
        
        self.contextual_unigram_threshold = 7
        self.contextual_bigram_threshold = 7
        self.contextual_trigram_threshold = 6 
#     2. word_tag setup  - Set of contextual Features + current word

    def extract_word_tag_features(self,d):
        word_tag_triple_cnt = 0
        for (word,triple_tags_list) in d.items():
            for tag,tag1,tag2 in triple_tags_list:
                self.sorted_word_tag_freq[(word,tag,tag1,tag2)] = word_tag_triple_cnt
                word_tag_triple_cnt +=1
    
    def extract_contextual_bigram_tag_with_word(self,list_of_sentences):
        print "extracting contextual bigram tag..."
        temp_word_tag_dict = {}
        word_2tags_list_dict = {}

        for sentece in list_of_sentences:
            for index in range(0,len(sentece.words)):
                try:
                    if index == 0:
                        if (sentece.words[index],sentece.POS_tags[index+2],"*") in temp_word_tag_dict.keys():
                            temp_word_tag_dict[sentece.words[index],sentece.POS_tags[index+2],"*"] += 1
                        else:
                            temp_word_tag_dict[sentece.words[index],sentece.POS_tags[index+2],"*"] = 1
                    else:
                        if (sentece.words[index],sentece.POS_tags[index+2],sentece.POS_tags[index+1]) in temp_word_tag_dict.keys():
                            temp_word_tag_dict[sentece.words[index],sentece.POS_tags[index+2],sentece.POS_tags[index+1]] += 1
                        else:
                            temp_word_tag_dict[sentece.words[index],sentece.POS_tags[index+2],sentece.POS_tags[index+1]] = 1
                    #seen_tags_set.add(tags[index])
                except Exception as err:
                    sys.stderr.write("problem")
                    print err.args
                    print err
        #rank temp_word_tag_dict according to counts/take top k features
        temp_word_2tag_dict_sorted = OrderedDict(sorted(temp_word_tag_dict.items(), key= lambda x: (x[1]),reverse=True))
        for ((word,tag,tag_1),count) in temp_word_2tag_dict_sorted.items():
            if word in word_2tags_list_dict.keys():
                    word_2tags_list_dict[word].append((tag,tag_1))
            else:
                word_2tags_list_dict[word] = [(tag,tag_1)]
            if count >= self.word_tag_threshold:
                self.word_tag_threshold_counter += 1
                if (word,tag_1,tag) in self.frequent_word_tags_2gram_dict.keys():
                    self.frequent_word_tags_2gram_dict[(word,tag_1,tag)]+=1
                else:
                    self.frequent_word_tags_2gram_dict[(word,tag_1,tag)] = 1
        self.num_of_word_tag_2gram_features = len(self.frequent_word_tags_2gram_dict)
        return self.num_of_word_tag_2gram_features
    
    def extract_contextual_unigram_tag_with_word(self, list_of_sentences):
        print "extracting contextual unigram tag features"
        temp_word_tag_dict = {}
        word_1tags_list_dict = {}
        for sentece in list_of_sentences:
            for index in range(0,len(sentece.words)):
                try:
                    if (sentece.words[index],sentece.POS_tags[index+2]) in temp_word_tag_dict.keys():
                            temp_word_tag_dict[sentece.words[index],sentece.POS_tags[index+2]] += 1
                    else:
                            temp_word_tag_dict[sentece.words[index],sentece.POS_tags[index+2]] = 1
                    #seen_tags_set.add(tags[index])
                except Exception as err:
                    sys.stderr.write("problem in extract_contextual_unigram_tag_with_word")
                    print err.args
                    print err
        #rank temp_word_tag_dict according to counts/take top k features
        temp_word_1tag_dict_sorted = OrderedDict(sorted(temp_word_tag_dict.items(), key= lambda x: (x[1]),reverse=True))
#         self.write_to_file(temp_word_1tag_dict_sorted)
        for ((word,tag),count) in temp_word_1tag_dict_sorted.items():
            if word in word_1tags_list_dict.keys():
                word_1tags_list_dict[word].append(tag)
            else:
                word_1tags_list_dict[word] = [tag]
            if count >= self.word_tag_threshold:
                self.word_tag_threshold_counter += 1
                if (word,tag) in self.frequent_word_tags_1gram_dict.keys():
                    self.frequent_word_tags_1gram_dict[(word,tag)] +=1
                else:
                    self.frequent_word_tags_1gram_dict[(word,tag)] = 1
#         print "self.word_tag_threshold_counter",self.word_tag_threshold_counter
        self.num_of_word_tag_1gram_features = len(self.frequent_word_tags_1gram_dict)
        return self.num_of_word_tag_1gram_features
    
    def write_to_file(self,d): 
        w = csv.writer(open("words_frequency.csv", "w"))
        for key, val in d.items():
            w.writerow([key, val])           

    def extract_trigram_tag_with_word(self,d):
        print "extracting contextual trigram tag..."
        self.frequent_word_tags_3gram_dict = self.sorted_word_tag_freq
        self.num_of_word_tag_3gram_features = len(self.frequent_word_tags_3gram_dict)
        return self.num_of_word_tag_3gram_features
    
    def extract_contextual_features_new(self,list_of_sentences):
        self.extract_contextual_unigram_tag_with_word(list_of_sentences)
        self.extract_contextual_bigram_tag_with_word(list_of_sentences)
        self.extract_trigram_tag_with_word()
    
    def set_contextual_features_dict(self):
        if self.setup == "contextual_unigram" or self.setup == "contextual_all":
            for i in range(0,len(self.frequent_word_tags_1gram_dict)):
                new_key = self.frequent_word_tags_1gram_dict.keys()[i]
                self.feature_tag_unigram[new_key] = i
            filename = "features_dict_contextual_unigram_sen_num_"+str(self.num_of_sentences)+"_threshold_"+str(self.word_tag_threshold)
            self.save_to_pickle(self.feature_tag_unigram, filename)
        if self.setup == "contextual_bigram" or self.setup == "contextual_all":
            for i in range(0,len(self.frequent_word_tags_2gram_dict)):
                new_key = self.frequent_word_tags_2gram_dict.keys()[i]
                self.feature_tag_bigram[tuple(new_key)]=i
            filename = "features_dict_contextual_bigram_sen_num_"+str(self.num_of_sentences)+"_threshold_"+str(self.word_tag_threshold)
            self.save_to_pickle(self.feature_tag_bigram, filename)
        if self.setup == "contextual_trigram" or self.setup == "contextual_all": 
            for i in range(0,len(self.frequent_word_tags_3gram_dict)):
                new_key = self.frequent_word_tags_3gram_dict.keys()[i]
                self.feature_tag_trigram[tuple(new_key)]=i
            filename = "features_dict_contextual_trigram_sen_num_"+str(self.num_of_sentences)+"_threshold_"+str(self.word_tag_threshold)
            self.save_to_pickle(self.feature_tag_trigram, filename)
            
    def save_to_pickle(self,d, filename):
        with open(filename, 'wb') as handle:
                cPickle.dump(d, handle)
        handle.close()        
    
    def read_features_dict_for_test(self):
        if self.setup == "contextual_unigram" or self.setup == "contextual_all" or self.setup == "smoothing_contextual" or self.setup == "linear_inter":
            self.feature_tag_unigram = cPickle.load( open( "features_dict_contextual_unigram_sen_num_"+str(self.num_of_sentences)+"_threshold_"+str(self.contextual_unigram_threshold), "rb" ) )
        if self.setup == "contextual_bigram" or self.setup == "contextual_all" or self.setup == "smoothing_contextual" or self.setup == "linear_inter":
            self.feature_tag_bigram = cPickle.load( open( "features_dict_contextual_bigram_sen_num_"+str(self.num_of_sentences)+"_threshold_"+str(self.contextual_bigram_threshold), "rb" ) )    
        if self.setup == "contextual_trigram" or self.setup == "contextual_all" or self.setup == "smoothing_contextual" or self.setup == "linear_inter": 
            self.feature_tag_trigram = cPickle.load( open( "features_dict_contextual_trigram_sen_num_"+str(self.num_of_sentences)+"_threshold_"+str(self.contextual_trigram_threshold), "rb" ) )
                         
    def apply_word_tags_features(self,d,list_of_sentences):
        self.extract_word_tag_features(d)
        
        for func in self.apply_feature_function_dict[self.setup]:
            self.num_of_feature_func += func(list_of_sentences)
        self.set_contextual_features_dict()
    
    def get_word_tags_unigram_index(self,t,t_1,t_2,word,setup):
        index = []
        if self.feature_tag_unigram.has_key((word,t)):
            index.append(self.feature_tag_unigram[(word,t)])
        return index
        
    def get_word_tags_bigram_index(self,t,t_1,t_2,word,setup):
        index = []
        if self.feature_tag_bigram.has_key((word,t_1,t)):
            index.append(self.feature_tag_bigram[(word,t_1,t)])
        if setup == "contextual_all": #divert the index
            index = [ind+self.num_of_contextual_unigram_features for ind in index]    
        return index
     
    def get_word_tags_trigram_index(self,t,t_1,t_2,word,setup):
        index = []
        if self.feature_tag_trigram.has_key((word,t,t_1,t_2)):
            index.append(self.feature_tag_trigram[(word,t,t_1,t_2)])
        if setup == "contextual_all": #divert the index
            index = [ind + self.num_of_contextual_unigram_features +\
                     self.num_of_contextual_bigram_features for ind in index]    
        return index
    
    def get_word_tags_index(self,t,t_1,t_2,word):
        index = []
        index.append(self.get_word_tags_unigram_index(t,t_1,t_2,word))
        index.append(self.get_word_tags_bigram_index(t,t_1,t_2,word))
        index.append(self.get_word_tags_trigram_index(t,t_1,t_2,word))
        return index
    #     1. advanced_setup - Set of morphological features for all prefixes/suffixes and trigram tags
    
    def apply_morphological_word_tags_features(self,d):
        self.extract_morphological_features(d)
        if self.setup == "morphological_all":
            self.num_of_feature_func = self.num_of_morphological_unigram_features+self.num_of_morphological_bigram_features+self.num_of_morphological_trigram_features
        elif self.setup == "morphological_unigram":
            self.num_of_feature_func = self.num_of_morphological_unigram_features
        elif self.setup == "morphological_bigram":
            self.num_of_feature_func = self.num_of_morphological_bigram_features
        elif self.setup == "morphological_trigram":
            self.num_of_feature_func = self.num_of_morphological_trigram_features
    #LIORA 
    def curr_word_ends_with_suffix_func(self,suffix,word):
        if word.endswith(suffix) == True:
            return 1
        else:
            return 0
        
    def curr_word_starts_with_prefix_func(self,prefix,word):
        if word.startswith(prefix) == True:
            return 1
        else:
            return 0
    """
    def curr_prefix_ends_with_ing_func(self,tag,word):
        if word.endswith('ing') == True:
            return 1
        else:
            return 0
        
    def curr_prefix_ends_with_s_func(self,tag,word):
        if word.endswith('s') == True:
            return 1
        else:
            return 0
        
    def curr_prefix_ends_with_ed_func(self,tag,word):
        if word.endswith('ed') == True:
            return 1
        else:
            return 0
    
    def curr_prefix_ends_with_e_func(self,tag,word):
        if word.endswith('e') == True:
            return 1
        else:
            return 0
    
    def curr_prefix_starts_with_pre_func(self,tag,word):
        if word.startswith('pre') == True:
            return 1
        else:
            return 0

   
    def get_morphological_trigram_index(self,t,t_1,t_2,word,setup):
        index = []
        if self.curr_prefix_ends_with_ing_func(t,word) == 1:
            if self.morphological_trigram_features.has_key(("ing",t_2,t_1,t)):
                index.append(self.morphological_trigram_features[("ing",t_2,t_1,t)])
        if self.curr_prefix_starts_with_pre_func(t,word) == 1:
            if self.morphological_trigram_features.has_key(("pre",t_2,t_1,t)):
                index.append(self.morphological_trigram_features[("pre",t_2,t_1,t)])
        if self.curr_prefix_ends_with_ed_func(t,word) == 1:
            if self.morphological_trigram_features.has_key(("ed",t_2,t_1,t)):
                index.append(self.morphological_trigram_features[("ed",t_2,t_1,t)])
        if self.curr_prefix_ends_with_s_func(t,word) == 1:
            if self.morphological_trigram_features.has_key(("s",t_2,t_1,t)):
                index.append(self.morphological_trigram_features[("s",t_2,t_1,t)])
        if self.curr_prefix_ends_with_e_func(t,word) == 1:
            if self.morphological_trigram_features.has_key(("e",t_2,t_1,t)):
                index.append(self.morphological_trigram_features[("e",t_2,t_1,t)])
        if setup == "morphological_all":
            index = [ind + self.num_of_morphological_unigram_features +\
                     self.num_of_morphological_bigram_features for ind in index]
        return index
     """
    #LIORA
    def get_morphological_trigram_index(self,t,t_1,t_2,word,setup):
        try:
            index = []
            for prefix in self.prefix_list:
                if self.curr_word_starts_with_prefix_func(prefix, word):
                    if self.morphological_trigram_features.has_key((prefix,t_2,t_1,t)):
                        index.append(self.morphological_trigram_features[(prefix,t_2,t_1,t)])
            for suffix in self.suffix_list:
                if self.curr_word_ends_with_suffix_func(suffix, word):
                    if self.morphological_trigram_features.has_key((suffix,t_2,t_1,t)):
                        index.append(self.morphological_trigram_features[(suffix,t_2,t_1,t)])
            return index
        except Exception as err: 
            sys.stderr.write("problem get_morphological_trigram_index:tag", t,"t_1:",t_1,"t_2:",t_2,"word",word)     
            print err.args      
            print err 
        
    def get_morphological_bigram_index(self,t,t_1,t_2,word,setup):
        try:
            index = []
            for prefix in self.prefix_list:
                if self.curr_word_starts_with_prefix_func(prefix, word):
                    if self.morphological_bigram_features.has_key((prefix,t_1,t)):
                        index.append(self.morphological_bigram_features[(prefix,t_1,t)])
            for suffix in self.suffix_list:
                if self.curr_word_ends_with_suffix_func(suffix, word):
                    if self.morphological_bigram_features.has_key((suffix,t_1,t)):
                        index.append(self.morphological_bigram_features[(suffix,t_1,t)])
            return index
        except Exception as err: 
            sys.stderr.write("problem get_morphological_bigram_index:tag", t,"t_1:",t_1,"t_2:",t_2,"word",word)     
            print err.args      
            print err   
    def get_morphological_unigram_index(self,t,t_1,t_2,word,setup):
        try:
            index = []
            for prefix in self.prefix_list:
                if self.curr_word_starts_with_prefix_func(prefix, word):
                    if self.morphological_unigram_features.has_key((prefix,t)):
                        index.append(self.morphological_unigram_features[(prefix,t)])
            for suffix in self.suffix_list:
                if self.curr_word_ends_with_suffix_func(suffix, word):
                    if self.morphological_unigram_features.has_key((suffix,t)):
                        index.append(self.morphological_unigram_features[(suffix,t)])
            return index
        except Exception as err: 
            sys.stderr.write("problem get_morphological_unigram_index:tag", t,"t_1:",t_1,"t_2:",t_2,"word",word)     
            print err.args      
            print err      
#LIORA
    """
    def get_morphological_bigram_index(self,t,t_1,t_2,word,setup):
        index = []
        if self.curr_prefix_ends_with_ing_func(t,word)==1:
            if self.morphological_bigram_features.has_key(("ing",t_1,t)):
                index.append(self.morphological_bigram_features[("ing",t_1,t)])
        if self.curr_prefix_starts_with_pre_func(t,word)==1:
            if self.morphological_bigram_features.has_key(("pre",t_1,t)):
                index.append(self.morphological_bigram_features[("pre",t_1,t)])
        if self.curr_prefix_ends_with_ed_func(t,word) == 1:
            if self.morphological_bigram_features.has_key(("ed",t_1,t)):
                index.append(self.morphological_bigram_features[("ed",t_1,t)])
        if self.curr_prefix_ends_with_s_func(t,word) == 1:
            if self.morphological_bigram_features.has_key(("s",t_1,t)):
                index.append(self.morphological_bigram_features[("s",t_1,t)])
        if self.curr_prefix_ends_with_e_func(t,word) == 1:
            if self.morphological_bigram_features.has_key(("e",t_1,t)):
                index.append(self.morphological_bigram_features[("e",t_1,t)])
        if setup == "morphological_all": #divert the index
            index = [ind+self.num_of_morphological_unigram_features for ind in index]
        return index
"""
#LIORA

    def extract_morphological_features(self,d):
        word_tag_triple_cnt = 0
        freq_morphological_trigram_features = {}
        freq_morphological_bigram_features = {}
        freq_morphological_unigram_features = {}
        sorted_morphological_trigram_features = {}
        sorted_morphological_bigram_features = {}
        sorted_morphological_unigram_features = {}
        for (word,triple_tags_list) in d.items():
            for tag,tag1,tag2 in triple_tags_list:
                for prefix in self.prefix_list:
                    if self.curr_word_starts_with_prefix_func(prefix, word):
                        freq_morphological_trigram_features[(prefix,tag,tag1,tag2)] = word_tag_triple_cnt
                        freq_morphological_bigram_features[(prefix,tag,tag1)] = word_tag_triple_cnt
                        freq_morphological_unigram_features[(prefix,tag)] = word_tag_triple_cnt
                        word_tag_triple_cnt += 1
                for suffix in self.suffix_list:
                    if self.curr_word_ends_with_suffix_func(suffix, word):
                        freq_morphological_trigram_features[(suffix,tag,tag1,tag2)] = word_tag_triple_cnt
                        freq_morphological_bigram_features[(suffix,tag,tag1)] = word_tag_triple_cnt
                        freq_morphological_unigram_features[(suffix,tag)] = word_tag_triple_cnt
                        word_tag_triple_cnt += 1
                    
                """
                if self.curr_prefix_ends_with_ing_func(tag,word)==1:
                    freq_morphological_trigram_features[("ing",tag,tag1,tag2)] = word_tag_triple_cnt
                    freq_morphological_bigram_features[("ing",tag,tag1)] = word_tag_triple_cnt
                    freq_morphological_unigram_features[("ing",tag)] = word_tag_triple_cnt
                    word_tag_triple_cnt +=1
                if self.curr_prefix_starts_with_pre_func(tag,word)==1:
                    freq_morphological_trigram_features[("pre",tag,tag1,tag2)] = word_tag_triple_cnt
                    freq_morphological_bigram_features[("pre",tag,tag1)] = word_tag_triple_cnt
                    freq_morphological_unigram_features[("pre",tag)] = word_tag_triple_cnt
                    word_tag_triple_cnt +=1
                if self.curr_prefix_ends_with_ed_func(tag,word) == 1:
                    freq_morphological_trigram_features[("ed",tag,tag1,tag2)] = word_tag_triple_cnt
                    freq_morphological_bigram_features[("ed",tag,tag1)] = word_tag_triple_cnt
                    freq_morphological_unigram_features[("ed",tag)] = word_tag_triple_cnt
                    word_tag_triple_cnt +=1
                if self.curr_prefix_ends_with_s_func(tag,word) == 1:
                    freq_morphological_trigram_features[("s",tag,tag1,tag2)] = word_tag_triple_cnt
                    freq_morphological_bigram_features[("s",tag,tag1)] = word_tag_triple_cnt
                    freq_morphological_unigram_features[("s",tag)] = word_tag_triple_cnt
                    word_tag_triple_cnt +=1
                if self.curr_prefix_ends_with_e_func(tag,word) == 1:
                    freq_morphological_trigram_features[("e",tag,tag1,tag2)] = word_tag_triple_cnt
                    freq_morphological_bigram_features[("e",tag,tag1)] = word_tag_triple_cnt
                    freq_morphological_unigram_features[("e",tag)] = word_tag_triple_cnt
                    word_tag_triple_cnt +=1
                """
                #LIORA
        sorted_morphological_trigram_features = OrderedDict(sorted(freq_morphological_trigram_features.items(), key= lambda x: (x[1])))
        sorted_morphological_bigram_features = OrderedDict(sorted(freq_morphological_bigram_features.items(), key= lambda x: (x[1])))
        sorted_morphological_unigram_features = OrderedDict(sorted(freq_morphological_unigram_features.items(), key= lambda x: (x[1])))


        if self.setup == "morphological_unigram" or self.setup == "morphological_all":
            for i in range(0,len(sorted_morphological_unigram_features)):
                new_key = sorted_morphological_unigram_features.keys()[i]
                self.morphological_unigram_features[new_key] = i
            self.num_of_morphological_unigram_features = len(self.morphological_unigram_features)
            filename = "features_dict_morphological_unigram_sen_num_"+str(self.num_of_sentences)+"_threshold_"+str(self.word_tag_threshold)
            self.save_to_pickle(self.morphological_unigram_features, filename)
        if self.setup == "morphological_bigram" or self.setup == "morphological_all":
            for i in range(0,len(sorted_morphological_bigram_features)):
                new_key = sorted_morphological_bigram_features.keys()[i]
                self.morphological_bigram_features[tuple(new_key)]=i
            self.num_of_morphological_bigram_features = len(self.morphological_bigram_features)
            filename = "features_dict_morphological_bigram_sen_num_"+str(self.num_of_sentences)+"_threshold_"+str(self.word_tag_threshold)
            self.save_to_pickle(self.morphological_bigram_features, filename)
        if self.setup == "morphological_trigram" or self.setup == "morphological_all":
            for i in range(0,len(sorted_morphological_trigram_features)):
                new_key = sorted_morphological_trigram_features.keys()[i]
                self.morphological_trigram_features[tuple(new_key)]=i
            filename = "features_dict_morphological_trigram_sen_num_"+str(self.num_of_sentences)+"_threshold_"+str(self.word_tag_threshold)
            self.num_of_morphological_trigram_features = len(self.morphological_trigram_features)
            self.save_to_pickle(self.morphological_trigram_features, filename)
"""
    def get_morphological_unigram_index(self,t,t_1,t_2,word,setup):
        index = []
        if self.curr_prefix_ends_with_ing_func(t,word)==1:
            if self.morphological_unigram_features.has_key(("ing",t)):
                index.append(self.morphological_unigram_features[("ing",t)])
        if self.curr_prefix_starts_with_pre_func(t,word)==1:
            if self.morphological_unigram_features.has_key(("pre",t)):
                index.append(self.morphological_unigram_features[("pre",t)])
        if self.curr_prefix_ends_with_ed_func(t,word) == 1:
            if self.morphological_unigram_features.has_key(("ed",t)):
                index.append(self.morphological_unigram_features[("ed",t)])
        if self.curr_prefix_ends_with_s_func(t,word) == 1:
            if self.morphological_unigram_features.has_key(("s",t)):
                index.append(self.morphological_unigram_features[("s",t)])
        if self.curr_prefix_ends_with_e_func(t,word) == 1:
            if self.morphological_unigram_features.has_key(("e",t)):
                index.append(self.morphological_unigram_features[("e",t)])
        return index
"""