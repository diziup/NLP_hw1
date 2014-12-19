'''
@author: liorab
Feature functions for the POS tagging task
'''
from collections import OrderedDict
import sys
import csv
import cPickle


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
        self.input_sentence_file = self.data_path+r"\sec2-21.words"
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
#        
#
# #     1. advanced_setup - Set of morphological features for all prefixes/suffixes and trigram tags
#     def curr_prefix_ends_with_ing_func(self,tag,word):
#         if word.endswith('ing') == True and tag=="VBG":
#             return 1
#         else:
#             return 0
#
#     def curr_prefix_starts_with_pre_func(self,tag,word):
#         if word.startswith('pre') == True and tag=="NN":
#             return 1
#         else:
#             return 0
#
#
#     def set_morphological_features(self,t,t_1,t_2,word):
#         morphological_vector = {}
#         for i in range(0,len(self.sorted_word_tag_freq)):
#             if self.curr_prefix_ends_with_ing_func(t,word).equal(1):
#                 if morphological_vector.has_key(("ing",t_2,t_1,t)):
#                     morphological_vector[("ing",t_2,t_1,t)]+=1
#                 else:
#                     morphological_vector[("ing",t_2,t_1,t)] = 1
#             if self.curr_prefix_starts_with_pre_func(t,word).equal(1):
#                 if morphological_vector.has_key(("pre",t_2,t_1,t)):
#                     morphological_vector[("pre",t_2,t_1,t)]+=1
#                 else:
#                     morphological_vector[("pre",t_2,t_1,t)] = 1
#
#         morph_tmp = OrderedDict(sorted(morphological_vector.items(), key= lambda x: (x[1])))
#         self.sorted_morphological_features = {key:morph_tmp[key] for key in morph_tmp.keys()[0:500]}
#
#         for i in range(0,len(self.sorted_sorted_morphological_features)):
#             self.sorted_morphological_features[self.sorted_morphological_features.keys()[i]]=i
#         self.num_of_set3_features = len(self.sorted_morphological_features)
#
#     def apply_set3_features(self,word,t_2,t_1,t):
#         self.set_morphological_features(self,t,t_1,t_2,word)
#
#     def get_set3_features(self,t,t_1,t_2,word):
#         index = {}
#         if self.curr_prefix_ends_with_ing_func(t,word).equal(1):
#             if self.sorted_morphological_features.has_key(("ing",t_2,t_1,t)):
#                 if index.has_key("ing"):
#                     index["ing"] = self.sorted_morphological_features[(word,t_2,t_1,t)]
#                 else:
#                     index["ing"].append(self.sorted_morphological_features[(word,t_2,t_1,t)])
#         if self.curr_prefix_starts_with_pre_func(t,word).equal(1):
#             if self.sorted_morphological_features.has_key(("pre",t_2,t_1,t)):
#                 if index.has_key("pre"):
#                     index["pre"] = self.sorted_morphological_features[(word,t_2,t_1,t)]
#                 else:
#                     index["pre"].append(self.sorted_morphological_features[(word,t_2,t_1,t)])
#         return index

#     2. word_tag setup  - Set of contextual Features + current word


    def extract_word_tag_features(self,d):
        word_tag_triple_cnt = 0
        for (word,triple_tags_list) in d.items():
            for tag,tag1,tag2 in triple_tags_list:
                self.sorted_word_tag_freq[(word,tag,tag1,tag2)] = word_tag_triple_cnt
                word_tag_triple_cnt +=1
    
    def extract_contextual_bigram_tag_with_word(self,list_of_sentences):
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
#LIORA  
#     def extract_contextual_bigram_tag_with_word(self):
#         temp_word_tag_dict = {}
#         word_2tags_list_dict = {}
# 
#         #get bigram tag with word
#         sentences_file = open(self.input_sentence_file,"rb").readlines()
#         tags_file = open(self.input_tags_file,"rb").readlines()
#         #seen_tags_set = set()
#         for i in range(0,self.num_of_sentences):
#             words = sentences_file[i].split()
#             tags = tags_file[i].split()
#             #create the word-tag counts dict
#             for index in range(0,len(words)):
#                 try:
#                     if index == 0:
#                         if (words[index],tags[index],"*") in temp_word_tag_dict.keys():
#                             temp_word_tag_dict[words[index],tags[index],"*"] += 1
#                         else:
#                             temp_word_tag_dict[words[index],tags[index],"*"] = 1
#                     else:
#                         if (words[index],tags[index],tags[index-1]) in temp_word_tag_dict.keys():
#                             temp_word_tag_dict[words[index],tags[index],tags[index-1]] += 1
#                         else:
#                             temp_word_tag_dict[words[index],tags[index],tags[index-1]] = 1
#                     #seen_tags_set.add(tags[index])
#                 except Exception as err:
#                     sys.stderr.write("problem")
#                     print err.args
#                     print err
#         #rank temp_word_tag_dict according to counts/take top k features
#         temp_word_2tag_dict_sorted = OrderedDict(sorted(temp_word_tag_dict.items(), key= lambda x: (x[1]),reverse=True))
#         for ((word,tag,tag_1),count) in temp_word_2tag_dict_sorted.items():
#             if word in word_2tags_list_dict.keys():
#                     word_2tags_list_dict[word].append((tag,tag_1))
#             else:
#                 word_2tags_list_dict[word] = [(tag,tag_1)]
#             if count >= self.word_tag_threshold:
#                 self.word_tag_threshold_counter += 1
#                 if (word,tag_1,tag) in self.frequent_word_tags_2gram_dict.keys():
#                     self.frequent_word_tags_2gram_dict[(word,tag_1,tag)]+=1
#                 else:
#                     self.frequent_word_tags_2gram_dict[(word,tag_1,tag)] = 1
#         print "self.word_tag_threshold_counter",self.word_tag_threshold_counter
#         self.num_of_word_tag_2gram_features = len(self.frequent_word_tags_2gram_dict)
#LIORA 
    
    def extract_contextual_unigram_tag_with_word(self, list_of_sentences):
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
        print "self.word_tag_threshold_counter",self.word_tag_threshold_counter
        self.num_of_word_tag_1gram_features = len(self.frequent_word_tags_1gram_dict)
        return self.num_of_word_tag_1gram_features
    
    def write_to_file(self,d): 
        w = csv.writer(open("words_frequency.csv", "w"))
        for key, val in d.items():
            w.writerow([key, val])           

#LIORA
#     def extract_contextual_unigram_tag_with_word(self):
#         temp_word_tag_dict = {}
#         word_1tags_list_dict = {}
# 
#         #get bigram tag with word
#         sentences_file = open(self.input_sentence_file,"rb").readlines()
#         tags_file = open(self.input_tags_file,"rb").readlines()
#         #seen_tags_set = set()
#         for i in range(0,self.num_of_sentences):
#             words = sentences_file[i].split()
#             tags = tags_file[i].split()
#             #create the word-tag counts dict
#             for index in range(0,len(words)):
#                 try:
#                     if (words[index],tags[index]) in temp_word_tag_dict.keys():
#                             temp_word_tag_dict[words[index],tags[index]] += 1
#                     else:
#                             temp_word_tag_dict[words[index],tags[index]] = 1
#                     #seen_tags_set.add(tags[index])
#                 except Exception as err:
#                     sys.stderr.write("problem")
#                     print err.args
#                     print err
#         #rank temp_word_tag_dict according to counts/take top k features
#         temp_word_1tag_dict_sorted = OrderedDict(sorted(temp_word_tag_dict.items(), key= lambda x: (x[1]),reverse=True))
#         for ((word,tag),count) in temp_word_1tag_dict_sorted.items():
#             if word in word_1tags_list_dict.keys():
#                 word_1tags_list_dict[word].append(tag)
#             else:
#                 word_1tags_list_dict[word] = [tag]
#             if count >= self.word_tag_threshold:
#                 self.word_tag_threshold_counter += 1
#                 if (word,tag) in self.frequent_word_tags_1gram_dict.keys():
#                     self.frequent_word_tags_1gram_dict[(word,tag)] +=1
#                 else:
#                     self.frequent_word_tags_1gram_dict[(word,tag)] = 1
# #         print "self.word_tag_threshold_counter",self.word_tag_threshold_counter
#         self.num_of_word_tag_1gram_features = len(self.frequent_word_tags_1gram_dict)
#LIORA   

    def extract_trigram_tag_with_word(self,d):
        self.frequent_word_tags_3gram_dict = self.sorted_word_tag_freq
        self.num_of_word_tag_3gram_features = len(self.frequent_word_tags_3gram_dict)
        return self.num_of_word_tag_3gram_features
    
    def extract_contextual_features_new(self,list_of_sentences):
        self.extract_contextual_unigram_tag_with_word(list_of_sentences)
        self.extract_contextual_bigram_tag_with_word(list_of_sentences)
        self.extract_trigram_tag_with_word()
    
    def set_contextual_features_dict(self):
        filename = "features_dict_"+self.setup+"_reg_lambda_"+str(self.regularization_lambda)+"_sen_num_"+str(self.num_of_sentences)+"_thredshold_"+str(self.word_tag_threshold)
        if self.setup == "contextual_unigram" or self.setup == "contextual_all":
            for i in range(0,len(self.frequent_word_tags_1gram_dict)):
                new_key = self.frequent_word_tags_1gram_dict.keys()[i]
                self.feature_tag_unigram[new_key] = i
            self.save_to_pickle(self.feature_tag_unigram, filename)
        elif self.setup == "contextual_bigram" or self.setup == "contextual_all":
            for i in range(0,len(self.frequent_word_tags_2gram_dict)):
                new_key = self.frequent_word_tags_2gram_dict.keys()[i]
                self.feature_tag_bigram[tuple(new_key)]=i
            self.save_to_pickle(self.feature_tag_bigram, filename)
        elif self.setup == "contextual_trigram" or self.setup == "contextual_all": 
            for i in range(0,len(self.frequent_word_tags_3gram_dict)):
                new_key = self.frequent_word_tags_3gram_dict.keys()[i]
                self.feature_tag_trigram[tuple(new_key)]=i
            self.save_to_pickle(self.feature_tag_trigram, filename)
            
    def save_to_pickle(self,d, filename):
        with open(filename, 'wb') as handle:
                cPickle.dump(d, handle)
        handle.close()        
    
    def read_features_dict_for_test(self):
        if self.setup == "contextual_unigram" or self.setup == "contextual_all":
            self.feature_tag_unigram = cPickle.load( open( "features_dict_"+self.setup+"_reg_lambda_"+str(self.regularization_lambda)+"_sen_num_"+str(self.num_of_sentences)+"_thredshold_"+str(self.word_tag_threshold), "rb" ) )
        elif self.setup == "contextual_bigram" or self.setup == "contextual_all":
            self.feature_tag_bigram = cPickle.load( open( "features_dict_"+self.setup+"_reg_lambda_"+str(self.regularization_lambda)+"_sen_num_"+str(self.num_of_sentences)+"_thredshold_"+str(self.word_tag_threshold), "rb" ) )    
        elif self.setup == "contextual_trigram" or self.setup == "contextual_all": 
            self.feature_tag_trigram = cPickle.load( open( "features_dict_"+self.setup+"_reg_lambda_"+str(self.regularization_lambda)+"_sen_num_"+str(self.num_of_sentences)+"_thredshold_"+str(self.word_tag_threshold), "rb" ) )
              
#LIORA
#     def extract_contextual_features_new(self):
#         self.extract_contextual_unigram_tag_with_word()
#         self.extract_contextual_bigram_tag_with_word()
#         self.extract_trigram_tag_with_word()
#  
#     def set_contextual_features_dict(self):
#         feature_tag_unigram = {}
#         feature_tag_bigram = {}
#         feature_tag_trigram = {}
#         for i in range(0,len(self.frequent_word_tags_1gram_dict)):
#             new_key = self.frequent_word_tags_1gram_dict.keys()[i]
#             feature_tag_unigram[new_key] = i
#         self.feature_tag_unigram = OrderedDict(sorted(feature_tag_unigram.items(), key= lambda x: (x[1])))
#         for i in range(0,len(self.frequent_word_tags_2gram_dict)):
#             new_key = self.frequent_word_tags_2gram_dict.keys()[i]
#             feature_tag_bigram[tuple(new_key)]=i
#         self.feature_tag_bigram = OrderedDict(sorted(feature_tag_bigram.items(), key= lambda x: (x[1])))
#         for i in range(0,len(self.frequent_word_tags_3gram_dict)):
#             new_key = self.frequent_word_tags_3gram_dict.keys()[i]
#             feature_tag_trigram[tuple(new_key)]=i
#         self.feature_tag_trigram = OrderedDict(sorted(feature_tag_trigram.items(), key= lambda x: (x[1])))
#             #  print "DONE"
# 
 #LIORA              
    def apply_word_tags_features(self,d,list_of_sentences):
        self.extract_word_tag_features(d)
        
        for func in self.apply_feature_function_dict[self.setup]:
            self.num_of_feature_func += func(list_of_sentences)
        self.set_contextual_features_dict()
    
#LIORA"
#     def apply_word_tags_features(self,d,list_of_sentences):
#         self.extract_word_tag_features(d)
#         
#         self.extract_contextual_features_new(list_of_sentences)
#         self.num_of_contextual_unigram_features = len(self.frequent_word_tags_1gram_dict)
#         self.num_of_contextual_bigram_features = len(self.frequent_word_tags_2gram_dict)
#         self.num_of_contextual_trigram_features = len(self.frequent_word_tags_3gram_dict)
#         self.num_of_word_tags_features = len(self.frequent_word_tags_3gram_dict)+len(self.frequent_word_tags_2gram_dict)+len(self.frequent_word_tags_1gram_dict)
#         self.set_contextual_features_dict()
#LIORA
#LIORA
#     def apply_word_tags_features(self,d):
#         self.extract_word_tag_features(d)
#         self.extract_contextual_features_new()
#         self.num_of_contextual_unigram_features = len(self.frequent_word_tags_1gram_dict)
#         self.num_of_contextual_bigram_features = len(self.frequent_word_tags_2gram_dict)
#         self.num_of_contextual_trigram_features = len(self.frequent_word_tags_3gram_dict)
#         self.num_of_word_tags_features = len(self.frequent_word_tags_3gram_dict)+len(self.frequent_word_tags_2gram_dict)+len(self.frequent_word_tags_1gram_dict)
#         self.set_contextual_features_dict()
#LIORA
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
    
    def curr_prefix_ends_with_ing_func(self,tag,word):
        if word.endswith('ing') == True and tag=="VBG":
            return 1
        else:
            return 0
    
    def curr_prefix_starts_with_pre_func(self,tag,word):
        if word.startswith('pre') == True and tag=="NN":
            return 1
        else:
            return 0

    def get_morphological_trigram_index(self,t,t_1,t_2,word):
        index = []
        if self.curr_prefix_ends_with_ing_func(t,word) == 1:
            if self.morphological_trigram_features.has_key(("ing",t_2,t_1,t)):
                index.append(self.morphological_trigram_features[("ing",t_2,t_1,t)])
        if self.curr_prefix_starts_with_pre_func(t,word) == 1:
            if self.morphological_trigram_features.has_key(("pre",t_2,t_1,t)):
                index.append(self.morphological_trigram_features[("pre",t_2,t_1,t)])
        return index
    
    def get_morphological_bigram_index(self,t,t_1,t_2,word):
        index = []
        if self.curr_prefix_ends_with_ing_func(t,word)==1:
            if self.morphological_bigram_features.has_key(("ing",t_1,t)):
                index.append(self.morphological_bigram_features[("ing",t_1,t)])
        if self.curr_prefix_starts_with_pre_func(t,word)==1:
            if self.morphological_bigram_features.has_key(("pre",t_1,t)):
                index.append(self.morphological_bigram_features[("pre",t_1,t)])
        return index
    
    def get_morphological_unigram_index(self,t,t_1,t_2,word):
        index = []
        if self.curr_prefix_ends_with_ing_func(t,word)==1:
            if self.morphological_unigram_features.has_key(("ing",t)):
                index.append(self.morphological_unigram_features[("ing",t)])
        if self.curr_prefix_starts_with_pre_func(t,word)==1:
            if self.morphological_unigram_features.has_key(("pre",t)):
                index.append(self.morphological_unigram_features[("pre",t)])
        return index
    
    def get_morphological_features(self,t,t_1,t_2,word):
        index = []
        index.append(self.get_morphological_trigram_index(t,t_1,t_2,word))
        index.append(self.get_morphological_bigram_index(t,t_1,t_2,word))
        index.append(self.get_morphological_unigram_index(t,t_1,t_2,word))
        return index

    def apply_morphological_features(self,d):
        self.extract_morphological_features(d)
        self.num_of_morphological_features = self.num_of_morphological_bigram_features+self.num_of_morphological_trigram_features+ self.num_of_morphological_unigram_features
    
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
                if self.curr_prefix_ends_with_ing_func(tag,word)==1:
                    freq_morphological_trigram_features[("ing",tag,tag1,tag2)] = word_tag_triple_cnt
                    freq_morphological_bigram_features[("ing",tag,tag1)] = word_tag_triple_cnt
                    freq_morphological_unigram_features[("ing",tag)] = word_tag_triple_cnt
                    word_tag_triple_cnt +=1
                elif self.curr_prefix_starts_with_pre_func(tag,word)==1:
                    freq_morphological_trigram_features[("pre",tag,tag1,tag2)] = word_tag_triple_cnt
                    freq_morphological_bigram_features[("ing",tag,tag1)] = word_tag_triple_cnt
                    freq_morphological_unigram_features[("ing",tag)] = word_tag_triple_cnt
                    word_tag_triple_cnt +=1
        sorted_morphological_trigram_features = OrderedDict(sorted(freq_morphological_trigram_features.items(), key= lambda x: (x[1])))
        sorted_morphological_bigram_features = OrderedDict(sorted(freq_morphological_bigram_features.items(), key= lambda x: (x[1])))
        sorted_morphological_unigram_features = OrderedDict(sorted(freq_morphological_unigram_features.items(), key= lambda x: (x[1])))

        for i in range(0,len(sorted_morphological_trigram_features)):
            self.morphological_trigram_features[sorted_morphological_trigram_features.keys()[i]]=i
        self.num_of_morphological_trigram_features = len(self.morphological_trigram_features)
        for i in range(0,len(sorted_morphological_bigram_features)):
            self.morphological_bigram_features[sorted_morphological_bigram_features.keys()[i]]=i
        self.num_of_morphological_bigram_features = len(self.morphological_bigram_features)
        for i in range(0,len(sorted_morphological_unigram_features)):
            self.morphological_unigram_features[sorted_morphological_unigram_features.keys()[i]]=i
        self.num_of_morphological_unigram_features = len(self.morphological_unigram_features)