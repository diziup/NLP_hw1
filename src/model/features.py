'''
@author: liorab
Feature functions for the POS tagging task
'''
from collections import OrderedDict
import sys


class feature_functions():
    feature_func_list = []

    def __init__(self):
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
        self.num_of_set2_features = 0
        self.num_of_morphological_features = 2
        self.num_of_word_tag_features = 0
        self.data_path = r"C:\study\technion\MSc\3rd_semester\NLP\hw1\sec2-21"
        self.input_tags_file = self.data_path+r"\sec2-21.pos"
        self.tags_list = []
        self.num_of_sentences = 1000



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
 
    def get_num_of_set3_features(self):
        return self.num_of_set3_features
     
    def set_morphological_features(self,t,t_1,t_2,word):
        morphological_vector = {}
        for i in range(0,len(self.sorted_word_tag_freq)):
            if self.curr_prefix_ends_with_ing_func(t,word).equal(1):
                if morphological_vector.has_key(("ing",t_2,t_1,t)):
                    morphological_vector[("ing",t_2,t_1,t)]+=1      
                else:
                    morphological_vector[("ing",t_2,t_1,t)] = 1
            if self.curr_prefix_starts_with_pre_func(t,word).equal(1):
                if morphological_vector.has_key(("pre",t_2,t_1,t)):
                    morphological_vector[("pre",t_2,t_1,t)]+=1
                else:
                    morphological_vector[("pre",t_2,t_1,t)] = 1            
     
        morph_tmp = OrderedDict(sorted(morphological_vector.items(), key= lambda x: (x[1])))
        self.sorted_morphological_features = {key:morph_tmp[key] for key in morph_tmp.keys()[0:500]}
     
        for i in range(0,len(self.sorted_sorted_morphological_features)):
            self.sorted_morphological_features[self.sorted_morphological_features.keys()[i]]=i
        self.num_of_set3_features = len(self.sorted_morphological_features)
   
    def apply_set3_features(self,word,t_2,t_1,t):
        self.set_morphological_features(self,t,t_1,t_2,word)
       
    def get_set3_features(self,t,t_1,t_2,word):
        index = {}
        if self.curr_prefix_ends_with_ing_func(t,word).equal(1):
            if self.sorted_morphological_features.has_key(("ing",t_2,t_1,t)):
                if index.has_key("ing"):
                    index["ing"] = self.sorted_morphological_features[(word,t_2,t_1,t)]
                else:
                    index["ing"].append(self.sorted_morphological_features[(word,t_2,t_1,t)])
        if self.curr_prefix_starts_with_pre_func(t,word).equal(1):
            if self.sorted_morphological_features.has_key(("pre",t_2,t_1,t)):
                if index.has_key("pre"):
                    index["pre"] = self.sorted_morphological_features[(word,t_2,t_1,t)]
                else:
                    index["pre"].append(self.sorted_morphological_features[(word,t_2,t_1,t)])
        return index
#     2. medium_setup - Set of contextual Features +current word
    
    def extract_contextual_features(self,word_dict):
        try:
            tags_file = open(self.input_tags_file,"rb").readlines()
            for i in range(0,self.num_of_sentences):
                for j in range(0,len(tags_file[i].split())):
                    self.tags_list.append(tags_file[i].split()[j])
            unigram_features_vector = {}
            bigram_features_vector = {}
            trigram_features_vector = {}
            for ind in range(0,(len(self.tags_list)-2)):
                if unigram_features_vector.has_key((word_dict.keys()[ind][1],self.tags_list[ind])):
                    unigram_features_vector[(word_dict.keys()[ind][1],self.tags_list[ind])]+=1
                else:
                    unigram_features_vector[(word_dict.keys()[ind][1],self.tags_list[ind])]=1
                if bigram_features_vector.has_key((word_dict.keys()[ind][1],self.tags_list[ind],self.tags_list[ind+1])):
                    bigram_features_vector[(word_dict.keys()[ind][1],self.tags_list[ind],self.tags_list[ind+1])]+=1
                else:
                    bigram_features_vector[(word_dict.keys()[ind][1],self.tags_list[ind],self.tags_list[ind+1])] = 1
                if trigram_features_vector.has_key((word_dict.keys()[ind][1],self.tags_list[ind],self.tags_list[ind+1],self.tags_list[ind+2])):
                    trigram_features_vector[(word_dict.keys()[ind][1],self.tags_list[ind],self.tags_list[ind+1],self.tags_list[ind+2])]+=1
                else:
                    trigram_features_vector[(word_dict.keys()[ind][1],self.tags_list[ind],self.tags_list[ind+1],self.tags_list[ind+2])]=1
    
            self.sorted_unigram_features_freq = OrderedDict(sorted(unigram_features_vector.items(), key= lambda x: (x[1])))
    
            bigram_tmp = OrderedDict(sorted(bigram_features_vector.items(), key= lambda x: (x[1])))
            self.sorted_bigram_features_freq = {key:bigram_tmp[key] for key in bigram_tmp.keys()[0:500]}
    
            trigram_tmp = OrderedDict(sorted(trigram_features_vector.items(), key= lambda x: (x[1])))
            self.sorted_trigram_features_freq = {key:trigram_tmp[key] for key in trigram_tmp.keys()[0:500]}
            
            # self.sorted_trigram_features_freq = OrderedDict(sorted(trigram_features_vector.items(), key= lambda x: (x[1])))
            self.num_of_contextual_unigram_features = len(self.sorted_unigram_features_freq)
            self.num_of_contextual_bigram_features = len(self.sorted_bigram_features_freq)
            self.num_of_contextual_trigram_features = len(self.sorted_trigram_features_freq)
            self.num_of_set2_features = len(self.sorted_unigram_features_freq)+len(self.sorted_bigram_features_freq)+len(self.sorted_trigram_features_freq)
        except Exception as err: 
            sys.stderr.write("problem in compute_features_on_all_words")     
            print err.args      
            print err
        
        
    def get_num_of_set2_features(self):
        return self.num_of_set2_features
    
    def set_contextual_features_dict(self):
        for i in range(0,len(self.sorted_unigram_features_freq)):
            self.feature_tag_unigram[self.sorted_unigram_features_freq.keys()[i]]=i
        for i in range(0,len(self.sorted_bigram_features_freq)):
            self.feature_tag_bigram[self.sorted_bigram_features_freq.keys()[i]]=i
        for i in range(0,len(self.sorted_trigram_features_freq)):
            self.feature_tag_trigram[self.sorted_trigram_features_freq.keys()[i]]=i
            #  print "DONE"
    
    def get_set2_feature_vec_indices(self,t,t_1,t_2,word):
        index= []
        if self.feature_tag_unigram.has_key((word,t)):
            index.append(self.feature_tag_unigram[(word,t)])
        if self.feature_tag_bigram.has_key((word,t_1,t)):
            index.append(self.feature_tag_bigram[(word,t_1,t)])
        if self.feature_tag_trigram.has_key((word,t_2,t_1,t)):
            index.append(self.feature_tag_trigram[(word,t_2,t_1,t)])
        return index
    
    def apply_set2_features(self,d):
        self.extract_word_tag_features(d)
        self.extract_contextual_features(d)
        self.set_contextual_features_dict()
    
    #3. basic_setup - get Word/tag features for all word/tag pairs 
    
    def extract_word_tag_features(self,d):
        self.sorted_word_tag_freq = d
    
    def get_set1_features_index(self,t,t_1,t_2,word):
        index = []
        if self.sorted_word_tag_freq.has_key((word,t,t_1)):
            index.append(self.sorted_word_tag_freq[(word,t,t_1)])
        if self.feature_tag_unigram.has_key((word,t)):
            index.append(self.feature_tag_unigram[(word,t)])
        if self.feature_tag_bigram.has_key((word,t_1,t)):
            index.append(self.feature_tag_bigram[(word,t_1,t)])
        return index
    
    def apply_set1_features(self,d):
        self.extract_word_tag_features(d)
        self.extract_contextual_features(d)
        self.num_of_set1_features = len(self.sorted_unigram_features_freq)+len(self.sorted_bigram_features_freq)
        self.set_contextual_features_dict()
    
    def get_num_of_set1_features(self):
        return self.num_of_set1_features

    def get_num_of_features(self,setup):
        if setup == "advanced_set":
            return self.num_of_set3_features
        elif setup == "medium_set":
            return self.num_of_set2_features
        elif setup == "basic_set":
            return self.num_of_set1_features