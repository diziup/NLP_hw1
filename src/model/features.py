'''
@author: liorab
Feature functions for the POS tagging task
'''
from collections import OrderedDict
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
        self.num_of_contextual_unigram_features = 0
        self.num_of_contextual_bigram_features = 0
        self.num_of_contextual_trigram_features = 0
        self.num_of_contextual_features = 0
        self.num_of_morphological_features = 2
        self.num_of_word_tag_features = 0
        self.data_path = r"C:\study\technion\MSc\3rd_semester\NLP\hw1\sec2-21"
        self.input_tags_file = self.data_path+r"\sec2-21.pos"
        self.tags_list = []
        self.num_of_sentences = 1000



    # 1. medium_setup - Set of morphological features for all prefixes/suffixes
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
    
    def get_morphological_features_index(self,t,t_1,t_2,word):
        index = []
        i = 0
        if self.curr_prefix_ends_with_ing_func(t,word).equal(1):
            index[i]=1
            i+=1
        if self.curr_prefix_starts_with_pre_func(t,word).equal(1):
            index[i]=2
            i+=1
        return index
    
    def get_num_of_morphological_features(self):
        return self.num_of_morphological_features
    
    #2. advanced_setup - Set of contextual Features
    def extract_contextual_features(self):
        tags_file = open(self.input_tags_file,"rb").readlines()
        for i in range(0,self.num_of_sentences):
            for j in range(0,len(tags_file[i].split())):
                self.tags_list.append(tags_file[i].split()[j])
        unigram_features_vector = {}
        bigram_features_vector = {}
        trigram_features_vector = {}
        for ind in range(0,(len(self.tags_list)-2)):
            if unigram_features_vector.has_key(self.tags_list[ind]):
                unigram_features_vector[self.tags_list[ind]]+=1
            else:
                unigram_features_vector[self.tags_list[ind]]=1
            if bigram_features_vector.has_key((self.tags_list[ind],self.tags_list[ind+1])):
                bigram_features_vector[(self.tags_list[ind],self.tags_list[ind+1])]+=1
            else:
                bigram_features_vector[(self.tags_list[ind],self.tags_list[ind+1])] = 1
            if trigram_features_vector.has_key((self.tags_list[ind],self.tags_list[ind+1],self.tags_list[ind+2])):
                trigram_features_vector[(self.tags_list[ind],self.tags_list[ind+1],self.tags_list[ind+2])]+=1
            else:
                trigram_features_vector[(self.tags_list[ind],self.tags_list[ind+1],self.tags_list[ind+2])]=1

        self.sorted_unigram_features_freq = OrderedDict(sorted(unigram_features_vector.items(), key= lambda x: (x[1])))

        bigram_tmp = OrderedDict(sorted(bigram_features_vector.items(), key= lambda x: (x[1])))
        self.sorted_bigram_features_freq = {key:bigram_tmp[key] for key in bigram_tmp.keys()[0:500]}

        trigram_tmp = OrderedDict(sorted(trigram_features_vector.items(), key= lambda x: (x[1])))
        self.sorted_trigram_features_freq = {key:trigram_tmp[key] for key in trigram_tmp.keys()[0:500]}

       # self.sorted_trigram_features_freq = OrderedDict(sorted(trigram_features_vector.items(), key= lambda x: (x[1])))
        self.num_of_contextual_unigram_features = len(self.sorted_unigram_features_freq)
        self.num_of_contextual_bigram_features = len(self.sorted_bigram_features_freq)
        self.num_of_contextual_trigram_features = len(self.sorted_trigram_features_freq)
        self.num_of_contextual_features = len(self.sorted_unigram_features_freq)+len(self.sorted_bigram_features_freq)+len(self.sorted_trigram_features_freq)
    
    def get_num_of_contextual_features(self):
        return self.num_of_contextual_features
    
    def set_contextual_features_dict(self):
        for i in range(0,len(self.sorted_unigram_features_freq)):
            self.feature_tag_unigram[self.sorted_unigram_features_freq.keys()[i]]=i
        for i in range(0,len(self.sorted_bigram_features_freq)):
            self.feature_tag_bigram[self.sorted_bigram_features_freq.keys()[i]]=i
        for i in range(0,len(self.sorted_trigram_features_freq)):
            self.feature_tag_trigram[self.sorted_trigram_features_freq.keys()[i]]=i
            #  print "DONE"
    def get_contextual_feature_vec_indices(self,t,t_1,t_2,word):
        index= []
        if self.feature_tag_unigram.has_key(t):
            index.append(self.feature_tag_unigram[t])
        if self.feature_tag_bigram.has_key((t_1,t)):
            index.append(self.feature_tag_bigram[(t_1,t)])
        if self.feature_tag_trigram.has_key((t_2,t_1,t)):
            index.append(self.feature_tag_trigram[(t_2,t_1,t)])
        return index
    
    def apply_contextual_features(self):
        self.extract_contextual_features()
        self.set_contextual_features_dict()
    
    #3. basic_setup - get Word/tag features for all word/tag pairs 
    
    def extract_word_tag_features(self,d):
        self.sorted_word_tag_freq = d
        self.num_of_word_tag_features = len(self.sorted_word_tag_freq)
    
    def get_word_tag_features_index(self,t,t_1,t_2,word):
        index = []
        if self.sorted_word_tag_freq.has_key((word,t)):
            index.append(self.sorted_word_tag_freq[(word,t)])
        return index
    
    def apply_word_tag_features(self,d):
        self.extract_word_tag_features(d)
    
    def get_num_of_word_tag_features(self):
        return self.num_of_word_tag_features

    def get_num_of_features(self,setup):
        if setup == "word_tag":
            return self.num_of_word_tag_features
        elif setup == "morphological":
            return self.num_of_morphological_features
        elif setup == "contextual":
            return self.num_of_contextual_features