'''
@author: liorab
Feature functions for the POS tagging task
'''

class feature_functions():
    feature_func_list = []

    def __init__(self):
#         self.num_of_feature_functions = num_of_feature_functions
#         self.features_res_vector = np.zeros(num_of_feature_functions)
#         self.feature_func_list = []
#         self.create_feature_functions_set_and_weight_vector()

        self.sorted_unigram_features_freq = []
        self.sorted_bigram_features_freq = []
        self.sorted_trigram_features_freq = []
        self.feature_tag_bigram = []
        self.feature_tag_trigram = []
        self.feature_tag_unigram = []
        self.sorted_word_tag_freq = []

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
        if self.curr_prefix_ends_with_ing_func(self,t,word).equal(1):
            index[1]=1
        else:
            index[1]=0
        if self.curr_prefix_starts_with_pre_func(self,t,word).equal(1):
            index[2]=1
        else:
            index[2]=0
        return index
    
#     def apply_morphological_features(self):
#         #TODO: continue!
        
    #2. advanced_setup - Set of contextual Features
    def extract_contextual_features(self,POS_set):
        unigram_features_vector = {}
        bigram_features_vector = {}
        trigram_features_vector = {}
        for ind in (len(POS_set)-2):
            if unigram_features_vector.haskey(POS_set[ind]):
                unigram_features_vector[POS_set[ind]]+=1
            else:
                unigram_features_vector[POS_set[ind]]=1
            if bigram_features_vector.haskey(POS_set[ind],POS_set[ind+1]):
                bigram_features_vector[(POS_set[ind],POS_set[ind+1])]+=1
            else:
                bigram_features_vector[(POS_set[ind],POS_set[ind+1])] = 1
            if trigram_features_vector.haskey(POS_set[ind],POS_set[ind+1],POS_set[ind+2]):
                trigram_features_vector[(POS_set[ind],POS_set[ind+1],POS_set[ind+2])]+=1
            else:
                trigram_features_vector[(POS_set[ind],POS_set[ind+1],POS_set[ind+2])]=1

        self.sorted_unigram_features_freq = sorted(unigram_features_vector.items(), key= lambda x: (x[1]))
        self.sorted_bigram_features_freq = sorted(bigram_features_vector.items(), key= lambda x: (x[1]))
        self.sorted_trigram_features_freq = sorted(trigram_features_vector.items(), key= lambda x: (x[1]))
    
    def set_contextual_features_dict(self):
        for i in len(self.sorted_unigram_features_freq):
            self.feature_tag_unigram[self.sorted_unigram_features_freq.keys()[i]]=i
        for i in len(self.sorted_bigram_features_freq):
            self.feature_tag_bigram[self.sorted_bigram_features_freq.keys()[i]]=i
        for i in len(self.sorted_trigram_features_freq):
            self.feature_tag_trigram[self.sorted_trigram_features_freq.keys()[i]]=i
        for i in len(self.sorted_word_tag_freq):
            self.feature_word_tag[self.sorted_word_tag_freq.keys()[i]]=i
    
    def get_contextual_feature_vec_indices(self,t,t_1,t_2,word):
        index= []
        if self.feature_tag_unigram.has_key(t):
            index[1]=self.feature_tag_unigram[t]
        else:
            index[1]=None
        if self.feature_tag_bigram.has_key((t-1,t)):
            index[2]=self.feature_tag_bigram[(t-1,t)]
        else:
            index[2]=None
        if self.feature_tag_trigram.has_key((t-2,t-1,t)):
            index[3]=self.feature_tag_trigram[(t-2,t-1,t)]
        else:
            index[3]=None
        return index
    
    def apply_contextual_features(self,POS_set):
        self.extract_contextual_features(POS_set)
        self.set_contextual_features_dict()
    
    #3. basic_setup - get Word/tag features for all word/tag pairs 
    
    def extract_word_tag_features(self,d):
        self.sorted_word_tag_freq = d
    
    def get_word_tag_features_index(self,t,t_1,t_2,word):
        if self.sorted_word_tag_freq.has_key((word,t)):
            index = self.sorted_word_tag_freq[(word,t)]
        else:
            index = None
        return index
    
    def apply_word_tag_features(self,d):
        self.extract_word_tag_features(d)
        
        