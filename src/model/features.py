'''
@author: liorab
Feature functions for the POS tagging task
'''
import numpy as np

class feature_functions():
    feature_func_list = []
     
    def __init__(self,num_of_feature_functions):
#         self.num_of_feature_functions = num_of_feature_functions
        self.features_res_vector = np.zeros(num_of_feature_functions)
        self.morphological_feature_func_list = []
        self.contextual_feature_func_list = []
        self.word_tag_feature_func_list = []
        self.create_feature_functions_set_and_weight_vector()

    def tag2_DT_tag1_JJ_curr_tag_NN_feature_func(self,tag_minus_2,tag_minus_1,curr_tag,curr_word):
        if tag_minus_2 == "DT" and tag_minus_1 =="JJ" and curr_tag == "NN":
            return 1
        else:
            return 0
    
    def curr_tag_NNP_func(self,tag_minus_2,tag_minus_1,curr_tag,curr_word):
        if curr_tag == "NNP":
            return 1 
        else:
            return 0
    
    def create_feature_functions_set_and_weight_vector(self):
        self.feature_func_list = [self.curr_tag_NNP_func,self.tag2_DT_tag1_JJ_curr_tag_NN_feature_func]
        self.features_weight_vector = np.zeros(len(self.feature_func_list))
    
    def apply_morphological_feature_functions(self,tag_minus_2,tag_minus_1,curr_tag,curr_word):
        features_res_vector = [0]*len(self.morphological_feature_func_list)
        for func in self.morphological_feature_func_list:
            features_res_vector[self.feature_func_list.index(func)] = func(tag_minus_2,tag_minus_1,curr_tag,curr_word)
        return features_res_vector
    
    def apply_contextual_feature_functions(self,tag_minus_2,tag_minus_1,curr_tag,curr_word):
        features_res_vector = [0]*len(self.contextual_feature_func_list)
        for func in self.contextual_feature_func_list:
            features_res_vector[self.feature_func_list.index(func)] = func(tag_minus_2,tag_minus_1,curr_tag,curr_word)
        return features_res_vector
    
    def apply_word_tag_feature_functions(self,tag_minus_2,tag_minus_1,curr_tag,curr_word):
        features_res_vector = [0]*len(self.word_tag_feature_func_list)
        for func in self.word_tag_feature_func_list:
            features_res_vector[self.feature_func_list.index(func)] = func(tag_minus_2,tag_minus_1,curr_tag,curr_word)
        return features_res_vector
    
#     def apply_feature_functions_with_all_tags(self,curr_tuple,POS_set):
#         for func in self.feature_func_list:
#             for POS in POS_set:
#                 self.features_res_vector[self.feature_func_list.index(func)] += func(curr_tuple,POS)
#         return self.features_res_vector