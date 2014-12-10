'''
@author: liorab
Feature functions for the POS tagging task
'''
import numpy as np

class feature_functions():
    feature_func_list = []
     
    def __init__(self,num_of_feature_functions):
        self.num_of_feature_functions = num_of_feature_functions
        self.features_res_vector = np.zeros(num_of_feature_functions)
        self.feature_func_list = []
        self.create_feature_functions_set_and_weight_vector()

    def tag2_DT_tag1_JJ_curr_tag_NN_feature_func(self,sentence_tuple,tag,sentence):
        if sentence_tuple.tag_position_2 == "DT" and sentence_tuple.tag_position_1 =="JJ" and tag == "NN":
            return 1
        else:
            return 0
    
    def curr_tag_NNP_func(self,sentence_tuple,tag,sentence):
        if tag == "NNP":
            return 1 
        else:
            return 0
    
    def create_feature_functions_set_and_weight_vector(self):
        self.feature_func_list = [self.curr_tag_NNP_func,self.tag2_DT_tag1_JJ_curr_tag_NN_feature_func]
        self.features_weight_vector = np.zeros(len(self.feature_func_list))
    
    def apply_feature_functions(self,curr_tuple,tag,sentence):
        features_res_vector = [0]*self.num_of_feature_functions
        for func in self.feature_func_list:
            features_res_vector[self.feature_func_list.index(func)] = func(curr_tuple,tag,sentence)
        return features_res_vector
    
#     def apply_feature_functions_with_all_tags(self,curr_tuple,POS_set):
#         for func in self.feature_func_list:
#             for POS in POS_set:
#                 self.features_res_vector[self.feature_func_list.index(func)] += func(curr_tuple,POS)
#         return self.features_res_vector