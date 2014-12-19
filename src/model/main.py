'''
Created on Dec 13, 2014
activate the MEMM train and test
@author: liorab
'''
import MEMM
import time
import sys

try:
    t_start = time.clock()
    setup = "contextual_bigram"
    num_of_sentences = 5000
    reg_lambda = 0.2
    threshold = 8
    MEMM_model = MEMM.MEMM(setup, num_of_sentences, reg_lambda, threshold)
#     MEMM_model.train_memm()
    MEMM_model.test_memm()
#         MEMM_model.analyze_results()
    t_end = time.clock()
    print "\n ---finished---",t_end - t_start
except Exception as err: 
        sys.stderr.write("problem")     
        print err.args      
        print err