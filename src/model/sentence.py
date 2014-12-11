'''
Created on Dec 11, 2014

@author: liorab
'''
class sentence():
    def __init__(self,sen_words,sen_tags):
        self.words = sen_words
        self.POS_tags = ['*','*'] + sen_tags
        self.length = len(sen_words)
        
    def get_word(self,at_index):
        return self.words[at_index]
    
    def get_tag(self,at_index):
        return self.POS_tags[at_index]

    