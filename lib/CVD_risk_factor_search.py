# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 13:57:25 2018

@author: Jihye Moon
"""

class run_intrisic_evaluation():
    def __int__(self):
        return None
    def setting(self, path, gene_symb):
        import loading_literature_embedding as emb
    
        emb2simi=emb.embedding_vector()  
        words_list, index2word, syn0norm, syn1norm = emb2simi.setting(path, gene_symb)
        self.emb2simi=emb2simi
    def running(self, query, output_path, Top_Words):
        self.emb2simi.similarity_display(query, output_path, Top_Words)
    
