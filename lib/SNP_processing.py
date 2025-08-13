# -*- coding: utf-8 -*-
"""

@author: Jihye Moon
(jihye.moon@uconn.edu)

"""

import numpy as np

class SNP_processor():
    def setting(self):
        
        return
    def connecting_embedding_model(self, syn0norm, rsid_indexs, RsId2symble, words_list2): 
        recorded_gene=[]
        recoreded_rsid_indexs=[]
        embeding_X0 = [] 
        removal = [] 
        count=0
        for i in range(len(rsid_indexs)):
            gene = RsId2symble[rsid_indexs[i]]
            if '#' in gene:
                gene_buffer=gene.split('#')
                embedding_X0_buffer=[] 
                
                for j in range(len(gene_buffer)):
                    if words_list2.get('#'+gene_buffer[j], -2)!=-2:
                        count=1
                if count==1:
                    for j in range(len(gene_buffer)):
                        if words_list2.get('#'+gene_buffer[j], -2)!=-2:
                            index = words_list2['#'+gene_buffer[j]]  
                            embedding_X0_buffer.append(syn0norm[index]) 
                    if len(embedding_X0_buffer)>0 and len(embedding_X0_buffer)<2: 
                        buffer_e0=embedding_X0_buffer[0] 
                        embeding_X0.append(buffer_e0) 
                    if len(embedding_X0_buffer)>1:
                        buffer_e0 = embedding_X0_buffer[0] 
                        for j in range(1, len(embedding_X0_buffer)):
                            buffer_e0 = (buffer_e0+embedding_X0_buffer[j])/2 
                        embeding_X0.append(buffer_e0) 
                    if len(embedding_X0_buffer)==0:
                        removal.append(i)
                        #print(i, 'missing: ', gene) # to show which gene has no related literature
                else:
                    #print(i, 'missing: ', gene) # to show which gene has no related literature
                    removal.append(i) 
            else:
                if words_list2.get('#'+gene, -2) != -2:
                    index = words_list2['#'+gene]
                    embeding_X0.append(syn0norm[index]) 
                else:
                    removal.append(i)
                    #print(i, 'missing: ', '#'+gene) # to show which gene has no related literature
            recorded_gene.append(gene)
            recoreded_rsid_indexs.append(rsid_indexs[i])
        embeding_X0=np.array(embeding_X0) 
         
        return embeding_X0, recorded_gene, recoreded_rsid_indexs, removal

    def removal_missing_SNP_from_literature(self, rsid_indexs, RsId2symble, words_list2): # same with connecting_embedding_model but it does not have embedding vectors 
        recorded_gene=[]
        recoreded_rsid_indexs=[]
        removal = [] 
        count=0
        for i in range(len(rsid_indexs)):
            gene = RsId2symble[rsid_indexs[i]]
            if '#' in gene:
                gene_buffer=gene.split('#')
                embedding_X0_buffer=[] 
                
                for j in range(len(gene_buffer)):
                    if words_list2.get('#'+gene_buffer[j], -2)!=-2:
                        count=1
                if count==1:
                    for j in range(len(gene_buffer)):
                        if words_list2.get('#'+gene_buffer[j], -2)!=-2:
                            index = words_list2['#'+gene_buffer[j]]  
                            embedding_X0_buffer.append(index) 
                    if len(embedding_X0_buffer)==0:
                        removal.append(i)
                else:
                    removal.append(i) 
            else:
                if words_list2.get('#'+gene, -2) != -2:
                    index = words_list2['#'+gene]
                else:
                    removal.append(i)
            recorded_gene.append(gene)
            recoreded_rsid_indexs.append(rsid_indexs[i])
         
        return recorded_gene, recoreded_rsid_indexs, removal
    
    def arranged_X(self, X, removal):
        new_X=[]
        X=X.T
        for i in range(len(X)):
            if i not in removal:
                new_X.append(np.array(X[i]))
        new_X=np.array(new_X).T
        return new_X
    
    def replacing_indexing(self, data): 
        indexing={-1:-1, 0:0, 1:1, 2:2} 
        for i in range(len(data)):
            for j in range(len(data[i])):
                if indexing.get(int(data[i][j]),-2) == -2:
                    data[i][j]=-1
                else:
                    data[i][j]=indexing[int(data[i][j])]
        return data

    def removing_missing(self, data):
        data2=data.T
        index=[]
        for i in range(len(data2)):
            buffer=((data2[i].tolist().count(-1))/len(data2.T))*100
            if buffer>5:
                index.append(i)
        return index