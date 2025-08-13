# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 16:48:48 2019

@author: Jihye Moon
(jihye.moon@uconn.edu)

""" 
import numpy as np
import os 
import tensorflow as tf  
 
class embedding_vector(): 
    def text_open(self,path):
        with open(path, 'r') as f:
            data=f.read().strip().split('\n')
        return data
    
    def data_split(self, key):
        return key.split('#')
    
    def setting(self,path, gene_symb):
        sess = tf.Session()
        word2index = {} 
        index2word=np.load(os.path.join(path, "name.dat"), allow_pickle=True)
        saver = tf.train.import_meta_graph(os.path.join(path, "model.ckpt.meta"))
        saver.restore(sess, (os.path.join(path, "model.ckpt")))
        out_matrix=sess.run('nce_w:0') 
        in_matrix=sess.run('embed1:0') 
         
        index2word=index2word.tolist()
        
        words_list = dict(zip(index2word.values(), index2word.keys()))
        for i in range(len(index2word)):
            word2index[index2word[i]] = i 
            
        self.index2word= dict(zip(word2index.values(), word2index.keys()))
        self.word2index=word2index
        self.syn0norm = np.array( [v/n for v, n in zip(in_matrix, np.linalg.norm(in_matrix, ord=2, axis=1))] )
        self.syn1norm = np.array( [v/n for v, n in zip(out_matrix, np.linalg.norm(out_matrix, ord=2, axis=1))] )
        
        query_symbol=self.text_open(gene_symb+'.txt') #'../gene_name_info/query_symbol.txt'
        
        self.symble2name = {}
        for i in range(len(query_symbol)):
            self.symble2name[query_symbol[i]]=i

        return words_list, index2word, self.syn0norm, self.syn1norm
    
    def filtering(self, word, scores):
        word_x, word_y = word, word
        unique_set=[]
        non_single=[]
        for x in word_x:
            for y in word_y:
                if x +'s' == y:
                    non_single.append(y)
                elif x +'es' == y:
                    non_single.append(y) 
        unique_set=list(set(non_single)) 
        re_word=[]; re_score=[]
        for i in range(len(word)):
            if word[i] in unique_set:
                continue
            else:
                re_word.append(word[i])
                re_score.append(scores[i])
        return re_word, re_score
        
    def compute_cosine_similarity(self,x,y):
        return (np.dot(x,y)/(np.linalg.norm(x,2)*np.linalg.norm(y,2)))

    def get_simwords(self,vec, matrix, TOPNUM):
        sim_list = np.dot(matrix, vec.T)
        word_sim_list = [ (s,w) for s, w in zip(sim_list, self.index2word)]
        word_sim_list.sort(reverse=True)
        print(TOPNUM)
        return [ (v[1],v[0]) for v in word_sim_list[:TOPNUM]]

    def get_simgenes(self,vec, matrix, TOPNUM):
        symble2name=self.symble2name
        sim_list = np.dot(matrix, vec.T)
        word_sim_list = [ (s,w) for s, w in zip(sim_list, self.index2word)]
        word_sim_list.sort(reverse=True)
        count = 0
        results = []
        for v in word_sim_list:  
            if symble2name.get(self.index2word[v[1]].replace('#',''),-1) != -1:
                results.append((v[1],v[0]))
                if count==TOPNUM:
                    break;
                count+=1
        return results 

    def print_sim_result(self,result, query, output): 
        scores=[]; word=[]
        for w, s in result: 
            word.append(self.index2word[w])
            scores.append(s)
            
        w, s = self.filtering(word, scores)
        word=[]
        for i in range(len(w)):
            if w[i] not in query:
                print("\t",w[i], s[i]) 
                word.append(str(w[i])+' '+str(s[i])) 
            self.logs(output+' '.join(query), word)
        return None
    
    def type_similarity_display(self, output, TOPNUM):
        kw=''  
        while kw!='0': 
            kw = input("query word (exit: 0): ")
            datatype= 'm' 
            keyword=kw 
            keywords = keyword.split(" ") 
            
            if datatype=='m':
                index_keywords = [self.word2index.get(k,0) for k in keywords] 
                buffer_index_keywords=index_keywords.copy()
                index_keywords=[]
                print("==== Available Words (In-of-vocabulary):")
                for ix in buffer_index_keywords:
                    if ix!=0:
                        index_keywords.append(ix)
                        print(self.index2word.get(ix,0))
                if index_keywords ==[]:
                    print("There are no available words. Try different queries! ")
                elif index_keywords !=[]:
                    vec_keyword = np.mean([self.syn0norm[ki] for ki in index_keywords], axis=0)
                    
                    print ("=== Intrinsic Evaludation: Words ")
                    result_inin = self.get_simwords(vec_keyword, self.syn0norm, TOPNUM)
                    _ = self.print_sim_result(result_inin, keywords, output)
                    
                    print ("========")
                    print ("=== Intrinsic Evaludation: Gene Names")
                    result_inin = self.get_simgenes(vec_keyword, self.syn0norm, TOPNUM)
                    _ = self.print_sim_result(result_inin, keywords, output)

            else:
                print("Type Correctly")
                continue; 

    def similarity_display(self, kw, output, TOPNUM):
        if kw!='0':  
            keyword=kw 
            keywords = keyword.split(" ") 
            
            index_keywords = [self.word2index.get(k,0) for k in keywords]
 
            buffer_index_keywords=index_keywords.copy()
            index_keywords=[]
            print("==== Available Words (In-of-vocabulary):")
            for ix in buffer_index_keywords:
                if ix!=0:
                    index_keywords.append(ix)
                    print(self.index2word.get(ix,0))
            if index_keywords ==[]:
                print("There are no available words. Try different queries! ")
            elif index_keywords !=[]:
                vec_keyword = np.mean([self.syn0norm[ki] for ki in index_keywords], axis=0)
                    
                print ("=== Intrinsic Evaludation: Words ")
                result_inin = self.get_simwords(vec_keyword, self.syn0norm, TOPNUM)
                _ = self.print_sim_result(result_inin, keywords, output+'/word_')
                    
                print ("========")
                print ("=== Intrinsic Evaludation: Gene Names")
                result_inin = self.get_simgenes(vec_keyword, self.syn0norm, TOPNUM)
                _ = self.print_sim_result(result_inin, keywords, output+'/gene_')
 
    def variable2embed(self, words_list, syn0norm, variables_index, additional_dictionary):
        variables_lists = list(variables_index.keys())
        buffer_embedding = []
        embedding=[]
        removal = []
        embedding_list = {}
        index2variables = {}
        removed_words=[]
        for i in range(len(variables_lists)):
            buffer_embedding=[]
            words = variables_index[variables_lists[i]]
            words = words.split()
            for w in words:
                if words_list.get(w, -2)!=-2:
                    buffer_embedding.append(syn0norm[words_list[w]]) 
                else:
                    removed_words.append(w)
                if additional_dictionary.get(w, -2)!=-2:
                    buffer_embedding.append(syn0norm[words_list[additional_dictionary[w]]]) 
            if buffer_embedding==[]:
                removal.append(variables_lists[i])
            else:
                embedding.append(np.mean(buffer_embedding, axis=0))
                embedding_list[variables_lists[i]] = i
                index2variables[i] = variables_lists[i]
        self.index2variables=index2variables
        return embedding_list, index2variables, embedding, removal, removed_words
     
    def get_simvariables(self, vec, matrix, index2variables, TOPNUM):
        sim_list = np.dot(matrix, vec.T)
        word_sim_list = [ (s,w) for s, w in zip(sim_list, index2variables)]
        word_sim_list.sort(reverse=True)
        return [ (v[1],v[0]) for v in word_sim_list[:TOPNUM]]

    def logs(self, path, word):
        f = open(path+'_logs.txt','w') 
        for w in word:
            f.write('{}\n'.format(w))
        f.close()
        
    def target2variable(self, words, key_embedding, wordlist, embedding, embedding_list, index2variables,variables_indexing, TOPNUM): # variables to variables
        buffer = words.split(' ') 
        if len(buffer)==1:
            vec_keyword = key_embedding[wordlist[words]]
        else:
            vec_keyword = []
            for i in range(len(buffer)):
                if wordlist.get(buffer[i], -1)!=-1:
                    vec_keyword.append(key_embedding[wordlist[buffer[i]]])
            vec_keyword = np.array(vec_keyword)
            vec_keyword = np.mean(vec_keyword,axis=0)
        result_inin = self.get_simvariables(vec_keyword, embedding, index2variables, TOPNUM)  
        data = ''; name = []
        for w, s in result_inin: 
            data=data+index2variables[w]+' '
            name.append(index2variables[w]) 
        return data, name  
    