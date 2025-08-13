# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 16:48:48 2019

@author: Jihye Moon
(jihye.moon@uconn.edu)

"""
import numpy as np 
import os 

def text_open(path):
    with open(path, 'r') as f:
        data=f.read().strip().split('\n')
    return data

def data_split(key):
    return key.split('#') 
eb_names = ['IN', 'OUT', 'Word2Vec', 'GloVe',\
                  'ALBERT_i',  'ALBERT_d', 'FastText',\
                      'ELMo', 'GPT2']
class embedding_vector():
    def emb_open(self, path):
        embedding = {}
        with open(path, 'r') as f:
            data=f.read().strip().split('\n')
        for i in range(len(data)):
            strings=data[i].split(' ')[1:129]
            floats = [float(x) for x in strings]
            embedding[data[i].split(' ')[0]] = floats
        return embedding
    
    def setting(self, model_type, query):
        if query == '':
            self.query = ''
        elif query != '':
            self.query = query
        if model_type == 'IN':
            import tensorflow as tf
            import numpy as np
            sess = tf.Session()
            word2index = {} 
            path = '../embeddings/literature_embedding/'
            index2word=np.load(os.path.join(path, "name.dat"), allow_pickle=True)
            saver = tf.train.import_meta_graph(os.path.join(path, "model.ckpt.meta"))
            saver.restore(sess, (os.path.join(path, "model.ckpt"))) 
            in_matrix=sess.run('embed1:0') 
             
            index2word=index2word.tolist()
            
            words_list = dict(zip(index2word.values(), index2word.keys()))
            for i in range(len(index2word)):
                word2index[index2word[i]] = i 
                
            self.index2word= dict(zip(word2index.values(), word2index.keys()))
            self.word2index=word2index
            self.syn0norm = np.array( [v/n for v, n in zip(in_matrix, np.linalg.norm(in_matrix, ord=2, axis=1))] )

        elif model_type == 'OUT':
            import tensorflow as tf
            import numpy as np
            sess = tf.Session()
            word2index = {} 
            path = '../embeddings/literature_embedding/'
            index2word=np.load(os.path.join(path, "name.dat"), allow_pickle=True)
            saver = tf.train.import_meta_graph(os.path.join(path, "model.ckpt.meta"))
            saver.restore(sess, (os.path.join(path, "model.ckpt"))) 

            out_matrix=sess.run('nce_w:0') 
             
            index2word=index2word.tolist()
            
            words_list = dict(zip(index2word.values(), index2word.keys()))
            for i in range(len(index2word)):
                word2index[index2word[i]] = i 
                
            self.index2word= dict(zip(word2index.values(), word2index.keys()))
            self.word2index=word2index
            self.syn0norm = np.array( [v/n for v, n in zip(out_matrix, np.linalg.norm(out_matrix, ord=2, axis=1))] )

        elif model_type == 'Word2Vec':
            import gensim
            import numpy as np
            
            path = '../embeddings/word2vec/cbow'
            model = gensim.models.Word2Vec.load(path)
            matrix  = model.wv.vectors 
            
            self.words_list = model.wv.index_to_key # word lists
            index2word = self.words_list
            word2index={}

            for i in range(len(index2word)):
                word2index[index2word[i]] = i 
            self.index2word= dict(zip(word2index.values(), word2index.keys()))
            self.word2index=word2index
            index2word = self.index2word
            words_list=word2index
            self.syn0norm = np.array( [v/n for v, n in zip(matrix, np.linalg.norm(matrix, ord=2, axis=1))] )

        elif model_type == 'GloVe':
            import numpy as np
            path = '../embeddings/glove/glove.txt' #extracted embedding vectors.txt from pre_trained glove
            embedding = self.emb_open(path)
            
            matrix=list(embedding.values())
            word=list(embedding.keys())
            index2word = {}
            words_list = {}
            
            for i in range(len(word)):
                index2word[i] = word[i]
                words_list[word[i]] = i 
              
            self.index2word= dict(zip(words_list.values(), words_list.keys()))
            self.word2index=words_list
            self.syn0norm = np.array( [v/n for v, n in zip(matrix, np.linalg.norm(matrix, ord=2, axis=1))] )

        elif model_type == 'FastText':
            import numpy as np
            path = '../embeddings/fasttext/fasttext-128.vec'
            embedding = self.emb_open(path)
            
            matrix=list(embedding.values())
            word=list(embedding.keys())
            matrix=matrix[1:len(matrix)]
            word=word[1:len(word)]
            index2word = {}
            words_list = {}
            
            for i in range(len(word)):
                index2word[i] = word[i]
                words_list[word[i]] = i
  
            self.index2word= dict(zip(words_list.values(), words_list.keys()))
            self.word2index=words_list
            self.syn0norm = np.array( [v/n for v, n in zip(matrix, np.linalg.norm(matrix, ord=2, axis=1))] )
        
        elif model_type == 'ELMo':
            import numpy as np
            path = '../embeddings/elmo/'
            vector_ = np.load(path + 'elmo_embedding.dat', allow_pickle=True)
            label_ = np.load(path + 'elmo_token.dat', allow_pickle=True)
             
            label = []
            matrix = []
            for i in range(len(vector_)):
                matrix.extend(vector_[i].reshape(-1,128))
            for i in range(len(label_)):
                for j in range(len(label_[i])):
                    label.append(label_[i][j][0])
            matrix = np.array(matrix)
            label = np.array(label)

            vocab = np.load(path + 'vocab.dat', allow_pickle=True)
            vocab = vocab[0:len(label)]

            word=vocab
            index2word = {}
            words_list = {}
            
            for i in range(len(word)):
                index2word[i] = word[i]
                words_list[word[i]] = i
            
            self.label2ori = {}
            
            for i in range(len(vocab)): 
                self.label2ori[vocab[i]] = label[i] 
            self.index2word= dict(zip(words_list.values(), words_list.keys()))
            self.word2index=words_list
            self.syn0norm = np.array( [v/n for v, n in zip(matrix, np.linalg.norm(matrix, ord=2, axis=1))] )

        elif model_type == 'GPT2':
            import numpy as np

            path = '../embeddings/gpt2/'
            vector_ = np.load(path + 'gpt2_last_embedding.dat', allow_pickle=True)
            label_ = np.load(path + 'gpt2_token.dat', allow_pickle=True)

            label = []
            matrix = []
            for i in range(len(vector_)):
                matrix.extend(vector_[i].reshape(-1,128))
            for i in range(len(label_)):
                for j in range(len(label_[i])):
                    label.append(label_[i][j][0])
            matrix = np.array(matrix)
            label = np.array(label)
            vocab = np.load(path + 'vocab.dat', allow_pickle=True)
            vocab = vocab[0:len(label)]
            word=vocab
            index2word = {}
            words_list = {}
            
            for i in range(len(word)):
                index2word[i] = word[i]
                words_list[word[i]] = i
            
            self.label2ori = {}
            
            for i in range(len(vocab)): 
                self.label2ori[vocab[i]] = label[i] 
              
            self.index2word= dict(zip(words_list.values(), words_list.keys()))
            self.word2index=words_list
            self.syn0norm = np.array( [v/n for v, n in zip(matrix, np.linalg.norm(matrix, ord=2, axis=1))] )

        elif model_type == 'ALBERT_i':
            import numpy as np
            path = '../embeddings/albert_i/'

            matrix = np.load(path + 'albert_embedding.dat', allow_pickle=True)
            label = np.load(path + 'albert_token.dat', allow_pickle=True)
             
            vocab = np.load(path + 'vocab.dat', allow_pickle=True)
            vocab = vocab[0:len(label)]

            word=vocab
            index2word = {}
            words_list = {}
            
            for i in range(len(word)):
                index2word[i] = word[i]
                words_list[word[i]] = i
            
            self.label2ori = {}
            
            for i in range(len(vocab)): 
                self.label2ori[vocab[i]] = label[i] 
              
            self.index2word= dict(zip(words_list.values(), words_list.keys()))
            self.word2index=words_list
            self.syn0norm = np.array( [v/n for v, n in zip(matrix, np.linalg.norm(matrix, ord=2, axis=1))] )

        elif model_type == 'ALBERT_d':
            import numpy as np
            path = '../embeddings/albert_d/'

            matrix = np.load(path + 'albert_d_embedding.dat', allow_pickle=True)
            label = np.load(path + 'albert_d_token.dat', allow_pickle=True)
             
            vocab = np.load(path + 'vocab.dat', allow_pickle=True)
            word=vocab
            index2word = {}
            words_list = {}
            
            for i in range(len(word)):
                index2word[i] = word[i]
                words_list[word[i]] = i
            
            self.label2ori = {}
            
            for i in range(len(vocab)): 
                self.label2ori[vocab[i]] = label[i]
  
            self.index2word= dict(zip(words_list.values(), words_list.keys()))
            self.word2index=words_list
            self.syn0norm = np.array( [v/n for v, n in zip(matrix, np.linalg.norm(matrix, ord=2, axis=1))] )
        elif model_type not in eb_names:
            print('enter model names in the list')
            words_list=[];index2word=[];self.syn0norm=[];
        return words_list, index2word, self.syn0norm
    
    def compute_cosine_similarity(self,x,y):
        return (np.dot(x,y)/(np.linalg.norm(x,2)*np.linalg.norm(y,2)))

    def get_simwords(self,vec, matrix, TOPNUM):
        sim_list = np.dot(matrix, vec.T)
        word_sim_list = [ (s,w) for s, w in zip(sim_list, self.index2word)]
        word_sim_list.sort(reverse=True)
        return [ (v[1],v[0]) for v in word_sim_list[:TOPNUM]]
    
 
    def print_sim_result(self,result):
        data=''
        for w, s in result:
            print("\t",self.index2word[w], s)
            data=data+self.index2word[w]+' '
        return data 
            
    def similarity_display(self, TOPNUM):
        if self.query != '':
            keyword=self.query 
            keywords = keyword.split(" ") 
            index_keywords = []
            usable_keywords = []
            oov = []
            for k in keywords:
                if self.word2index.get(k,-1) !=-1:
                    index_keywords.append(self.word2index[k])
                    usable_keywords.append(k)
                    print('in-of-vocab: ', k, self.word2index[k])
                else:
                    print('out-of-vocab: ', k) 
                    oov.append(k)
            
            if usable_keywords!=[]:
                for k in range(len(index_keywords)):
                    if len(index_keywords)>1:
                        print(self.index2word[index_keywords[k-1]],'/',self.index2word[index_keywords[k]],'=',self.compute_cosine_similarity(self.syn0norm[index_keywords[k-1]], self.syn0norm[index_keywords[k]]))
                    
                vec_keyword = np.mean([self.syn0norm[ki] for ki in index_keywords], axis=0)
              
                _=self.print_sim_result(self.get_simwords(vec_keyword, self.syn0norm, TOPNUM)) 
