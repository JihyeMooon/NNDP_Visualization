# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 18:30:00 2020

@author: Jihye Moon

"""
import os   
import pathlib 
import numpy as np 
class Gene2vec(): 
    def __init__(self, data=None, dic=None):
        self.data = data
        if dic != None:
            self.dic = dic
        else:
            self.dic = {}
        self.count = {}
        self.count['UNK']=-1
        self.dictionary={}
        
    def data_processing(self, ):
        return 0
        
    def data_loading(self, dataload):
        sent=[]; gene=[]
        with open(dataload, 'r', encoding='UTF-8') as f:
            buffer_data = f.readlines()
            for lines in buffer_data:
                stripped = lines.rstrip() 
                stripeed = stripped.split('\t')
                if len(stripeed)>1:
                    buffer = "".join(stripeed[1]).split()
                    self.vocab(buffer)
                    sent.append(buffer)
                    if stripeed[0]!=-1:
                        gene.append(stripeed[0])
        return sent, gene 
    
    def vocab(self, sent):
        for word in sent:
            if self.count.get(word, -1) != -1:
                self.count[word] +=1
            else:
                self.count[word] = 1 
                
    def vocab_save(self, name, vocab, path=None):
        if path == None:
            vocab_dir =  '/tmp/vocab'
            pathlib.Path(vocab_dir).mkdir(parents=True, exist_ok=True)
        else:
            vocab_dir = path + '/vocab'
            pathlib.Path(vocab_dir).mkdir(parents=True, exist_ok=True)
            
        handle = open(os.path.join(vocab_dir, name+'.vocab.txt'), "w")
        for k, v in vocab.items():
            handle.write(k+'\t'+str(v)+'\n')
        handle.close()
        
    def vocab_output(self): 
        vocab_dir = '/tmp/vocab'
        pathlib.Path(vocab_dir).mkdir(parents=True, exist_ok=True)
        count ={k: v for k, v in sorted(self.count.items(), key=lambda item: item[1])}
        handle = open(vocab_dir + 'vocab.txt', "w")
        for k, v in count.items():
            handle.write(k+'\t'+str(v)+'\n')
        handle.close()
        
        return count
    
    def vocab_import(self): 
        count ={k: v for k, v in sorted(self.count.items(), key=lambda item: item[1])} 
        x={v: k for k, values in count.items() for v in values}
        return count, x
    
    def selecting_vocab(self, data, appearance, min_size=1):
        if data == None:
            dic = self.dictionary
        else:
            dic = data
        new_dic = {};
        dic_num = 0
        for word in self.count: 
            if self.count[word]>appearance and len(word)>min_size: 
                new_dic[word] = dic_num
                dic_num +=1
        return new_dic
    
    def data_combine(self, sent):
        sentences=[] 

        for i in range(len(sent)):
            sentences.append("".join(sent[i]).split())  
             
    def gene_dic(self, gene, dictionary): 
        dic = dictionary.copy()
        gene_list=list(set(gene))
        for gn in gene_list:    
            if dic.get(gn, -1) == -1:
                dic[gn]=len(dic) 
                
        reversed_dic = dict(zip(dic.values(), dic.keys()))
        self.dic= dic
        self.gene_reverse_dict = reversed_dic
        return dic, reversed_dic 
     
    def sent2idx(self, sent):
        sents=[]
        for j in range(len(sent)):
            if self.dic.get(sent[j],-1) != -1:
                sents.append(self.dic[sent[j]])
        return sents

    def idx2sent(self, sent):
        sents=[]
        for j in range(len(sent)):
            if self.gene_reverse_dict.get(sent[j],-1) != -1:
                sents.append(self.gene_reverse_dict[sent[j]])
        return sents 
    
    def gene2associated_cbow(self, sents, gene, sents_count, window_size): 
        window_size = int(window_size/2) 
        saveD=[]
        preprocessing=self.sent2idx(sents)
        gene_index=gene
        prelen=len(preprocessing)
        x=[]    
        for k in range(prelen):
            if self.gene_reverse_dict.get(preprocessing[k],-1)!=-1:
                x.extend([preprocessing[k]])
        value=0
        for i in range(len(x)):
            index=i
            buffer=0
            last=0
            if i<window_size+1 and i>-1: 
                buffer=2+buffer    
                value=x[0:index+window_size+1]
            elif i< (len(x)-window_size) and i>len(x): 
                last=1+last
                value=x[index-window_size:index-last]
            else: 
                value=x[index-window_size:index+window_size+1]
            saveD.append(value)
        ix=0
        data_index=0
        sz=sum([len(saveD[j])-1 for j in range(len(saveD))])+0 
        batch = np.zeros(shape=(sz), dtype=np.int32)*-1
        labels = np.zeros(shape=(sz), dtype=np.int32)*-1
        span = 2 * window_size + 1  
        size_counting=0
        size=[]
        total_size=[]
        if data_index + span > len(x):
            data_index = 0
        for i in range(len(saveD)):
            buffer = saveD[i] 
            data_index += span
            context_words = [w for w in buffer]  
            for j in range(len(context_words)-1): 
                labels[ix+j]= gene_index  
                batch[ix + j] = context_words[j]
                size_counting+=1
            for k in range(size_counting):
                size.append(size_counting)
            total_size.extend(size)
            size=[]
            size_counting=0
            ix=ix+len(buffer)-1 
        return batch, labels, total_size 
