# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 00:16:25 2020

@author: Jihye Moon
(jihye.moon@uconn.edu)

"""

import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from io import StringIO
from sklearn.feature_extraction import stop_words
import re 
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
import Medline

cachedStopWords = stop_words.ENGLISH_STOP_WORDS

class preprocessing():
    def __init__(self, data_dir, batch_dir, final_dir, preprocessed_dir): 
        self.final_dir=final_dir
        self.preprocessed_dir=preprocessed_dir
        self.batch_dir=batch_dir
        return None
     
    def Indexing(self, name, word):
        f = open(name+'.txt','w') 
        for i in range(len(word)):
            f.write('{}\n'.format(word[i])) 
        f.close()
    
    def batch_data_matching(self, full_path, including_list):
        try:
            arr_list=[]
            dir_names = os.listdir(full_path)
            j = 0; file_names = []
            for dir_name in dir_names:
                if dir_name in including_list:
    
                    arr = []
                    i = 0
                    full_dir_name = os.path.join(full_path, dir_name)
                    if (os.path.isdir(full_dir_name)!=True):
                        continue
                    text_file_names = os.listdir(full_dir_name)
                
                    for text_file_name in text_file_names:
                        full_text_file_name = os.path.join(full_dir_name, text_file_name)
                        ext = os.path.splitext(full_text_file_name)[-1]
                        if ext == '.txt': 
                            arr.insert(i, full_text_file_name)
                            i = i+1 
                    
                    file_names.append(dir_name)
                    arr_list.insert(j, arr)
                    j = j + 1 
            return file_names, arr_list
        except PermissionError:
            pass
    
    def file_detection(self, data, name, point):
        predata=[]
        for i in range(len(data)):
            if name in data[i]:
                predata.append(data[i]) 
        missing=[];sorting=[]
        list_sorting=[]
        for i in range(len(predata)):
            buffer=predata[i].split('\\')
            buffer_num = int(buffer[len(buffer)-1].split('.')[point])
            sorting.append(buffer_num)
            list_sorting.append([buffer_num, predata[i]])
        sorting.sort()
        list_sorting.sort()
        arranged_list=[]
        for i in range(len(predata)):
            arranged_list.append(list_sorting[i][1])
            if sorting[i]!=i:
                missing.append(i) 
                break;
        return sorting, missing, arranged_list
    
    def combining_files(self, file_names, data_list, names, point):
        arr_list={}
        for i in range(len(file_names)):
            #print(file_names[i]) 
            sorting, missing, arranged_list = self.file_detection(data_list[i], names[i], point)
            counting=0
            extending=[]
            if missing==[]:
                for k in range(len(arranged_list)):
                    with open(arranged_list[k], 'r') as f:
                        data = f.read().strip()
                        data = data.split('\n')
                        if data!=['']:
                            extending.extend(data)
                    if counting==100:
                        #print(k, '/', len(arranged_list))
                        #print(data)
                        counting=0
                    counting+=1
                arr_list[file_names[i]]=extending
        return arr_list
    
    def combining_query2doc(self, file_names, data_list, names, point):
        arr_list={}
        for i in range(len(file_names)):
            #print(file_names[i]) 
            sorting, missing, arranged_list = self.file_detection(data_list[i], names[i], point)
            counting=0
            extending=[]
            if missing==[]:
                print("NONE MISSING")
                for k in range(len(arranged_list)):
                    with open(arranged_list[k], 'r') as f:
                        data = f.read()
                        data=data.split('\nPMID')
                        full_data=[]
                        for n in range(len(data)):
                            if len(data[n])>0:
                                full_data.append('\nPMID'+data[n])
                        extending.extend(full_data)
                    if counting==100:
                        #print(k, '/', len(arranged_list))
                        #print(data)
                        counting=0
                    counting+=1
                arr_list[file_names[i]]=extending
        return arr_list
    
    def Medine_mapping(self, data):  
        LR=[]; TI=[]; AB=[]; MH=[]; RN=[]; PMID=[]; DCOM=[]
        FullText=''; Meta=''
        rec_file = StringIO(data)
        medline_rec = Medline.read(rec_file)
        if 'AB' in medline_rec:
            if 'LR' in medline_rec:
                LR.append(medline_rec['LR'])  
            else:
                LR.append('.')
            if 'TI' in medline_rec:
                TI.append(medline_rec['TI']) 
            else:
                TI.append('.')
            if 'AB' in medline_rec:
                AB.append(medline_rec['AB']) 
            else:
                AB.append('.')
            if 'MH' in medline_rec:
                MH.append(medline_rec['MH']) 
            else:
                MH.append('.')
            if 'PMID' in medline_rec:
                PMID.append(medline_rec['PMID']) 
            else:
                PMID.append('.') 
            if 'DCOM' in medline_rec:
                DCOM.append(medline_rec['DCOM']) 
            else:
                DCOM.append('.') 
            if 'RN' in medline_rec:
                RN.append(medline_rec['RN']) 
            else:
                RN.append('.') 
                 
        for i in range(len(AB)):
            FullText += '#'+PMID[i]+'\t'+DCOM[i]+'\t'+LR[i]+'\t'+TI[i]+'\t'+AB[i] 
            Meta += "\t@".join(RN[i])+'\t#'.join(MH[i])
        FullText+='\n' 
        Meta+='\n' 
        return FullText, Meta
    
    def gene2doc_mapping(self, data_list):
        gene2doc={}
        total_size=len(data_list)
        for i in range(total_size):
            total_data=''
            #print(i, '/', len(data_list), round(i/total_size,2)*100)
            data = data_list[i].split('\t#')
            gene = data[0].split('\t')[1]
            data = data[1:len(data)]
            if len(data)>=1:
                for j in range(len(data)):        
                    total_data += data[j].split('\t')[3] + ' ' + data[j].split('\t')[4] 
            
            if gene2doc.get(gene,-1) == -1:
                gene2doc[gene] = total_data
            else:
                gene2doc[gene] += gene2doc[gene] + total_data + ' '
        return gene2doc

    def check_valid_word(self, word):
        if word not in cachedStopWords:#is_english_word(word) and \
            return True
        else:
            return False
        
    def stem_word(self, word):
        ps = PorterStemmer()
        return ps.stem(word)
    
    def replace_all(self, text):
        patterns= [r'[^\w\s]']
        for p in patterns:
            match= re.findall(p, text)
            for m in match:
                if m != '-': 
                    text = text.replace(m, ' ') 
        return text
    
    def replace_num(self, text):
        patterns= ['[0-9]+']
        for p in patterns:
            match= re.findall(p, text)
            for m in match:
                if ' '+m+' ' in text: 
                    text = text.replace(m, ' ')
        return text
    
    def replace_single_num(text):
        text = text.replace('-', '')
        patterns= ['[0-9]+']
        for p in patterns:
            match= re.findall(p, text)
        if len(match)>0:
            if len(text) == len(match[0]):
                single=0
            else:
                single=1
        else:
            single=1
            
        return single
       
    def removal_unwanted_pos(self, data):
        unwanted = ['IN', 'DT', 'PRP', 'RB', 'PRP$', 'WRB', 'MD', 'TO', 'RB', 'RBR', 'RBS', 'CC', 'EX'] 
        unwanted = ['IN', 'DT', 'PRP', 'PRP$', 'WRB', 'MD', 'TO', 'RBR', 'RBS', 'CC', 'EX', 'RBR', 'RBS'] 
        text=nltk.pos_tag(word_tokenize(data))
        results = ''
        for txt, pos in text:
            if pos not in unwanted:
                results+=txt+' '
        return results

    def sentence_preprocessor(self, sentence, stem=False): 
        sentence = self.removal_unwanted_pos(sentence)
        sentence = sentence.lower()   
        sentence = self.replace_all(sentence) 
        sentence = sentence.replace('.', '. ')
        sentence = re.sub('[0-9]+', '#', sentence) 
    
        new_sentence = "" 
        words = sentence.split(' ')
        for word in words: 
            if stem == True:
                word = self.stem_word(word)
            else:
                word = word
            if (self.check_valid_word(word)):
                new_sentence += word + " "
         
        
        new_sentence = new_sentence.replace(' - ', ' ') 
        new_sentence = new_sentence.replace('- ', '-# ') 
        new_sentence = new_sentence.replace(' -', ' #-') 
        
        new_sentence = new_sentence.replace(' -# ', ' ') 
        new_sentence = new_sentence.replace(' #- ', ' ') 
        new_sentence = new_sentence.replace(' -#- ', ' ') 
        
        new_sentence = new_sentence.replace(' # ', ' ') 
        new_sentence = new_sentence.replace(' - ', ' ') 
         
        new_sentence = new_sentence + ' ' 
        new_sentence = new_sentence.strip()
    
        return new_sentence
    
    def doc_preprocessor(self, sentence, stem=False): 
        sentence = self.removal_unwanted_pos(sentence)
        sentence = sentence.lower()   
        sentence = self.replace_all(sentence) 
        sentence = sentence.replace('.', '. ')
        sentence = re.sub('[0-9]+', '#', sentence) 
    
        new_sentence = "" 
        words = sentence.split(' ')
        for word in words:
            # for each word, stem it and check if it is in English dictionary and stopword or not
            if stem == True:
                word = self.stem_word(word)
            else:
                word = word
            if (self.check_valid_word(word)):
                new_sentence += word + " "
         
        new_sentence = new_sentence.replace(' - ', ' ') 
        new_sentence = new_sentence.replace('- ', '-# ') 
        new_sentence = new_sentence.replace(' -', ' #-') 
        
        new_sentence = new_sentence.replace(' -# ', ' ') 
        new_sentence = new_sentence.replace(' #- ', ' ') 
        new_sentence = new_sentence.replace(' -#- ', ' ') 
        
        new_sentence = new_sentence.replace(' # ', ' ') 
        new_sentence = new_sentence.replace(' - ', ' ') 
        
        # remove uninformative words 
        new_sentence = new_sentence + ' ' 
        new_sentence = new_sentence.strip()
    
        return new_sentence
    
    
    def making_doc_data(self, gene_list, name, dic):
        preprocessed_dir=self.preprocessed_dir
        counting=0
        handle = open(os.path.join(preprocessed_dir, name+'.data.doc.txt'), "w")
        if gene_list == None:
            for i in range(len(dic)): 
                if counting==10000:
                    print(i, '/', len(dic))
                    counting=0
                buffer = dic[i].split('\t')
                if buffer[0] != '\n':
                    buffer = buffer[3] + buffer[4]
                    if buffer != '':
                        buffer = self.doc_preprocessor(buffer) 
                        handle.write('-1' + '\t' + buffer + '\n')
                counting+=1
                
        else:
            for i in range(len(gene_list)): 
                if counting==1000:
                    print(i, '/', len(gene_list))
                    counting=0
                data = dic[gene_list[i]] 
                buffer = self.doc_preprocessor(data)
                if buffer != '':
                    handle.write('#'+ gene_list[i] + '\t' + buffer + '\n')
                counting+=1
        handle.close()

