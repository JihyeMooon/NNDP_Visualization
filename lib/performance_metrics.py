# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 14:17:51 2021

@author: Jihye Moon
(jihye.moon@uconn.edu)

"""
import pandas as pd
import numpy as np
import os
import scipy.stats as st

from imblearn.metrics import sensitivity_score, specificity_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score 

class performance_metrics(): 
    def saving_5folded_results(self, data, label):
        total=[]
        for i in range(len(data)):
            total.append(self.metric(data[i],label))
        return pd.DataFrame(total)
    
    def metric(self, y_pred,y_true): 
        A = accuracy_score(y_true,y_pred)
        R2 = recall_score(y_true,y_pred, average='macro')
        F2 = f1_score(y_true,y_pred, average='macro')
        P2 = precision_score(y_true,y_pred, average='macro') 
    
        SS = sensitivity_score(y_true,y_pred)
        SP = specificity_score(y_true,y_pred)
    
        return [A, P2, R2, F2, SS, SP] 
    
    def averaged_results(self, N, path, data_name):
        data={}
        for i in range(N):
            label = np.load(os.path.join(path, 'CVD_label.dat'), allow_pickle=True)
            data[i] = self.saving_5folded_results(np.load(os.path.join(path, data_name+'.dat'), allow_pickle=True),label)     
        name=['accuracy', 'macro-precision', 'macro-recall', 'macro-f1', 'sensitivity', 'specificity']
        for k in range(len(name)):
            print("=============== ", name[k] , "=================== ")
            print(self.ci2(data_name, data,k)) 
    
    def direct_averaged_results(self, N, data_name, labels, total_result):
        data={}
        for i in range(N):
            label = labels[i]
            data[i] = self.saving_5folded_results(total_result[i],label)     
        name=['accuracy', 'macro-precision', 'macro-recall', 'macro-f1', 'sensitivity', 'specificity']
        for k in range(len(name)):
            print("=============== ", name[k] , "=================== ")
            print(self.ci2(data_name, data,k))
            
    def ci2(self, name, _data, k): # 95% CI with averaged results
        print('=== ', name, ' ===')
        data=pd.concat([_data[0][k],_data[1][k],_data[2][k],_data[3][k],_data[4][k]])#.groupby(level=0))
        all_size=int(len(data)/max(data.index.tolist()))
        all_size=max(data.index.tolist())+1
        for az in range(all_size): 
            print(" ", str(round(np.mean(data.loc[az]),2))+' ('+str(round(st.t.interval(alpha=0.95, df=len(data.loc[az])-1, loc=np.mean(data.loc[az]), scale=st.sem(data.loc[az]))[0],2))+', '+str(round(st.t.interval(alpha=0.95, df=len(data.loc[az])-1, loc=np.mean(data.loc[az]), scale=st.sem(data.loc[az]))[1],2 ))+')')
        return '=================== '
