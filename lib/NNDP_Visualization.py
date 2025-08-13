# -*- coding: utf-8 -*-
"""
Created on Sat May 20 18:22:18 2023

@author: Jihye Moon
(jihye.moon@uconn.edu)

"""
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier  
import SNP_processing as snpp
sp = snpp.SNP_processor()

class NNDP_Visualization():
    def seed_setting(self, seed):
        self.seed = seed
        
    def Visualization(self, X_train, y_train1, y_train2, X_test, y_test1, y_test2, name):   
        from sklearn.decomposition import PCA 
        import matplotlib.pyplot as plt
        import numpy as np
        import umap  
        
        pca = PCA(n_components=2)
        projected_ebd = pca.fit_transform(X_train) 
        st=time.time()
        projected_ebd_test = pca.fit_transform(X_test) 
        ed=time.time()
        print('PCA operation at testing data :', ed-st)
        
        Xax=projected_ebd[:,0]; Yax=projected_ebd[:,1] 
        Xax_test=projected_ebd_test[:,0]; Yax_test=projected_ebd_test[:,1] 
        labels=y_train1; labels2=y_train2; labels_test=y_test1; labels2_test=y_test2 
    
        cdict={0:'tab:blue',1:'tab:orange',2:'yellowgreen',3:'tab:red',4:'tab:purple'}
        labl1={1:'MI event age 20-50',0:'MI event age 50-60'} 
        labl2={0:'Black',1:'European', 2:'Unknown', 3:'Other', 4:'Hispanic'}
        
        ## PCA for MI event age group
        
        plt.figure(1, figsize=(5,5))
        fig,ax=plt.subplots(figsize=(6,4))
        fig.patch.set_facecolor('white')
        for l in np.unique(labels):
            ix=np.where(labels==l)
            ax.scatter(Xax[ix],Yax[ix],s=40,alpha=0.8,c=cdict[l],
                       label=labl1[l])   
    
        plt.legend(prop={'size': 14})      
        fig.patch.set_facecolor('white')
        for l in np.unique(labels_test):
            ix=np.where(labels_test==l)
            ax.scatter(Xax_test[ix],Yax_test[ix],s=40,alpha=0.4, c=cdict[l],
                       label=labl1[l])
        plt.xlabel("First PCA",fontsize=14)
        plt.ylabel("Second PCA",fontsize=14) 
        plt.show()     
        
        ## PCA for race
        
        plt.figure(2, figsize=(5,5))
        fig,ax=plt.subplots(figsize=(6,4))
        fig.patch.set_facecolor('white')
        for l in np.unique(labels2):
            ix=np.where(labels2==l)
            ax.scatter(Xax[ix],Yax[ix],s=40,alpha=0.8,c=cdict[l],
                       label=labl2[l])   
    
        plt.legend(prop={'size': 14})      
        fig.patch.set_facecolor('white')
        for l in np.unique(labels2_test):
            ix=np.where(labels2_test==l)
            ax.scatter(Xax_test[ix],Yax_test[ix],s=40,alpha=0.4, c=cdict[l],
                       label=labl2[l])
        plt.xlabel("First PCA",fontsize=14)
        plt.ylabel("Second PCA",fontsize=14) 
        plt.show()       
        
        ## UMAP for MI event age group
        
        plt.figure(3, figsize=(5,5))
        up = umap.UMAP(n_components=2, random_state=42)
        up.fit(X_train) 
        projected_ebd = up.transform(X_train) 
        st=time.time()
        projected_ebd_test = up.transform(X_test) 
        ed=time.time()
        print('UMAP operation at testing data :', ed-st)
        
        Xax=projected_ebd[:,0]
        Yax=projected_ebd[:,1] 
        Xax_test=projected_ebd_test[:,0]
        Yax_test=projected_ebd_test[:,1]  
        
        fig,ax=plt.subplots(figsize=(6,4))
        fig.patch.set_facecolor('white')
        for l in np.unique(labels):
            ix=np.where(labels==l)
            ax.scatter(Xax[ix],Yax[ix],s=40,alpha=0.8, c=cdict[l],
                       label=labl1[l])
    
        plt.legend(prop={'size': 14})     
        fig.patch.set_facecolor('white')
        for l in np.unique(labels_test):
            ix=np.where(labels_test==l)
            ax.scatter(Xax_test[ix],Yax_test[ix],s=40,alpha=0.4, c=cdict[l],
                       label=labl1[l])
        plt.xlabel("First UMAP",fontsize=14)
        plt.ylabel("Second UMAP",fontsize=14)
        plt.show()       
        
        ## UMAP for race
        
        plt.figure(4, figsize=(5,5)) 
        
        fig,ax=plt.subplots(figsize=(6,4))
        fig.patch.set_facecolor('white')
        for l in np.unique(labels2):
            ix=np.where(labels2==l)
            ax.scatter(Xax[ix],Yax[ix],s=40,alpha=0.8, c=cdict[l],
                       label=labl2[l])
    
        plt.legend(prop={'size': 14})     
        fig.patch.set_facecolor('white')
        for l in np.unique(labels2_test):
            ix=np.where(labels2_test==l)
            ax.scatter(Xax_test[ix],Yax_test[ix],s=40,alpha=0.4, c=cdict[l],
                       label=labl2[l])
        plt.xlabel("First UMAP",fontsize=14)
        plt.ylabel("Second UMAP",fontsize=14)
        plt.show()       

    def Literature_based_SNP(self, embeding_X0, data, full_size, indices): 
        from sklearn.preprocessing import StandardScaler 
        reduced_emb0 = embeding_X0[indices[:full_size],:] # extract 6,400 feature-related embedding vectors
        LiterSNP=np.matmul(data, reduced_emb0)
        scaler = StandardScaler()
        scaler.fit(LiterSNP)
        LiterSNP = scaler.transform(LiterSNP)
        return LiterSNP
    
    def Standardization(self, _X_train, _X_test):
        from sklearn.preprocessing import StandardScaler 
        scaler = StandardScaler()
        scaler.fit(_X_train)
        X_train = scaler.transform(_X_train)
        scaler.fit(_X_test)
        X_test = scaler.transform(_X_test) 
        return X_train, X_test
    def RF_FS(self, X_train, y_MI_train):
        rnd_clf = RandomForestClassifier(n_estimators= 128, max_features= 128, random_state=15)   
         
        rnd_clf.fit(X_train, y_MI_train) 
        values=rnd_clf.feature_importances_ 
        indices = np.argsort(values).tolist()
        indices = indices[::-1]
        return indices
    def demo_cleaning_data_raw_version(self, ori_X, y_MI, y_race, full_size): 
        index=sp.removing_missing(ori_X.T)
        X=sp.replacing_indexing(ori_X.T) 
        X=pd.DataFrame(X).T.drop(index) 
        for i, (train_index, test_index) in enumerate(StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1).split(np.array(X).T, y_MI)):
            _X_train = np.array(X).T[train_index]
            y_MI_train = y_MI[train_index]
            y_race_train = y_race[train_index]
            _X_test = np.array(X).T[test_index]
            y_MI_test = y_MI[test_index]
            y_race_test = y_race[test_index] 
            
        indices = self.RF_FS(_X_train, y_MI_train)
        _X_train = np.reshape(_X_train, (_X_train.shape[0], -1)) 
        _X_test = np.reshape(_X_test, (_X_test.shape[0], -1))
        X_train, X_test = self.Standardization(_X_train[:,indices[:full_size]], _X_test[:,indices[:full_size]])
        
        y_MI_train = np.squeeze(y_MI_train); y_MI_test = np.squeeze(y_MI_test)   
        y_race_train = np.squeeze(y_race_train); y_race_test = np.squeeze(y_race_test)   
        
        return X_train, X_test, y_MI_train, y_MI_test, y_race_train, y_race_test, indices
    
    def demo_cleaning_data_arranged_version(self, ori_X, y_MI, y_race, rsid_indexs, RsId2symble, words_list, full_size):
        recorded_gene, recoreded_rsid_indexs, removal = sp.removal_missing_SNP_from_literature(rsid_indexs, RsId2symble, words_list2=words_list)
        X=sp.arranged_X(ori_X.T, removal)
        
        index=sp.removing_missing(X)
        X=sp.replacing_indexing(X) 
        
        X=pd.DataFrame(X).T.drop(index) 
        print('data size :', X.shape)
        
        for i, (train_index, test_index) in enumerate(StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1).split(np.array(X).T, y_MI)):
            _X_train = np.array(X).T[train_index]
            y_MI_train = y_MI[train_index]
            y_race_train = y_race[train_index]
            _X_test = np.array(X).T[test_index]
            y_MI_test = y_MI[test_index]
            y_race_test = y_race[test_index] 
        
        indices = self.RF_FS(_X_train, y_MI_train)
        _X_train = np.reshape(_X_train, (_X_train.shape[0], -1)) 
        _X_test = np.reshape(_X_test, (_X_test.shape[0], -1))
        X_train, X_test = self.Standardization(_X_train[:,indices[:full_size]], _X_test[:,indices[:full_size]])
        
        y_MI_train = np.squeeze(y_MI_train); y_MI_test = np.squeeze(y_MI_test)   
        y_race_train = np.squeeze(y_race_train); y_race_test = np.squeeze(y_race_test)   
        
        return X_train, X_test, y_MI_train, y_MI_test, y_race_train, y_race_test, indices
    def demo_NNDP_version(self, ori_X, y_MI, y_race, rsid_indexs, RsId2symble, words_list, syn0norm, full_size, eb_name):
        embeding_X0, recorded_gene, recoreded_rsid_indexs, removal = sp.connecting_embedding_model(syn0norm, rsid_indexs, RsId2symble, words_list2=words_list)
        X=sp.arranged_X(ori_X.T, removal)
        
        index=sp.removing_missing(X)
        X=sp.replacing_indexing(X) 
        
        X=pd.DataFrame(X).T.drop(index)  
        
        embeding_X0 = np.array(pd.DataFrame(embeding_X0).drop(index))
        
        for i, (train_index, test_index) in enumerate(StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1).split(np.array(X).T, y_MI)):
            _X_train = np.array(X).T[train_index]
            y_MI_train = y_MI[train_index]
            y_race_train = y_race[train_index]
            _X_test = np.array(X).T[test_index]
            y_MI_test = y_MI[test_index]
            y_race_test = y_race[test_index] 
        
        indices = self.RF_FS(_X_train, y_MI_train)
        _X_train = np.reshape(_X_train, (_X_train.shape[0], -1)) 
        _X_test = np.reshape(_X_test, (_X_test.shape[0], -1))
        X_train, X_test = self.Standardization(_X_train[:,indices[:full_size]], _X_test[:,indices[:full_size]])
        
        y_MI_train = np.squeeze(y_MI_train); y_MI_test = np.squeeze(y_MI_test)   
        y_race_train = np.squeeze(y_race_train); y_race_test = np.squeeze(y_race_test)   
        A1 = self.Literature_based_SNP(embeding_X0, X_train, full_size, indices)
        A2 = self.Literature_based_SNP(embeding_X0, X_test, full_size, indices)
        self.Visualization(A1,y_MI_train, y_race_train, A2,y_MI_test, y_race_test, eb_name)

        return X_train, X_test, y_MI_train, y_MI_test, y_race_train, y_race_test, indices
    
    #def demo_Literature_SNP(self, ori_X, y_MI, y_race, rsid_indexs, RsId2symble, words_list, full_size):
    #    A1 = self.Literature_based_SNP(embeding_X0, X_train, full_size, indices)
    #    A2 = self.Literature_based_SNP(embeding_X0, X_test, full_size, indices)

    #    self.Visualization(A1,y_MI_train, y_race_train, A2,y_MI_test, y_race_test, eb_name)
    def demo_original_visualization_raw(self, ori_X, y_MI, y_race, full_size): 
        index=sp.removing_missing(ori_X.T)
        X=sp.replacing_indexing(ori_X.T) 
        X=pd.DataFrame(X).T.drop(index)
        print('data size :', X.shape)
        for i, (train_index, test_index) in enumerate(StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1).split(np.array(X).T, y_MI)):
            _X_train = np.array(X).T[train_index]
            y_MI_train = y_MI[train_index]
            y_race_train = y_race[train_index]
            _X_test = np.array(X).T[test_index]
            y_MI_test = y_MI[test_index]
            y_race_test = y_race[test_index] 
        
        indices = self.RF_FS(_X_train, y_MI_train)
        _X_train = np.reshape(_X_train, (_X_train.shape[0], -1)) 
        _X_test = np.reshape(_X_test, (_X_test.shape[0], -1))
        X_train, X_test = self.Standardization(_X_train[:,indices[:full_size]], _X_test[:,indices[:full_size]])
        
        y_MI_train = np.squeeze(y_MI_train); y_MI_test = np.squeeze(y_MI_test)   
        y_race_train = np.squeeze(y_race_train); y_race_test = np.squeeze(y_race_test)   
        
        self.Visualization(X_train,y_MI_train, y_race_train, X_test,y_MI_test, y_race_test, 'original_full')
        
    def demo_original_visualization_processed(self, ori_X, y_MI, y_race, rsid_indexs, RsId2symble, words_list, full_size):
        recorded_gene, recoreded_rsid_indexs, removal = sp.removal_missing_SNP_from_literature(rsid_indexs, RsId2symble, words_list2=words_list)
        X=sp.arranged_X(ori_X.T, removal)
        
        index=sp.removing_missing(X)
        X=sp.replacing_indexing(X) 
        
        X=pd.DataFrame(X).T.drop(index) 
        print('data size :', X.shape)
        
        for i, (train_index, test_index) in enumerate(StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1).split(np.array(X).T, y_MI)):
            _X_train = np.array(X).T[train_index]
            y_MI_train = y_MI[train_index]
            y_race_train = y_race[train_index]
            _X_test = np.array(X).T[test_index]
            y_MI_test = y_MI[test_index]
            y_race_test = y_race[test_index] 
        
        indices = self.RF_FS(_X_train, y_MI_train)
        _X_train = np.reshape(_X_train, (_X_train.shape[0], -1)) 
        _X_test = np.reshape(_X_test, (_X_test.shape[0], -1))
        X_train, X_test = self.Standardization(_X_train[:,indices[:full_size]], _X_test[:,indices[:full_size]])
        
        y_MI_train = np.squeeze(y_MI_train); y_MI_test = np.squeeze(y_MI_test)   
        y_race_train = np.squeeze(y_race_train); y_race_test = np.squeeze(y_race_test)   
        
        self.Visualization(X_train,y_MI_train, y_race_train, X_test,y_MI_test, y_race_test, 'original')
        
        return X_train, X_test, y_MI_train, y_MI_test, y_race_train, y_MI_test