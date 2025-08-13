# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 22:02:06 2022

@author: Jihye Moon
(jihye.moon@uconn.edu)

"""
import numpy as np
import pathlib
import pandas as pd
import os

import tensorflow as tf  

from sklearn.linear_model import LogisticRegression 
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler    

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.ensemble import ExtraTreesClassifier
from skfeature.function.similarity_based import fisher_score

from imblearn.over_sampling import ADASYN, SMOTE

class dimension_reducers():
    def __int__(self):
        return None
    
    def PCA(self, X_train, X_test, X_valid, dim):
        from sklearn.decomposition import PCA 
        scaler =StandardScaler()
        pca = PCA(n_components=dim)
        C1 = pca.fit_transform(X_train)  
        C2 = pca.fit_transform(X_test)
        C3 = pca.fit_transform(X_valid)
        
        scaler.fit(C1)
        C1 = scaler.transform(C1)
        C2 = scaler.transform(C2)
        C3 = scaler.transform(C3) 
        return C1, C2, C3
    
    def UMAP(self, X_train, X_test, X_valid, dim): 
        import umap 
        scaler =StandardScaler()
        reducer = umap.UMAP(n_components=dim) 
        reducer.fit(X_train)
        B1 = reducer.transform(X_train)  
        reducer.fit(X_test)
        B2 = reducer.transform(X_test)
        reducer.fit(X_valid)
        B3 = reducer.transform(X_valid)
        
        scaler.fit(B1)
        B1 = scaler.transform(B1)
        B2 = scaler.transform(B2)
        B3 = scaler.transform(B3) 
        return B1, B2, B3
    
    def Our_DR(self, reduced_emb0, X_train, X_test, X_valid, dim):
        scaler =StandardScaler()
        A2=np.matmul(X_test, reduced_emb0) 
        A1=np.matmul(X_train, reduced_emb0)  
        A3=np.matmul(X_valid, reduced_emb0) 
        
        scaler.fit(A1)
        A1 = scaler.transform(A1)
        A2 = scaler.transform(A2)
        A3 = scaler.transform(A3) 
        return A1, A2, A3
    
class feature_selectors():
    def __int__(self):
        return None
    
    def dataTump(self, result_dir, word, name):
        f = open(result_dir+'/'+name+'logs.txt','a') 
        f.write('{}\t'.format(word))
        f.write('\n') 
        
    def H2FS_fit(self, X_train, y_train, feature_size):
        fnn = round(feature_size*0.5)
        wg=self.HFS(X_train,y_train, feature_size)
        hf_score=list(wg.values())
        hf_idx = np.argsort(hf_score).tolist()
        hf_idx = hf_idx[::-1][0:fnn]
        self.hf_idx=hf_idx
        
    def H2FS_transform(self, X):
        new_X = X[:,self.hf_idx]
        return new_X
    
    def HFS(self, X_train, y_train, feature_size):
        all_feature=X_train.shape[1]
        weights = {}
        for i in range(all_feature):
            weights[i]=0
        fis_idx, f1_idx, et_idx = self.HFS_FS(X_train, y_train)
        
        cases = [fis_idx, f1_idx, et_idx]
        fns=[round(feature_size*0.3), round(feature_size*0.4),round(feature_size*0.5)] # 30%, 40%, and 50% of all features. Refer original H2FS paper 
        count=0
        for case in cases: 
            for fn in fns: 
                selected_features=case[0:fn] 
                acc1, acc2, acc3, acc4, acc5 = self.HFS_CS(X_train[:,selected_features], y_train)
                acc=[acc1, acc2, acc3, acc4, acc5] 
                for sf in selected_features:
                    weights[sf]=sum(acc)
            count+=1
        return weights
    
    def HFS_CS(self, X_train, y_train):
        from sklearn.naive_bayes import GaussianNB
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.naive_bayes import BernoulliNB
        clf = GaussianNB()
        clf.fit(X_train, y_train)
        
        acc1= clf.score(X_train, y_train)
        neigh = KNeighborsClassifier()
        neigh.fit(X_train, y_train)
        
        acc2= neigh.score(X_train, y_train)
        clf = BernoulliNB() 
        clf.fit(X_train, y_train)
        
        acc3= clf.score(X_train, y_train)
        clf = DecisionTreeClassifier()
        clf.fit(X_train, y_train)
        
        acc4= clf.score(X_train, y_train)
        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)
        acc5= clf.score(X_train, y_train)
        return acc1, acc2, acc3, acc4, acc5
    
    def HFS_FS(self, X_train, y_train, fn=282): 
        fis_idx = fisher_score.fisher_score(X_train, y_train, mode='rank') #returns rank directly instead of fisher score. so no need for feature_ranking
        fis_idx=fis_idx[0:fn]
        
        f1_clf=SelectKBest(f_regression, k=fn).fit(X_train,y_train)
        f1_score=f1_clf.scores_ 
        f1_idx = np.argsort(f1_score).tolist()
        f1_idx = f1_idx[::-1][0:fn]
        
        rnd_clf = ExtraTreesClassifier()
        rnd_clf.fit(X_train, y_train) 
        et_score=rnd_clf.feature_importances_ 
        et_idx = np.argsort(et_score).tolist()
        et_idx = f1_idx[::-1][0:fn]
        return fis_idx, f1_idx, et_idx
      
    def Our_FS(self, emb2simi, name, embedding_list, variables_indexing, disease_variables_indexing, additional_dictionary, embedding, target_embedding_list, index2target, index2variables, target_embedding, feature_size, result_dir): 
        gene_name = '../gene_name_info/query_full_name'; gene_symb='../gene_name_info/query_symbol' 
        _, embed_name = emb2simi.target2variable(" ".join(list(disease_variables_indexing.keys())), target_embedding, target_embedding_list, embedding, embedding_list, index2variables, variables_indexing, feature_size)
        df = pd.DataFrame(embed_name)
        df.to_csv(os.path.join(result_dir, name+'.csv'), index=False)  
        print('Selected features by our FS was saved in' , result_dir)

        return embed_name
    def RF(self, ix, X_train, y_train, X_test, y_test, names, result_dir):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import GridSearchCV 
        rnd_grid = [
            {'n_estimators': [128, 256, 384], 'max_features': [128]}, 
            ]
        rnd_clf = RandomForestClassifier()
        grid_search3 = GridSearchCV(rnd_clf, rnd_grid, cv=None, scoring='accuracy', return_train_score=True)
        grid_search3.fit(X_train, y_train)
        best_param=grid_search3.best_params_
      
        rnd_clf = RandomForestClassifier(**best_param)
        rnd_clf.fit(X_train, y_train)
        values=rnd_clf.feature_importances_ 
        indices = np.argsort(values).tolist()
        indices = indices[::-1]
        for i in range(len(indices)):
            rlt = str(names[indices[i]])+' '+str(values[indices[i]])+' '+str(indices[i])
            self.dataTump(result_dir, rlt,ix+' RF') 
        print('Selected features by RF was saved in' , result_dir)
        return names, values, indices

    def DT(self, ix, X_train, y_train, X_test, y_test, names, result_dir):
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.model_selection import GridSearchCV 
        rnd_grid = [
            {'max_features': [128], 'max_depth':[3, 5], 'max_leaf_nodes':[3, 5]}, 
            ] 
        rnd_clf = DecisionTreeClassifier()
        grid_search3 = GridSearchCV(rnd_clf, rnd_grid, cv=None, scoring='accuracy', return_train_score=True)
        grid_search3.fit(X_train, y_train)
        best_param=grid_search3.best_params_
    
        rnd_clf = DecisionTreeClassifier(**best_param)
        rnd_clf.fit(X_train, y_train)
        values=rnd_clf.feature_importances_ 
        indices = np.argsort(values).tolist()
        indices = indices[::-1]
        for i in range(len(indices)):
            rlt = str(names[indices[i]])+' '+str(values[indices[i]])+' '+str(indices[i]) 
            self.dataTump(result_dir, rlt,ix+' DT')
        print('Selected features by DT was saved in' , result_dir)
        return names, values, indices

class predictors():
    def __init__(self):
        return None
    
    def reset_graph(self, seed=42):
      tf.reset_default_graph()
      tf.set_random_seed(seed)
      np.random.seed(seed)
      
    def batch(self, X, y, batch_size, name='batch'):  
        n_size=len(X)
        rd_idx = np.random.permutation(n_size)  
        n_batches = n_size // batch_size
        for idx in np.array_split(rd_idx, n_batches):
            X_batch, y_batch = X[idx], y[idx]
            yield X_batch, y_batch
            
    def softmax(self, sx, name='softmax'):  
        sfxmax=[]
        for i in range(len(sx)):
            sfxmax.append((np.exp(sx[i])/np.sum(np.exp(sx),axis=1)))
        return sfxmax 

    def CNN_train(self, X_train, _y_train, X_test, _y_test, X_valid, _y_valid, n_inputs_label=2):   
        X_train = X_train.reshape(-1, X_train.shape[1], 1) 
        X_test = X_test.reshape(-1, X_test.shape[1], 1)
        X_valid = X_valid.reshape(-1, X_valid.shape[1], 1)
            
        _y_train = np.squeeze(_y_train)
        _y_test = np.squeeze(_y_test)  
        _y_valid = np.squeeze(_y_valid) 
             
        n_outputs = 2
    
        print("Class: ",n_outputs)
    
        learning_rate = 0.001 
    
        self.reset_graph()
    
        channels = 1
        n_inputs = n_inputs_label
        print(n_inputs)
    
        conv1_fmaps = 16  
        conv1_ksize = [3]
        conv1_stride = [2] 
    
        conv_pad = "SAME"  
        n_fc1 = 64  
        n_outputs = 2  
             
        folder_path="CNN_model"
        pathlib.Path(folder_path).mkdir(parents=True, exist_ok=True)  
    
        graph = tf.Graph()
    
        with graph.as_default():
            
            with tf.name_scope("inputs"):
                input_X = tf.placeholder(tf.float32, shape=[None, n_inputs, channels], name="X") 
                input_y = tf.placeholder(tf.int32, shape=[None], name="y")
                keep_prob = tf.placeholder(tf.float32) 
    
            with tf.name_scope("conv"):  
                conv1 = tf.layers.conv1d(input_X, filters=conv1_fmaps, kernel_size=conv1_ksize,
                             strides=conv1_stride, padding=conv_pad,
                             activation=tf.nn.elu, name="conv1") 
                pool1 = tf.layers.max_pooling1d(conv1, pool_size=2, strides=1, padding='SAME')
                drop_out1 = tf.nn.dropout(pool1, keep_prob)
                conv2 = tf.layers.conv1d(drop_out1, filters=conv1_fmaps, kernel_size=conv1_ksize,
                             strides=conv1_stride, padding=conv_pad,
                             activation=tf.nn.elu, name="conv2")
                pool2 = tf.layers.max_pooling1d(conv2, pool_size=2, strides=1, padding='SAME')
     
            with tf.name_scope("conv2"): 
                [a,b,c] = pool2.shape
                pool8_flat = tf.reshape(conv2, shape=[-1, int(b) * int(c)])
                drop_out9 = tf.nn.dropout(pool8_flat, keep_prob)
    
            with tf.name_scope("fc1"):
                fc1 = tf.layers.dense(drop_out9, n_fc1, activation=tf.nn.relu, name="fc1") 
    
            with tf.name_scope("output"):
                logits = tf.layers.dense(fc1, n_outputs, name="output") 
                outputs=logits
                
            with tf.name_scope("train"): 
                xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=outputs, labels=input_y)
                loss = tf.reduce_mean(xentropy) 
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate) 
                training_op = optimizer.minimize(loss)
                tf.summary.scalar('loss', loss)
                merged = tf.summary.merge_all()
        
            with tf.name_scope("eval"):
                correct = tf.nn.in_top_k(outputs, input_y, 1) 
                accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    
            with tf.name_scope("init_and_save"):
                init = tf.global_variables_initializer()
                saver = tf.train.Saver()
    
        n_epochs = 100
        batch_size = 128
        
        saved_acc=[]
        valid=0;test=0;epoch_count=0;valid_Z=0;
        
        with tf.Session(graph=graph) as sess:
            writer = tf.summary.FileWriter(folder_path+'/TB', sess.graph)
            init.run()
            for epoch in range(n_epochs):        
                run_metadata = tf.RunMetadata()  
                for X_batch, y_batch in self.shuffle_batch(X_train, _y_train, batch_size):  
                    _, summary, loss_val = sess.run([training_op, merged, loss], feed_dict={input_X: X_batch, input_y: y_batch, keep_prob:0.7},run_metadata=run_metadata)
                    writer.add_summary(summary, epoch)  
                acc_batch = accuracy.eval(feed_dict={input_X: X_batch, input_y: y_batch, keep_prob:1.0})
                X_test=X_test.reshape(-1, n_inputs_label, 1)
                X_valid=X_valid.reshape(-1, n_inputs_label, 1)
                acc_test = accuracy.eval(feed_dict={input_X: X_test, input_y: _y_test, keep_prob:1.0}) 
                acc_valid = accuracy.eval(feed_dict={input_X: X_valid, input_y: _y_valid, keep_prob:1.0})  
                
                if acc_valid>valid:
                    valid=acc_valid
                    test=acc_test 
                    Z=logits.eval(feed_dict={input_X:X_test, keep_prob:1.0})
                    prob=Z
                    y_pred=np.argmax(Z, axis=1) 
                    test_prediction=y_pred
                    valid_Z=logits.eval(feed_dict={input_X:X_valid, keep_prob:1.0})
                    valid_y_pred=np.argmax(valid_Z, axis=1) 
                    valid_prediction=valid_y_pred
                saved_acc.append([acc_batch, acc_test])
            
            acc_test = accuracy.eval(feed_dict={input_X: X_test, input_y: _y_test, keep_prob:1.0})
            print("best valid :", valid, " test sets:", test)
    
            save_path = saver.save(sess, folder_path+"/CNN_model.ckpt")
            print("this model is saved to ",save_path) 
        writer.close()
        return valid_prediction, test_prediction, prob[:,1]
    
    def DNN_train2(self, X_train, y_train, X_test, y_test, X_valid, y_valid, n_inputs_label): 
        self.reset_graph()
        n_inputs = n_inputs_label
        n_layers = 10
        n_hidden1 = 100
        n_outputs = 2 
        
        learning_rate = 0.001 
        
        n_epochs = 1000 
        batch_size = 128
        
        saved_acc=[] 
        
        folder_path="DNN_model"
        pathlib.Path(folder_path).mkdir(parents=True, exist_ok=True)
    
        X = tf.placeholder(tf.float32, [None, n_inputs])
        y = tf.placeholder(tf.int64, [None])
        keep_prob = tf.placeholder(tf.float32)
        training = tf.placeholder_with_default(False, shape=(), name='training')
    
        with tf.variable_scope("dnn"): 
            for i in range(n_layers):
                layer_name="hidden"+str(i)
                if i==0:
                    hidden = tf.layers.dense(X, n_hidden1, activation=tf.nn.elu, name=layer_name)
                else:
                    hidden = tf.nn.dropout(hidden, keep_prob)
                    hidden = tf.layers.dense(X, n_hidden1, activation=tf.nn.elu, name=layer_name)
                
            logits = tf.layers.dense(hidden, n_outputs, name="outputs")
        
        with tf.variable_scope("loss"): 
            crossentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
            loss = tf.reduce_mean(crossentropy, name="loss")
    
        with tf.name_scope("train"): 
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            training_op = optimizer.minimize(loss)
    
        with tf.name_scope("eval"): 
            correct = tf.nn.in_top_k(logits, y, 1)
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    
        init = tf.global_variables_initializer() 
        saver = tf.train.Saver() 
    
        epoch_count=0; valid=0; test=0; prob=0
        with tf.Session() as sess:
            init.run()
            for epoch in range(n_epochs):
                epoch_count=epoch_count+1
                for X_batch, y_batch in self.batch(X_train, y_train, batch_size):
                    sess.run(training_op, feed_dict={X: X_batch, y:y_batch, keep_prob:1.0, training:True}) 
                acc_batch = accuracy.eval(feed_dict={X:X_batch, y:y_batch, keep_prob:1.0}) 
                acc_test = accuracy.eval(feed_dict={X: X_test, y:y_test, keep_prob:1.0}) 
                acc_valid = accuracy.eval(feed_dict={X: X_valid, y:y_valid, keep_prob:1.0}) 

                if acc_valid>=valid:
                    valid=acc_valid
                    test=acc_test 
                    Z=logits.eval(feed_dict={X:X_test, keep_prob:1.0})
                    y_pred=np.argmax(Z, axis=1) 
                    prob=Z
                    test_prediction=y_pred
                    valid_Z=logits.eval(feed_dict={X:X_valid, keep_prob:1.0})
                    valid_y_pred=np.argmax(valid_Z, axis=1) 
                    valid_prediction=valid_y_pred
                saved_acc.append([acc_batch, acc_test])
            save_path = saver.save(sess, "DNN_model/test.ckpt") 
            print("DNN model is saved to :", save_path)
            print("The best valid ", valid," test", test)
            
        return valid_prediction, test_prediction, prob[:,1]
    
    def multi_models_running(self, _X, _y, X_test, y_test):
      total_prediction=[]; total_proba=[]
      print("=================== LinearSVC")
      grid = [
            {'C': [0.01, 0.1, 1.0]}, 
            ] 
      clf = LinearSVC() 
      grid_search = GridSearchCV(clf, grid, cv=None, scoring='accuracy', return_train_score=True)
      grid_search.fit(_X, _y)
      best_param=grid_search.best_params_
      clf = LinearSVC(**best_param) 
      clf.fit(_X, _y)
      y_preds1=clf.predict(X_test) 
      total_prediction.append(y_preds1) 
      y_score2 = clf.decision_function(X_test)
      total_proba.append(y_score2)
    
      print("=================== DT") 
      grid = [
        {'max_features': [128], 'max_depth':[3, 5], 'max_leaf_nodes':[3, 5]}, 
      ]
      clf = DecisionTreeClassifier() 
      grid_search = GridSearchCV(clf, grid, cv=None, scoring='accuracy', return_train_score=True)
      grid_search.fit(_X, _y)
      best_param=grid_search.best_params_
      clf = DecisionTreeClassifier(**best_param)   
      clf.fit(_X, _y)
      y_preds3=clf.predict(X_test)
      total_prediction.append(y_preds3)
     
      if len(list(set(y_test)))>2:
          y_score4 = clf.predict_proba(X_test)
      else:
          y_score4 = clf.predict_proba(X_test)[:,1] 
    
      total_proba.append(y_score4)
      
      print("=================== RF")
      grid = [
        {'n_estimators': [128, 256, 384], 'max_features': [128]}, 
      ]
      clf = RandomForestClassifier() 
      grid_search = GridSearchCV(clf, grid, cv=None, scoring='accuracy', return_train_score=True)
      grid_search.fit(_X, _y)
      best_param=grid_search.best_params_
      clf = RandomForestClassifier(**best_param)
      clf.fit(_X, _y)
      y_preds4=clf.predict(X_test)
      total_prediction.append(y_preds4)
    
      if len(list(set(y_test)))>2:
          y_score4 = clf.predict_proba(X_test)
      else:
          y_score4 = clf.predict_proba(X_test)[:,1] 
    
      total_proba.append(y_score4)
     
      print("=================== LR")
      grid = [
            {'C': [0.01, 0.1, 1.0]}, 
            ] 
      clf = LogisticRegression() 
      grid_search = GridSearchCV(clf, grid, cv=None, scoring='accuracy', return_train_score=True)
      grid_search.fit(_X, _y)
      best_param=grid_search.best_params_
      clf = LogisticRegression(**best_param)
      clf.fit(_X, _y)
      y_preds5=clf.predict(X_test) 
       
      if len(list(set(y_test)))>2:
          y_score5 = clf.predict_proba(X_test)
      else:
          y_score5 = clf.predict_proba(X_test)[:,1]
    
      total_prediction.append(y_preds5)
      total_proba.append(y_score5)
      
      return total_prediction, total_proba
    
    def shuffle_batch(self, X, y, batch_size):
      rnd_idx = np.random.permutation(len(X))
      n_batches = len(X) // batch_size  
      for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx] 
        yield X_batch, y_batch
         
    def dumpArrayFile(self, denseList, fileName):
        np.asarray(denseList).dump(fileName + '.dat')
    
    def run_save(self, X_train, y_train, X_test, y_test, X_valid, y_valid, name, sampling, dimension, result_dir):
        if sampling=='SMOTE':
            oversample = SMOTE()
        else:
            oversample = ADASYN()
            
        Re_X_train, Re_y_train = oversample.fit_resample(X_train, y_train)
        _, prediction, prob = self.DNN_train2(Re_X_train, Re_y_train, X_test, y_test, X_valid, y_valid, dimension)
        total_prediction, total_prob = self.multi_models_running(Re_X_train, Re_y_train, X_test, y_test) 
        total_prediction.extend([prediction])    
        total_prob.extend([prob])
        _, prediction, prob = self.CNN_train(Re_X_train, Re_y_train, X_test, y_test, X_valid, y_valid, dimension)
        total_prediction.extend([prediction])
        total_prob.extend([prob])
        self.dumpArrayFile(total_prediction,result_dir+'/'+name) 
        self.dumpArrayFile(total_prob,result_dir+'/'+'prob.'+name) 
        return total_prediction, total_prob
    
    def save_label(self, y_test, name, result_dir):
        self.dumpArrayFile(y_test,result_dir+'/'+name) 
    

