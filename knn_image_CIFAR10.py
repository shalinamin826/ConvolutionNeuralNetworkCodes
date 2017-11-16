# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 00:37:32 2017

@author: shalin
"""
import pickle
import glob
from collections import OrderedDict
import numpy as np

class KNNImageClassification:
    
    
    def unpickle(self,file):
        print("inside pickle")
        d = {}
        with open(file,'rb') as readfile:
            d = pickle.load(readfile,encoding='bytes')
        return d
    
    def unpackCIFAR10(self,datadict):
        
        xte = np.empty([10000,3072],dtype = int)
        yte = np.empty([10000],dtype = int)
    
        dtrain = {}
        l = ['images\data_batch_1','images\data_batch_2','images\data_batch_3'
             ,'images\data_batch_4','images\data_batch_5']
        dtrain = {key:datadict[key] for key in l if key in datadict}
        
        xtrl=[]
        ytrl=[]
        for key in dtrain.keys():
            print("training on....",key)
            d={}
            d = dtrain[key]
            xtrl.append(d[b'data'])
            ytrl.append(d[b'labels'])
            
        xtr1 = np.array(xtrl)
        ytr1 = np.array(ytrl)

        xtr = xtr1.reshape(xtr1.shape[0]*xtr1.shape[1],xtr1.shape[2])
        ytr = ytr1.reshape(ytr1.shape[0]*ytr1.shape[1])
        
        
        dtest={}
        dtest = datadict['images\\test_batch']
        print("testing set...")
        xte = dtest[b'data']
        yte = dtest[b'labels']
        
        return xtr,ytr,xte,yte 
    
    def train(self,xtr,ytr):
        
    #simply remember every train data. Nothing to do here
        self.x_tr = xtr
        self.y_tr = ytr
    
        
    def testing(self,xte):
        
        print("inside testing phase")
        pred_y = []
        for x in range(0,1000,1):
            print("Checking for test image number......",x)
            test_image = xte[x]
            dist = []
            for y in range(0,self.x_tr.shape[0],1):
                print("Extracting image number.......",x,y)
                train_image = self.x_tr[y]
                dist.append(np.sum(np.abs(test_image - train_image)))
        
            min_index = np.argmin(dist)
            pred_y.append(self.y_tr[min_index])
        
        return pred_y 
        
def run():
    
    knn = KNNImageClassification()
    path = "C:/Users/shalin/Desktop/Deep Learning/cs231n/cifar-10-python.tar/cifar-10-python/images/*"
    tr_files = glob.glob(path)
    datadict = OrderedDict()
    
    #Unpickle each data file into python dictionary
    for f in tr_files:
        datadict[f.split('/')[8]] = knn.unpickle(f) # unpickle method returns dict for each file
    
    #unpack dataset into 4 numpy array which contains training data/label and test data/label
    xtr,ytr,xte,yte = knn.unpackCIFAR10(datadict)
    
    #train knn model using train data/label
    knn.train(xtr,ytr)
    
    #testing phase on test data/model
    pred_y = knn.testing(xte)

    #accuracy calculation
    count = 0
    for indexi,i in enumerate(pred_y):
        for indexj,j in enumerate(yte[0:1000]):
            if indexi == indexj:
                if i == j:
                    count+=1
                    
    print('Accuracy....',(count/1000)*100,'%')
    
if __name__ == '__main__':
    run()