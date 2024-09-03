# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from pyts.transformation import ROCKET
from sklearn.feature_selection import mutual_info_classif

def tROCKET (X, Xtest, y, yTest):

    # ROCKET transformation
    rocket = ROCKET(n_kernels=500, random_state=42)
    X_rocket = rocket.fit_transform(X)
    
    # Find the most discriminative kernels
    n_kernels = 60
    mutual_info = mutual_info_classif(X_rocket, y, random_state=42)
    indices = np.floor_divide(np.argsort(mutual_info), 2)[-n_kernels:]
    
    XtestRocket=rocket.transform(Xtest)
    
    #Selection the 60 desired features
    ind=indices.tolist()
    sel=list(map(lambda x: (x*2, x*2+1),ind))
    from itertools import chain
    sel=list(chain(*sel))
    
    outTrain=X_rocket[:,sel]
    outTest=XtestRocket[:,sel]
    return outTrain, outTest

workdirectory="../LocalWorkSpace/UCR/"
inputdir="Univariate2018_arff/Univariate_arff/"
datasets=pd.read_csv(workdirectory+"UCR_datasets_sel.csv")

for ind in range(112):
    dataset=datasets.iloc[ind][0]
    print(dataset)
    #Lectura train array y extraccion de Shapelets
    data=np.genfromtxt(workdirectory+inputdir+str(dataset)+"/"+str(dataset)+
				   "_TRAIN.txt",dtype=None)
    X_train=data[0:,1:]
    y_train=data[0:,0].astype(int)
	
    data=np.genfromtxt(workdirectory+inputdir+str(dataset)+"/"+str(dataset)+
					   "_TEST.txt",dtype=None)
    X_test=data[0:,1:]
    y_test=data[0:,0].astype(int)
    
    outTrain, outTest = tROCKET (X_train, X_test, y_train, y_test)
    np.savetxt(workdirectory+"ROCKET/"+dataset+"_TRAIN.csv",outTrain,delimiter=",")
    np.savetxt(workdirectory+"ROCKET/"+dataset+"_TEST.csv",outTest,delimiter=",")