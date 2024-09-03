# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os

workdirectory="../LocalWorkSpace/UCR/"
inputdir="Univariate2018_arff/Univariate_arff/"

datasets=pd.read_csv(workdirectory+"UCR_datasets_sel.csv")

for ind in range(112):
	dataset=datasets.iloc[ind][0]
	print(dataset)

	data=np.genfromtxt(workdirectory+inputdir+str(dataset)+"/"+str(dataset)+
					   "_TRAIN.txt",dtype=None)
	X_train=data[0:,1:]
	y_train=data[0:,0].astype(int)
	
	data=np.genfromtxt(workdirectory+inputdir+str(dataset)+"/"+str(dataset)+
					   "_TEST.txt",dtype=None)
	X_test=data[0:,1:]
	y_test=data[0:,0].astype(int)
	
	from convst.classifiers import R_DST_Ridge
	from convst.utils.dataset_utils import load_sktime_dataset_split

	from sktime.datatypes._panel._convert import (
	    from_2d_array_to_nested,
	    from_nested_to_2d_array#,
	    # is_nested_dataframe,
	)

	from sktime.datatypes._panel._convert import (
	    from_3d_numpy_to_nested,
	    from_multi_index_to_3d_numpy,
	    from_nested_to_3d_numpy,
	)
	
	X_nested = from_2d_array_to_nested(X_train)
	X_train = from_nested_to_3d_numpy(X_nested)

	rdst = R_DST_Ridge(n_shapelets=10000).fit(X_train, y_train)

	X_nested = from_2d_array_to_nested(X_test)
	X_test = from_nested_to_3d_numpy(X_nested)
	
	trainX_new=rdst.transformer.transform(X_train)
	testX_new=rdst.transformer.transform(X_test)
	
	np.savetxt(workdirectory+"RDST_10000/"+dataset+"_RDSTinfo0.csv",rdst.transformer.shapelets_[0],delimiter=",")
	np.savetxt(workdirectory+"RDST_10000/"+dataset+"_RDSTinfo1.csv",rdst.transformer.shapelets_[1],delimiter=",")
	np.savetxt(workdirectory+"RDST_10000/"+dataset+"_RDSTinfo2.csv",rdst.transformer.shapelets_[2],delimiter=",")
	np.savetxt(workdirectory+"RDST_10000/"+dataset+"_RDSTinfo3.csv",rdst.transformer.shapelets_[3],delimiter=",")
	np.savetxt(workdirectory+"RDST_10000/"+dataset+"_RDSTinfo4.csv",rdst.transformer.shapelets_[4],delimiter=",")
	np.savetxt(workdirectory+"RDST_10000/"+dataset+"_RDSTTRAIN.csv",trainX_new,delimiter=",")
	np.savetxt(workdirectory+"RDST_10000/"+dataset+"_RDSTTEST.csv",testX_new,delimiter=",")
	
	d={"dataset":dataset, "acc":rdst.score(X_test, y_test)}
	out=pd.DataFrame(data=d,index=[ind])
	out.to_csv(workdirectory+"RDST_10000/"+"RDSTacc.csv", sep=",", mode='a',
			header=not os.path.exists(workdirectory+"RDST_10000/"+"RDSTacc.csv"))
    
    
def selRDST(workdirectory="datos/otro/", case="RDST_10000/", ns=200):
    for dataset in datasets:
        print(dataset)
        dataTrainRDST = pd.read_csv(workdirectory+case + dataset+"_TRAIN.csv", header=None)
        dataTestRDST = pd.read_csv(workdirectory+case  + dataset+"_TEST.csv", header=None)
        
        dataTrainRDST.columns=["V"+ str(i) for i in dataTrainRDST.columns]
        dataTestRDST.columns=["V"+ str(i) for i in dataTestRDST.columns]
        
        dataTrain=pd.read_csv(workdirectory+"ucrRDST_clean/"+dataset+"_TRAIN.csv")
        dataTest=pd.read_csv(workdirectory+"ucrRDST_clean/"+dataset+"_TEST.csv")
        
        for n in ns:
            print(n)
            outTrain=pd.concat([dataTrain.iloc[:,0:2],dataTrainRDST.iloc[:,0:n]], axis=1)
            outTest=pd.concat([dataTest.iloc[:,0:2],dataTestRDST.iloc[:,0:n]], axis=1)
    
            os.makedirs(workdirectory+"RDST_"+str(n)+"/", exist_ok=True)
            # np.savetxt(workdirectory+"RDST_"+str(n)+"/"+dataset+"_TRAIN.csv",outTrain,delimiter=",")
            # np.savetxt(workdirectory+"RDST_"+str(n)+"/"+dataset+"_TEST.csv",outTest,delimiter=",")
            
            outTrain.to_csv(workdirectory+"RDST_"+str(n)+"/"+dataset+"_TRAIN.csv",sep=",", header=True, index=False)
            outTest.to_csv(workdirectory+"RDST_"+str(n)+"/"+dataset+"_TEST.csv",sep=",", header=True, index=False)

        

selRDST(workdirectory="datos/otro/", case="RDST_10000/", ns=(3*np.array([2,4,6,8,10,60])).tolist())