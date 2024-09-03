# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import sys

datasets=pd.read_csv("../LocalWorkSpace/UCR/UCR_datasets_sel.csv")["x"].tolist()

inf=sys.float_info.max

def scale(A):
    return (A-np.min(A))/(np.max(A) - np.min(A))

def fixDatasetFormat(dataPath, dataset):
    dataTrain = pd.read_csv(dataPath + dataset+"_TRAIN.csv",header=None)
    dataTest = pd.read_csv(dataPath + dataset+"_TEST.csv",header=None)
    
    dataTrain.columns=["V"+str(val+1) for val in dataTrain.columns.tolist()]
    dataTest.columns=["V"+str(val+1) for val in dataTest.columns.tolist()]
    
    dataTrainClass=pd.read_csv("../LocalWorkSpace/UCR/UCRArchive_2018/"+dataset+"/"+dataset+"_TRAIN.tsv",delim_whitespace=True, header=None)[0]
    dataTestClass=pd.read_csv("../LocalWorkSpace/UCR/UCRArchive_2018/"+dataset+"/"+dataset+"_TEST.tsv",delim_whitespace=True, header=None)[0]
    
    dataTrain.insert(0,"tsClass",dataTrainClass)
    dataTest.insert(0,"tsClass",dataTestClass)
    
    dataTrain.insert(0,"tsIndex",[i for i in range(dataTrain.shape[0])])
    dataTest.insert(0,"tsIndex",[i for i in range(dataTest.shape[0])])
    
    dataTrain.to_csv(dataPath + dataset+"_TRAIN.csv",  index=None)
    dataTest.to_csv(dataPath + dataset+"_TEST.csv",  index=None)


def readUCRclassification(dataPath, dataset, ts=False):
    if(ts):
        dataTrain=pd.read_csv(dataPath+dataset+"/"+dataset+"_TRAIN.tsv",delim_whitespace=True, header=None)
        dataTest=pd.read_csv(dataPath+dataset+"/"+dataset+"_TEST.tsv",delim_whitespace=True, header=None)
        dataTrain.insert(0,"tsIndex",[i for i in range(dataTrain.shape[0])])
        dataTest.insert(0,"tsIndex",[i for i in range(dataTest.shape[0])])
        
        dataTrain.rename(columns={0:"tsClass"},inplace=True)
        dataTest.rename(columns={0:"tsClass"},inplace=True)
    else:
        dataTrain = pd.read_csv(dataPath + dataset+"_TRAIN.csv")
        dataTest = pd.read_csv(dataPath + dataset+"_TEST.csv")
    
    out={"dataTrain":dataTrain, "dataTest":dataTest}
    return out

def experimentation(dataPath, datasets, ts=False):
    out=pd.DataFrame()
    for dataset in datasets:
        data=readUCRclassification(dataPath,dataset,ts)
        classes=data["dataTrain"]["tsClass"].unique()
        
        allData={}
        for clase in classes:
            classTrainData=data["dataTrain"].loc[data["dataTrain"]["tsClass"]==clase,:]
            minDataTrain=classTrainData.min()[2:]
            maxDataTrain=classTrainData.max()[2:]
            meanDataTrain=classTrainData.mean()[2:]

            minDataTrain.index=minDataTrain.index.astype("string")
            maxDataTrain.index=maxDataTrain.index.astype("string")
            meanDataTrain.index=meanDataTrain.index.astype("string")

            aux={"clase": clase,
                 "minDataTrain":minDataTrain,
                 "maxDataTrain":maxDataTrain,
                 "meanDataTrain":meanDataTrain}
            allData.update({clase:aux})
            
        
        test=data["dataTrain"].copy()
        memD=np.ndarray((test.shape[0],
                         classes.size),
                         float)
        memP=pd.DataFrame(memD, columns=classes)
        correct=0
        for i in range(test.shape[0]):
            pertVector=np.zeros((classes.size,
                                test.shape[1]-2),
                                dtype=float)
            for j in range(0,test.shape[1]-2):
                for k in range(classes.size):
                    val=test.iloc[i,j+2]
                    aux=allData[classes[k]]
                    ele=j
                    pert=0
                                        
                    if(val<=aux["meanDataTrain"][ele]):
                        if(aux["meanDataTrain"][ele]==aux["minDataTrain"][ele]):
                            if(val==aux["meanDataTrain"][ele]):
                                pert=1
                            else:
                                pert=1-(aux["meanDataTrain"][ele]-val)/0.001
                        else:
                            pert=1-(aux["meanDataTrain"][ele]-val)/(aux["meanDataTrain"][ele]-aux["minDataTrain"][ele])
                    else:
                        if(aux["maxDataTrain"][ele]==aux["meanDataTrain"][ele]):
                            if(val==aux["meanDataTrain"][ele]):
                                pert=1
                            else:
                                pert=1-(val-aux["meanDataTrain"][ele])/0.001
                        else:
                            pert=1-(val-aux["meanDataTrain"][ele])/(aux["maxDataTrain"][ele]-aux["meanDataTrain"][ele])
                    pertVector[k,j]=pert
            
            scores=np.zeros(classes.size, dtype=float)
            for k in range(classes.size):
                for j in range(0,test.shape[1]-2):
                    scores[k]=scores[k]+pertVector[k,j]
            
            scores=scale(scores)
            
            mem=np.zeros(classes.size, dtype=float)
            for k in range(classes.size):
                memP.iloc[i,k]=scores[k]/scores.sum()
                mem[k]=scores[k]/scores.sum()
                
            pc=classes[memP.iloc[i,:].argmax()]
            
            if(pc==test.iloc[i,1]):
                correct=correct+1
        memP.to_csv("../LocalWorkSpace/UCR/mem/"+dataPath.split(sep="/")[-2]+dataset+"memTrain.csv")
        print("Dataset,",dataset,",Acc,",correct/test.shape[0]*100,sep="")
        out=out.append({"dataset":dataset, "acc":correct/test.shape[0]*100},ignore_index=True)
        out.to_csv("../LocalWorkSpace/UCR/results/mem/"+dataPath.split(sep="/")[-2]+"Train.csv")
        correct
    out.to_csv("../LocalWorkSpace/UCR/results/mem/"+dataPath.split(sep="/")[-2]+"Train.csv")


cases=["RDST_180"]
for i in cases:
    for j in datasets:
        print(j)
        fixDatasetFormat(dataPath="../LocalWorkSpace/UCR/"+i+"/", dataset=j)


cases=["RDST_6","RDST_12","RDST_18","RDST_24","RDST_30","ucrCMFTS_clean","RDST_180","ucrROCKET"]
for i in cases:
    print(i)
    experimentation(dataPath="../LocalWorkSpace/UCR/"+i+"/", datasets=datasets, ts=False)
    
    
experimentation("../LocalWorkSpace/UCR/UCRArchive_2018/", datasets, True)


from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier

import pandas as pd

datasets=pd.read_csv("../LocalWorkSpace/UCR/UCR_datasets_sel.csv")["x"].tolist()

def experimentationRF(case, datasets, fcase="feat"):
    out=pd.DataFrame()
    for dataset in datasets:
        match fcase:
            case "feat":
                if(case=="UCRArchive_2018"):
                    dataTrain=pd.read_csv("../LocalWorkSpace/UCR/"+case+"/"+dataset+"/"+dataset+"_TRAIN.tsv",delim_whitespace=True, header=None)
                    dataTest=pd.read_csv("../LocalWorkSpace/UCR/"+case+"/"+dataset+"/"+dataset+"_TEST.tsv",delim_whitespace=True, header=None)
                    
                    dataTrain.rename(columns={0:"tsClass"},inplace=True)
                    dataTest.rename(columns={0:"tsClass"},inplace=True)
                    
                    dataTrain.columns.values[1:] = ["V"+str(x) for x in range(len(dataTrain.columns[1:]))]
                    dataTest.columns.values[1:] = ["V"+str(x) for x in range(len(dataTest.columns[1:]))]
                    
                else:
                    dataTrain = pd.read_csv("../LocalWorkSpace/UCR/"+case+"/"+dataset+"_TRAIN.csv").iloc[:,1:]
                    dataTest = pd.read_csv("../LocalWorkSpace/UCR/"+case+"/"+dataset+"_TEST.csv").iloc[:,1:]
        
            case "mem":
                dataTrain = pd.read_csv("../LocalWorkSpace/UCR/mem/"+case+dataset+"memTrain.csv").iloc[:,1:]
                dataTest = pd.read_csv("../LocalWorkSpace/UCR/mem/"+case+dataset+"memTest.csv").iloc[:,1:]
                dataTrain=dataTrain.reindex(sorted(dataTrain.columns), axis=1)
                dataTest=dataTest.reindex(sorted(dataTest.columns), axis=1)
                
                if(case=="UCRArchive_2018"):
                    dataTrainClass=pd.read_csv("../LocalWorkSpace/UCR/"+case+"/"+dataset+"/"+dataset+"_TRAIN.tsv",delim_whitespace=True, header=None)
                    dataTestClass=pd.read_csv("../LocalWorkSpace/UCR/"+case+"/"+dataset+"/"+dataset+"_TEST.tsv",delim_whitespace=True, header=None)
                    dataTrainClass.rename(columns={0:"tsClass"},inplace=True)
                    dataTestClass.rename(columns={0:"tsClass"},inplace=True)
                else:
                    dataTrainClass = pd.read_csv("../LocalWorkSpace/UCR/"+case+"/"+dataset+"_TRAIN.csv").iloc[:,1:]
                    dataTestClass = pd.read_csv("../LocalWorkSpace/UCR/"+case+"/"+dataset+"_TEST.csv").iloc[:,1:]
                
                dataTrain=pd.concat([dataTrainClass.loc[:,"tsClass"],dataTrain], ignore_index= False,axis=1)
                dataTest=pd.concat([dataTestClass.loc[:,"tsClass"],dataTest], ignore_index= False,axis=1)
                
        
            case "featmem":
                dataTrainm = pd.read_csv("../LocalWorkSpace/UCR/mem/"+case+dataset+"memTrain.csv").iloc[:,1:]
                dataTestm = pd.read_csv("../LocalWorkSpace/UCR/mem/"+case+dataset+"memTest.csv").iloc[:,1:]
                dataTrainm=dataTrainm.reindex(sorted(dataTrainm.columns), axis=1)
                dataTestm=dataTestm.reindex(sorted(dataTestm.columns), axis=1)
                
                if(case=="UCRArchive_2018"):
                    dataTrainf=pd.read_csv("../LocalWorkSpace/UCR/"+case+"/"+dataset+"/"+dataset+"_TRAIN.tsv",delim_whitespace=True, header=None)
                    dataTestf=pd.read_csv("../LocalWorkSpace/UCR/"+case+"/"+dataset+"/"+dataset+"_TEST.tsv",delim_whitespace=True, header=None)
                    dataTrainf.rename(columns={0:"tsClass"},inplace=True)
                    dataTestf.rename(columns={0:"tsClass"},inplace=True)
                    
                    dataTrainf.columns.values[1:] = ["V"+str(x) for x in range(len(dataTrainf.columns[1:]))]
                    dataTestf.columns.values[1:] = ["V"+str(x) for x in range(len(dataTestf.columns[1:]))]
                else:
                    dataTrainf = pd.read_csv("../LocalWorkSpace/UCR/"+case+"/"+dataset+"_TRAIN.csv").iloc[:,1:]
                    dataTestf = pd.read_csv("../LocalWorkSpace/UCR/"+case+"/"+dataset+"_TEST.csv").iloc[:,1:]
                
                
                dataTrain=pd.concat([dataTrainf,dataTrainm], ignore_index= False,axis=1)
                dataTest=pd.concat([dataTestf,dataTestm], ignore_index= False,axis=1)
        
            case _:
                print("ERROR de lectura")
                
        clf = RandomForestClassifier(n_estimators=500, random_state=0)
        # clf=DecisionTreeClassifier()
        clf.fit(dataTrain.iloc[:,1:], dataTrain.iloc[:,0])
        
        pred=clf.predict(dataTest.iloc[:,1:])
        
        g=sum(pred==dataTest.iloc[:,0])
        
        print("Dataset,",dataset,",Acc,",g/dataTest.shape[0]*100,sep="")
        out=out.append({"dataset":dataset, "acc":g/dataTest.shape[0]*100},ignore_index=True)
        out.to_csv("results/completos/AccRF_"+case+"_"+fcase+".csv")


cases=["UCRArchive_2018","RDST_6","RDST_12","RDST_18","RDST_24","RDST_30",
       "ucrCMFTS_clean","RDST_180","ucrROCKET"]

combs=["feat","mem","featmem"]

for case in cases:
    for fcase in combs:
        print(case)
        experimentationRF(case,datasets,fcase)

