library(parallel)
workdirectory="../LocalWorkSpace/UCR/"
inputdir="Univariate2018_arff/Univariate_arff/"
datasets=(read.csv(paste0(workdirectory,"UCR_datasets_sel.csv"))[,1])
outputdir=paste0(workdirectory,"ucrCMFTS/")

lapply(datasets,function(dataset){
  set.seed(26)
  print(dataset)
  train_file=paste0(workdirectory, inputdir,"/",dataset,"/",dataset,"_TRAIN.txt")
  test_file=paste0(workdirectory, inputdir,"/",dataset,"/",dataset,"_TEST.txt")
  train=read.table(train_file)
  test=read.table(test_file)
  
  aux=mclapply(1:dim(train)[1], function(r){
    data=as.numeric(train[r,-c(1)])
    cmfts::cmfts(t(data.frame(data)), n_cores = 1, na = T,scale = T)
  },mc.cores = 4, mc.preschedule = T)
  
  aTrain=do.call(rbind,aux)
  aTrain=as.data.frame(aTrain)
  
  aux=mclapply(1:dim(test)[1], function(r){
    data=as.numeric(test[r,-c(1)])
    cmfts::cmfts(t(data.frame(data)), n_cores = 1, na = T,scale = T)
    
  },mc.cores = 4, mc.preschedule = T)
  aTest=do.call(rbind,aux)
  aTest=as.data.frame(aTest)
  
  aTrain=cbind.data.frame(1:dim(train)[1],train[,c(1)], aTrain)
  aTest=cbind.data.frame(1:dim(test)[1],test[,c(1)], aTest)
  
  names(aTrain)=c("tsIndex","tsClass",names(aTrain)[-c(1,2)])
  write.csv(aTrain,file = paste0(outputdir,dataset,"_TRAIN.csv"),row.names = F)
  
  names(aTest)=c("tsIndex","tsClass",names(aTest)[-c(1,2)])
  write.csv(aTest,file = paste0(outputdir,dataset,"_TEST.csv"),row.names = F)
})

postProcessing<-function(workdirectory, inputdir, outputdir){
  datasets=as.list(read.csv(paste0(workdirectory,"UCR_datasets_sel.csv")))$x
  results=mclapply(datasets,function(dataset){
    print(dataset)
    train=read.csv(paste0(workdirectory,inputdir,dataset,"_TRAIN.csv"))
    test=read.csv(paste0(workdirectory,inputdir,dataset,"_TEST.csv"))
    
    all=rbind(train,test)
    all[,2]=as.factor(all[,2])
    train=all[1:dim(train)[1],]
    test=all[(dim(train)[1]+1):dim(all)[1],]
    
    sum(apply(train,c(1,2),is.na))
    sum(apply(train,c(1,2),is.nan))
    sum(apply(train,c(1,2),is.infinite))
    
    trainInit=train[,c(1,2)]
    testInit=test[,c(1,2)]
    
    train=train[,-c(1,2)]
    test=test[,-c(1,2)]
    
    if(dim(train)[2]<=2){
      train=as.numeric(train)
      test=as.numeric(test)
    }else{
      train=as.data.frame(apply(train,c(1,2),as.numeric))
      test=as.data.frame(apply(test,c(1,2),as.numeric))
    }
    
    train[apply(train,c(1,2),is.nan) | apply(train,c(1,2),is.na)] <- NA
    test[apply(test,c(1,2),is.nan) | apply(test,c(1,2),is.na)] <- NA
    

    finite=apply(train,c(1,2),function(x){
      is.finite(x)
    })
    
    filteredColums=!(colSums(!finite)>dim(train)[1]*0.2)
    
    train=train[,filteredColums]
    test=test[,filteredColums]
    
    
    maxMin=apply(train,2,function(x){
      values=x[is.finite(x)]
      c(max(values),min(values))
    })
    
    for(i in 1:dim(train)[2]){
      train[,i][is.infinite(train[,i]) & train[,i]>0]=maxMin[1,i]
      train[,i][is.infinite(train[,i]) & train[,i]<0]=maxMin[2,i]
      
      test[,i][is.infinite(test[,i]) & test[,i]>0]=maxMin[1,i]
      test[,i][is.infinite(test[,i]) & test[,i]<0]=maxMin[2,i]
    }
    
    train=as.data.frame(apply(train,2,function(x){
      x[is.na(x)]=mean(x[is.finite(x)])
      x
    }))
    
    test=as.data.frame(apply(test,2,function(x){
      x[is.na(x)]=mean(x[is.finite(x)])
      x
    }))
    
    sum(apply(train,c(1,2),is.na))
    sum(apply(train,c(1,2),is.nan))
    sum(apply(train,c(1,2),is.infinite))
    
    colQuitar=which(apply(train,2,function(x){
      length(unique(x))<=1
    }))
    
    if(length(colQuitar)>=1){
      train=train[,-colQuitar]
      test=test[,-colQuitar]
    }
    
    train=cbind.data.frame(trainInit,train)
    test=cbind.data.frame(testInit,test)
    
    outputdir2=paste0(workdirectory,outputdir)
    
    if(!dir.exists(outputdir2))
      dir.create(outputdir2)
    write.csv(train, file = paste0(outputdir2,basename(dataset),"_TRAIN.csv"), row.names = F)
    write.csv(test, file = paste0(outputdir2,basename(dataset),"_TEST.csv"), row.names = F)
    
  },mc.cores=4)
}

postProcessing("../LocalWorkSpace/UCR/","ucrCMFTS/","ucrCMFTS_clean/")