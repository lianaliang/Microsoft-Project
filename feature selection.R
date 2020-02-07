library(data.table)#data.table is an R package that provides an enhanced version of data.frame
#especially with much faster data reading speed
library(caret) #caret contains many powerful tools for data pre-processing and feature selection
library(dplyr) #dplyr provides flexible grammar and powerful tools of data manipulation
train = fread('train100k.csv',integer64 = 'numeric')
test = fread('test100k.csv',integer64 = 'numeric')
str(train)

prop.table(table(train$HasDetections)) #make sure it's balanced

train1 = train[,- "MachineIdentifier"]
test1 = test[,- "MachineIdentifier"]

n = colSums(is.na(train1) > 0) # NA entries
m = colSums(train1 == '', na.rm = T) # Blank cells
(m+n)/100000 # We want to know the proportion of the missing value

miss_too_many=which((m+n)/100000>0.5) # We set the cutoff by 50%
train2 = subset(train1,select=-miss_too_many)
#delete those columns with too much missing data
test2=subset(test1,select=-miss_too_many)
#We can only apply with the criteria we get from the training data

y <- train$HasDetections # We will store our response variable in an individual column.
fea <- names(train2)
cats <- c()
cat_ind <- c()
for (f in colnames(train2))
  if (is.character(train2[[f]])|grepl(".Identifier", f)>0 )#Check is.character or include "Identifier"
  {cats <- c(cats, f) # Store the feature name
  cat_ind <- c(cat_ind, which(colnames(train2)==f))}#Store the column number of those features

uniquevalues <- c()
for (f in cat_ind)
{uniquevalues<- c(uniquevalues,length(unique(train2[[f]])))}

less_level_index=which(uniquevalues<5)
many_level_index=which(uniquevalues>=5)

#Remember the objective is to predict the label in the test data.
#Thus you cannot use them when you build the model
#We can only apply with the criteria we get from the training data.
cats[many_level_index]

tr=train2
ts=test2
for (i in many_level_index)
{
  col <- data.frame("predictor" = train2[[cat_ind[i]]], "target" = y) #create a new dataframe
  col_ts<- data.frame("predictor" = test2[[cat_ind[i]]]) # We won't use the test target
  lookup = col %>% #This part is using the dplyr package
    group_by(predictor) %>% #use groupby() to combine each level
    summarise(mean_target = mean(target)) #computing the target mean of each level
  col = left_join(col, lookup) #join the table with our computed mean
  col_ts = left_join(col_ts, lookup)
  tr[[cat_ind[i]]] <- col$mean_target #replace the value in the original table with computed mean
  ts[[cat_ind[i]]] <- col_ts$mean_target
}
str(tr)

#Remember the objective is to predict the label in the test data.
#Thus you cannot use them when you build the model
cats[less_level_index]

tr2=tr
ts2=ts
for (i in less_level_index)
{
  col <- data.frame("predictor" = tr2[[cats[i]]], "target" = y) # create a new data frame
  col_ts <- data.frame("predictor" = ts2[[cats[i]]], "target" = 0)
  # Here for test data we didn't use y, just use a zero column to stand on the place
  # We actually did not use "target", and we will delete this column afterwards
  col$predictor = as.factor(col$predictor)
  colnames(col) <- c(cats[i], "target") #get the original column name
  colnames(col_ts) <- c(cats[i], "target")
  dummy.vars = dummyVars(~ ., data = col, fullRank = TRUE)
  col.dummy = predict(dummy.vars, newdata = col) #create dummy (0-1) columns
  col_ts.dummy = predict(dummy.vars, newdata = col_ts) #apply to test data
  j=which(colnames(tr2)==cats[i]) #to locate which column would be replaced by dummy columns
  tr2=cbind(tr2[,1:(j-1)],col.dummy,tr2[,(j+1):dim(tr2)[2]]) #replace original column by dummies
  
  #using the approach of cbind()
  ts2=cbind(ts2[,1:(j-1)],col_ts.dummy,ts2[,(j+1):dim(ts2)[2]]) #apply to test data
  tr2=tr2[,-"target"]
  ts2=ts2[,-"target"]
}
train3=tr2[,-"ProductName"] # "ProductName" is the first column of the table,thus with j=1
test3=ts2[,-"ProductName"] # then the cbind() line above will not delete that column
# so we need to remove this column seperately
# One can surely code in a more elegant way
str(train3)



#Fill missing values
pre.impute = preProcess(train3, method = "medianImpute")
train5 = predict(pre.impute, train3)
test5 = predict(pre.impute, test3) #Apply the result of train to test
str(train5)

nn = colSums(is.na(train5) > 0)
mm = colSums(train5 == '', na.rm = T)
mm+nn

train100k_processed=train5
test100k_processed=test5
# write.csv(train100k_processed, 'train100k_processed.csv', row.names = FALSE)
# write.csv(test100k_processed, 'test100k_processed.csv', row.names = FALSE)

str(train5)


library(leaps)#ALL-SUBSETS REGRESSION-NO NEED FOR LASSO
sum(sapply(train100k_processed, is.character))
library(VIF)
library(caret)
############################################forward stepwise
regfit.full<- regsubsets(HasDetections~.,train100k_processed ,nvmax=83,method='forward')#,matrix.logical=TRUE wrongsummary(fsr_model)
reg.summary=summary(regfit.full)
reg.summary
names(reg.summary)

which.min(reg.summary$cp) #52
which.min(reg.summary$bic) #34
which.max(reg.summary$adjr2) #59

par(mfrow=c(2,2))
plot(reg.summary$rss,xlab = 'Number of Variables',ylab = 'RSS',type='l')
plot(reg.summary$adjr2,xlab = 'Number of Variables',ylab = 'Adjusted RSq',type='l')
which.max(reg.summary$adjr2)
plot(reg.summary$cp,xlab='number of variables',ylab='Cp',type='l')
which.min(reg.summary$cp)
plot(reg.summary$bic,xlab='number of variables',ylab='Bic',type='l')
which.min(reg.summary$bic)

bestvariables <-names(which(summary(regfit.full)$which[34,]))
bestvariables = list(bestvariables)

bestvariables[[1]] = bestvariables[[1]][-1]
bestvariables[[1]][35]='HasDetections'
length(bestvariables[[1]])

var34train = subset(train100k_processed, select = bestvariables[[1]])
var34test = subset(test100k_processed, select = bestvariables[[1]])
write.csv(var34train,  'var34train.csv', row.names = FALSE)
write.csv(var34test,  'var34test.csv', row.names = FALSE)




###########################################all subsets reg
# regfit.full<- regsubsets(HasDetections~.,train100k_processed ,nvmax=83, really.big=T)#,matrix.logical=TRUE wrongsummary(fsr_model)
# bestvariables <-names(which(summary(regfit.full)$which[20,]))
# bestvariables
# reg.summary=summary(regfit.full)
# names(reg.summary)
#1.??????????????????????????????RSS?????????R??????Cp??????BIC???????????????????????????


#3RF-SELECTION
#library(caret)
#rfCtrl1 = trainControl(method='repeatedcv', number = 5, repeats = 5)
#set.seed(555)
##rf = train(HasDetections ~ .,
#data =train100k_processed,
#method = 'rf',
#trControl = rfCtrl1,
#importance = TRUE)
#print(varImp(rf))
##################LASSO
# library(glmnet)
# `%ni%`<-Negate('%in%')
# 
# x<-model.matrix(HasDetections~.,data=train5)
# x=x[,-1]
# 
# glmnet1<-cv.glmnet(x=x,y=train5$HasDetections,type.measure='mse',nfolds=5,alpha=.5)
# 
# c<-coef(glmnet1,s='lambda.min',exact=TRUE)
# inds<-which(c!=0)
# variables<-row.names(c)[inds]
# variables<-variables[variables %ni% '(Intercept)']
# variables

# 
# 
# source("http://www.sthda.com/upload/rquery_cormat.r")
# rquery.cormat(train5, type = 'full')
# 
# #
# # library("Hmisc")
# # mydata.rcorr = rcorr(as.matrix(train5))
# # mydata.rcorr[[r]]
# # write.csv(mydata.rcorr[['r']], 'mydata.rcorr.csv', row.names = FALSE)
# #
# # model1 = lm(HasDetections ~ . , data=train5)
# # library(car)
# # library(VIF)
# # vif(lm(HasDetections ~ . , data=train5))
# 
# 
# library(leaps)#ALL-SUBSETS REGRESSION-NO NEED FOR LASSO
# sum(sapply(train100k_processed, is.character))
# fsr_model<- regsubsets(HasDetections~.,train100k_processed ,really.big = T, method='forward')#,matrix.logical=TRUE wrongsummary(fsr_model)
# summary(fsr_model)
# bestvariables <-names(which(summary(fsr_model)$which[20,]))
# 
# 
# 
# 
# #
# # regfit.full<- regsubsets(HasDetections~.,train100k_processed ,nvmax=20,method='forward')#,matrix.logical=TRUE wrongsummary(fsr_model)
# # bestvariables <-names(which(summary(regfit.full)$which[20,]))
# # bestvariables
# # reg.summary=summary(regfit.full)
# # names(reg.summary)
# # #1.??????????????????????????????RSS?????????R??????Cp??????BIC???????????????????????????
# # par(mfrow=c(2,2))
# # plot(reg.summary$rss,xlab = 'Number of Variables',ylab = 'RSS',type='l')
# # plot(reg.summary$adjr2,xlab = 'Number of Variables',ylab = 'Adjusted RSq',type='l')
# # which.max(reg.summary$adjr2)
# # points(11,reg.summary$adjr2[11],col='red',cex=2,pch=20)
# # plot(reg.summary$cp,xlab='number of variables',ylab='Cp',type='l')
# # which.min(reg.summary$cp)
# # points(10,reg.summary$cp[10],col='red',cex=2,pch=20)
# # plot(reg.summary$bic,xlab='number of variables',ylab='Bic',type='l')
# # which.min(reg.summary$bic)
# # points(6,reg.summary$bic[6],col='red',cex=2,pch=20)
# #
# # set.seed(1)
# # train=sample(c(TRUE,FALSE),nrow(train5),rep=TRUE)
# # test=(!train)
# # regfit.best=regsubsets(HasDetections~.,data=train5[train,],nvmax=20,really.big=T) #????????????????????????
# # summary(regfit.best)
# # test.mat=model.matrix(HasDetections~.,data=train5[test,])
# # val.errors=rep(NA,20)
# # for(i in 1:20){
# #   coefi=coef(regfit.best,id=i) # i?????????????????????
# #   pred=test.mat[,names(coefi)]%*%coefi
# #   val.errors[i]=mean((train5$HasDetections[test]-pred)^2)
# # }
# # val.errors
# # which.min(val.errors)
