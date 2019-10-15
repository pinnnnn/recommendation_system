library(Matrix)
library(data.table)
library(dplyr)
library(recommenderlab)
library(arules)


topic = '義大'
path = paste0('./data/',topic,'/')

basic_vars = c('CustomerID','Product','Amount','TransDate')
mydat = fread('./data/義大/all_transaction_data.csv', select = basic_vars)
mydat$CustomerID = as.character(mydat$CustomerID)
mydat$TransDate = as.Date(mydat$TransDate)
mydat %>% setorder(TransDate)

all_items = unique(mydat$Product)


# Evaluation
wide_dat = dcast(data = mydat,
                 formula = CustomerID ~ Product, 
                 value.var = 'Amount', 
                 fun.aggregate = length,
                 fill = 0)

user_id = wide_dat$CustomerID
wide_dat = wide_dat %>%
  select(-CustomerID) %>% 
  as.matrix
dimnames(wide_dat) = list(user=user_id, item=colnames(wide_dat))

wide_dat = wide_dat %>% 
  #dropNA %>% #sparseNA ?
  as(.,'binaryRatingMatrix') #%>% 
  #normalize(method = 'Z-score', row=F)



# evaluate scheme (how to evaluate, CV or Split?)
es_cross = evaluationScheme(data=wide_dat, method='cross-validation', train=0.9, k=5, given=1)

rec_method = c('AR',
               'IBCF',
               'POPULAR',
               'RANDOM'
               )
algorithm = list('Association Rule' = list(name='AR',
                                           param=list(supp=.01,conf=.01)),
                 'Item-based CF' = list(name='IBCF',param=NULL),
                 'Popular Item' = list(name='POPULAR',param=NULL),
                 'Random Item' = list(name='RANDOM',param=NULL))
n_recommended = c(1,2,3,4,5,7,10)



# evaluate
results = recommenderlab::evaluate(es_cross, algorithm, type='topNList', n=n_recommended)
save(results, file=paste0(path,'evaluate','.rdata'))



# tidy results
CrossValidationResult = function(result){
  temp = result %>% 
    getConfusionMatrix()
  
  as.data.frame(Reduce(f = "+", x = temp) / length(temp),stringsAsFactors=F) %>% 
    mutate(n=n_recommended) %>% 
    select(n, precision, recall, TPR, FPR)
}


results_tbl = results %>% 
  purrr::map(CrossValidationResult) %>% 
  enframe() %>% 
  unnest()

write.csv(results_tbl, file=paste0(path, 'evaluatioin_results_tbl.csv'), row.names=F)

# plot
# ROC curve
roc_curve = results_tbl %>%
  ggplot(aes(FPR, TPR, 
             colour = fct_reorder2(as.factor(name),  # set color order by FPR, TPR
                                   FPR, TPR))) +
  geom_line() +
  geom_label(aes(label = n))  +
  labs(title = "ROC curves", colour = "Model") +
  theme_grey(base_size = 14) # font_size

ggsave(roc_curve, filename = paste0(path,'roc_curve.png'))



# precision and recall curve
pr_curve = results_tbl %>% 
  ggplot(aes(recall, precision, 
             colour = fct_reorder2(.f=as.factor(name),
                                   .x=FPR, .y=TPR))) +
  geom_line() + 
  geom_label(aes(label=n)) +
  labs(title = 'Precision-Recall Curves', colour = 'Model') +
  theme_grey(base_size = 14)

ggsave(pr_curve, filename = paste0(path, 'precision_recall_curve.png'))
  







# Operating

method = 'AR'

CompletionItem = function(wide_dat, all_items, method){
  if(any(missing_item <- (!all_items %in% colnames(wide_dat)[which(!colnames(wide_dat)%in%c('CusomerID','TransID'))])) ){
    for(m in 1:length(which(missing_item))){
      wide_dat[,(all_items[which(missing_item)[m]]) := 0]
    }
  }
  
  setcolorder(x = wide_dat, neworder = c('CustomerID',all_items))

}

cut_date = as.Date('2017/06/30')
mydat[,TransID := paste0(CustomerID, ', ',TransDate)]
mydat_train = mydat[which(TransDate<=cut_date),]
mydat_test = mydat[which(TransDate>cut_date),]


wide_dat_train = dcast(data = mydat_train, 
                       formula = CustomerID ~ Product, 
                       value.var = 'Amount', 
                       fun.aggregate = length,
                       fill = 0) 


wide_dat_test = dcast(data = mydat_test, 
                      formula = CustomerID ~ Product, 
                      value.var = 'Amount', 
                      fun.aggregate = length,
                      fill = 0) 


CompletionItem(wide_dat_train, all_items=all_items, method=method)
CompletionItem(wide_dat_test, all_items=all_items, method=method)

# use binaryRatingMatrix (rather than realRatingMatrix)

train_dat = wide_dat_train[,-1] %>% 
  as.matrix %>% 
  as(.,'binaryRatingMatrix')
#  normalize(., row=F)


test_dat = wide_dat_test[,-1] %>% 
  as.matrix %>% 
  as(.,'binaryRatingMatrix')
#  normalize(., row=F)


dimnames(train_dat) = list(user=wide_dat_train$CustomerID,
                           item=all_items)

dimnames(test_dat) = list(user=wide_dat_test$CustomerID,
                          item=all_items)





# Chosen Model
### AR
n_recommended = 3
recommend_model = Recommender(data = train_dat, method=method, parameter = list(supp=0.1, conf=0.1))
predict_test = predict(object = recommend_model,  newdata=test_dat, n=n_recommended, type = 'topNList')
acc_summary = calcPredictionAccuracy(x=predict_test, data=test_dat, given=0, byUser=F)
acc_result = calcPredictionAccuracy(x=predict_test, data=test_dat, given=0, byUser=T)



set.seed(101)
id_sample = sample(nrow(acc_result),5)
wide_dat_test[id_sample,]
as(predict_test,'list')[id_sample]
acc_result[id_sample,]



### AR using arules
rule_dat = wide_dat_train %>% 
  select(-CustomerID)
tttt = rule_dat !=0

rule = apriori(tttt, 
               parameter=list(minlen=2,maxlen=4,supp=0.1, conf=0.1))
rr = inspect(rule)

rr[order(rr$lift, decreasing = T),]




