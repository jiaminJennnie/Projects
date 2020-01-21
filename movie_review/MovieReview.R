#------------------------------------------Creating triaining and test datasets
setwd("~/Desktop/movie review")
library(text2vec)
library(slam)
library(glmnet)
library(pROC)
library(MASS)
all = read.table("Project2_data.tsv",stringsAsFactors = F,header = T)
all$review = gsub('<.*?>', ' ', all$review)
splits = read.table("Project2_splits.csv", header = T)
s = 1 # Here we use the 1st training/test split. 
train = all[-which(all$new_id%in%splits[,s]),]
test = all[which(all$new_id%in%splits[,s]),]

#------------------------------------------Creating bags of words
prep_fun = tolower
tok_fun = word_tokenizer
it_train = itoken(train$review,
                  preprocessor = prep_fun, 
                  tokenizer = tok_fun)
it_test = itoken(test$review,
                 preprocessor = prep_fun, 
                 tokenizer = tok_fun)
vocab = create_vocabulary(it_train,ngram = c(1L,4L))
pruned_vocab = prune_vocabulary(vocab,
                                term_count_min = 5, 
                                doc_proportion_max = 0.5,
                                doc_proportion_min = 0.01)
bigram_vectorizer = vocab_vectorizer(pruned_vocab)
dtm_train = create_dtm(it_train, bigram_vectorizer)
dtm_test = create_dtm(it_test, bigram_vectorizer)
v.size = dim(dtm_train)[2]
ytrain = train$sentiment

summ = matrix(0, nrow=v.size, ncol=4)
summ[,1] = colapply_simple_triplet_matrix(
  as.simple_triplet_matrix(dtm_train[ytrain==1, ]), mean)
summ[,2] = colapply_simple_triplet_matrix(
  as.simple_triplet_matrix(dtm_train[ytrain==1, ]), var)
summ[,3] = colapply_simple_triplet_matrix(
  as.simple_triplet_matrix(dtm_train[ytrain==0, ]), mean)
summ[,4] = colapply_simple_triplet_matrix(
  as.simple_triplet_matrix(dtm_train[ytrain==0, ]), var)

n1=sum(ytrain); 
n=length(ytrain)
n0= n - n1
myp = (summ[,1] - summ[,3])/
  sqrt(summ[,2]/n1 + summ[,4]/n0)
words = colnames(dtm_train)
id = order(abs(myp), decreasing=TRUE)[1:3000]
write(words[id], file="myvocab.txt")

myvocab1 = scan(file = "myvocab1.txt", what = character())
myvocab2 = scan(file = "myvocab2.txt", what = character())
myvocab_final = myvocab2[which(myvocab2%in%myvocab1)]
write(myvocab_final, file="myvocab.txt")
myvocab3=myvocab1[-which(myvocab1%in%myvocab_final)]
write(myvocab_final, file="myvocab4.txt")
#-------------
myvocab = scan(file = "myvocab.txt", what = character())
all = read.table("Project2_data.tsv",stringsAsFactors = F,header = T)
all$review = gsub('<.*?>', ' ', all$review)
splits = read.table("Project2_splits.csv", header = T)

s = 1
train = all[-which(all$new_id%in%splits[,s]),]
test = all[which(all$new_id%in%splits[,s]),]
it_train = itoken(train$review,
                  preprocessor = tolower, 
                  tokenizer = word_tokenizer)
it_test = itoken(test$review,
                 preprocessor = tolower, 
                 tokenizer = word_tokenizer)
stop_words = c("i", "me", "my", "myself", 
               "we", "our", "ours", "ourselves", 
               "you", "your", "yours", 
               "their", "they", "his", "her", 
               "she", "he", "a", "an", "and",
               "is", "was", "are", "were", 
               "him", "himself", "has", "have", 
               "it", "its", "of", "one", "for", 
               "the", "us", "this")
vocab = create_vocabulary(it_train, ngram = c(1L,4L), stopwords = stop_words)
vocab = vocab[vocab$term %in% myvocab, ]
bigram_vectorizer = vocab_vectorizer(vocab)
dtm_train = create_dtm(it_train, bigram_vectorizer)
dtm_test = create_dtm(it_test, bigram_vectorizer)
#-----------------------------------------------------Logistic
t1 = Sys.time()
lasso = cv.glmnet(x = dtm_train, y = train$sentiment, 
                              family = 'binomial', 
                              alpha = 1,
                              type.measure = "auc")
tmp <-predict(lasso, s = lasso$lambda.min, newx = dtm_test,type = "response")
tmp1<-rep(0,25000)
tmp1[tmp>0.5]=1
auc(test$sentiment, tmp)
#----------------------------------------------------Naive Baysian
train$sentiment<-as.factor(train$sentiment)
train_x<-as.matrix(dtm_train)
mylda = lda(dtm_train,train$sentiment);
Ytest.pred = predict(dig.lda, Xtest)$class
table(Ytest, Ytest.pred)
Bayes<-NaiveBayes(x=train_x, grouping = train$sentiment, data = train)
tmp <-predict(Bayes,newx = dtm_test)
auc(test$sentiment, tmp)
#---------------------------------------