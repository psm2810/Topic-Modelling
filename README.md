# Topic-Modelling

#R code for topic analysis:
#Installing packages
install.packages("text2vec")
install.packages("tm")
install.packages("topicmodels")
install.packages("tidytext")
install.packages("dplyr")
install.packages("ggplot2")
install.packages("reshape2")
install.packages("ldatuning")
install.packages("stringr")
install.packages("caTools")
install.packages("data.table")
install.packages("slam")
slam_url <- "http://cran.r-project.org/src/contrib/Archive/slam/slam_0.1-30.tar.gz"
install.packages(slam_url, repos = NULL, type = "source")
install.packages("http://cran.r-project.org/bin/windows/contrib/3.0/tm_0.5-10.zip",repos=NULL)

library(tm)
library(text2vec)
library(topicmodels)
library(tidytext)
library(dplyr)
library(ggplot2)
library(reshape2)
library(ldatuning)
library(stringr)
library(caTools)
library(data.table)
library(slam)

setwd("H:/R Learning/FD Topic Modelling")
getwd()

#Uploading data in csv
x <- read.csv("FD LIVE CHAT R IMPORT.csv", stringsAsFactors = FALSE, header = TRUE)


#Create Corpus from the file that has been loaded as csv for test and train 
FD_Corpus <- Corpus(VectorSource(x$customer))

#Creating the custom stop words list from frequency words in doc x

stop_words1 <- x %>%
  unnest_tokens(word, customer)
stop_words2 <-stop_words1 %>%
  count(word, sort = TRUE)
write.csv(stop_words2,'H:/R Learning/FD Topic Modelling/FD_stop_term_freq.csv')
#From this term frequency create a custom stopword list and upload that csv file
stoping <-read.csv("custom_stop_OL_CSV.csv")

############Cleaning

#Cleaning the comments
Clean_Corp<- tm_map(FD_Corpus, content_transformer(tolower))
Clean_Corp<- tm_map(Clean_Corp, removePunctuation)
Clean_Corp<- tm_map(Clean_Corp, removeWords,stoping$words)
Clean_Corp<- tm_map(Clean_Corp, removeWords, stopwords('english'))
Clean_Corp<- tm_map(Clean_Corp, removeWords, stopwords('SMART'))
Clean_Corp<- tm_map(Clean_Corp, removeWords, c("please","your","today","number","today","can","now","will","just", "hsbc", "bank","fd","agent", "customer", "bye","company","make", 
                                               "if you have time to answer a few questions at the end of our chat it will help us improve this service click the x in the right hand corner of this chat window the survey will then be available",
                                               "to end the chat just click the x at the top right of this chat window if you have time to answer a few quick questions about using live chat today that would be really helpful",
                                               "the survey will pop up once you select the x", "you can select the close button to close this window", "click", "x", "close","right"))
Clean_Corp<- tm_map(Clean_Corp, removeNumbers)

Clean_Corp

Clean_Corp$content

#Creating Document- Term Matrix for train & test
#dtm_Both_Comment_Need <- as.matrix(DocumentTermMatrix(Both_Comment_Need_Corpus_clean))
dtm_Comment <- DocumentTermMatrix(Clean_Corp)
rowTotals <- apply(dtm_Comment,1,sum)
dtm_Comment.new <- dtm_Comment[rowTotals>0 ,]


#Finding Optimal Number of Topics from data
result_topics <- FindTopicsNumber(
  dtm_Comment.new,
  topics = seq(from = 2, to = 25, by = 1),
  metrics=c("Griffiths2004", "CaoJuan2009", "Arun2010", "Deveaud2014"),
  method = "Gibbs",
  control = list(seed = 12345),
  mc.cores = 2L,
  verbose = TRUE
)
FindTopicsNumber_plot(result_topics)

#calculating LDA
burnin <- 4000
iter <- 2000
thin <- 500
seed <- list(2003,5,63,100001,765)
nstart <- 5
best <- TRUE

#Number of topics
k <- 18

#Run LDA using Gibbs Sampling 
ldaOut <- LDA(dtm_Comment.new,k = k,method = "Gibbs" , control= list(nstart = nstart,seed = seed, best = best,burnin = burnin,iter = iter, thin = thin))

#Finding the Topic to Term Probability matrix
get_topics <- tidy(ldaOut, matrix = "beta")
get_topics10 <- as.data.frame(get_topics)

write.csv(get_topics10,'H:/R Learning/FD Topic Modelling/beta_Negative_Comment_conf10.csv')

#Graphical Representation of top ten words for each topic
get_top_terms <- get_topics %>%
  group_by(topic) %>%
  top_n(10, beta) %>%
  ungroup() %>%
  arrange(topic, -beta)
get_top_terms %>%
  mutate(term = reorder(term, beta)) %>%
  ggplot(aes(term, beta, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  coord_flip()

#Document-topic probabilities
get_documents <- tidy(ldaOut, matrix = "gamma")
get_documents10 <- as.data.frame(get_documents)
write.csv(get_documents10,'H:/R Learning/FD Topic Modelling/gamma_Negative_Comment_conf10.csv')

#Classifying New documents, evaluating over test data
test.topics <- posterior(ldaOut,dtm_Comment.new)
class(test.topics)
#viewing the probabilities

beta <- test.topics[[2]]
write.csv(beta,'H:/R Learning/FD Topic Modelling/Negative_Comment_graph.csv')

#Classifying the document by taking the max probability
(test.topics1 <- apply(test.topics$topics, 1, which.max))
test_topics2<-as.data.frame(test.topics1)

write.csv(test_topics2,'H:/R Learning/FD Topic Modelling/test_topic2.csv')

x_clean <- read.csv("cleaned_9999.csv", stringsAsFactors = FALSE, header = TRUE)

test_topics3 <- cbind(x_clean,test_topics2)

write.csv(test_topics3,'H:/R Learning/FD Topic Modelling/chat_topic_merge_back.csv')


#######################################################################
#R Sentiments Analysis:

#Emotion identification

library(dplyr)
require(tidyverse)
require(tidytext)
require(RColorBrewer)
require(gplots)
library(tidyr)
library(ggplot2)
theme_set(theme_bw(12))

dictionary1 <- get_sentiments("bing")

write.csv(dictionary1,'H:/R Learning/FD Topic Modelling/dictionary_bing.csv')

dictionary2 <- get_sentiments("afinn")

write.csv(dictionary2,'H:/R Learning/FD Topic Modelling/dictionary_afinn.csv')

#Uploading data in csv

getwd()

myVars <-read.csv("chat_topic_merge_back.csv")

#Emotion Words frequency and proportions

emotion_words_count <- myVars %>% 
  unnest_tokens(word,customer) %>%                           
  anti_join(stop_words, by = "word") %>%                  
  filter(!grepl('[0-9]', word)) %>%
  left_join(get_sentiments("nrc"), by = "word") %>%
  filter(!is.na(sentiment)) %>%
  group_by(test.topics1,sentiment) %>%
  summarize(emotions= n()) %>%
  ungroup()


#Depicting distribution of emotion words usage
### pull emotion words and aggregate by topic and emotion terms

emotions <- myVars %>% 
  unnest_tokens(word, customer) %>%                           
  anti_join(stop_words, by = "word") %>%                  
  filter(!grepl('[0-9]', word)) %>%
  left_join(get_sentiments("nrc"), by = "word") %>%
  filter(!is.na(sentiment)) %>%
  group_by(test.topics1, sentiment) %>%
  summarize( freq = n()) %>%
  mutate(percent=round(freq/sum(freq)*100)) %>%
  select(-freq) %>%
  ungroup()

### need to convert the data structure to a wide format

emo_box = emotions %>% spread(sentiment, percent, fill=0) %>% ungroup()

write.csv(emo_box,'H:/R Learning/FD Topic Modelling/document_emotion.csv')
