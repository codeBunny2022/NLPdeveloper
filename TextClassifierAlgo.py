import nltk
import random
import pickle

import numpy as np
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from nltk.classify import ClassifierI
from statistics import mode


class VoteClassifier(ClassifierI):
    def __init__(self,*classifiers):
        self._classifiers=classifiers
    
    def classify(self, features):
        votes=[]
        for c in self._classifiers:
            v=c.classify(features)
            votes.append(v)
        return mode(votes)
    
    def confidence(self,features):
        votes=[]
        for c in self._classifiers:
            v=c.classify(features)
            votes.append(v)

        choice_votes=votes.count(mode(votes))
        conf=choice_votes/len(votes)
        return conf




from sklearn.linear_model import SGDClassifier,LogisticRegression
from sklearn.svm import SVC,LinearSVC,NuSVC

from nltk.corpus import movie_reviews
documents =[(list(movie_reviews.words(fileid)),category)
            for category in movie_reviews.categories()
            for fileid in movie_reviews.fileids(category)]
'''
documents=[]
for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        documents.append(list(movie_reviews.words(fileid)),category)
'''

random.shuffle(documents)
# print(documents())
all_words=[]
for w in movie_reviews.words():
    all_words.append(w.lower())
all_words=nltk.FreqDist(all_words)

# print(all_words.most_common(10))
# print(all_words["stupid"])
word_features=list(all_words.keys())[:3000]

def find_features(document):
    words=set(document)
    features={}
    for w in word_features:
        features[w]=(w in words)
    return features


# print(find_features(movie_reviews.words('neg/cv000_29416.txt')))
featuresets=[(find_features(rev),category) for (rev,category) in documents]

train_set=featuresets[:1900]
test_set=featuresets[1900:]
'''
We can also give x_train and y_train seperately by doing this 

train_feats, train_labels = zip(*train_set)
MNB_classifier.train(list(zip(train_feats, train_labels)))

'''

# classifier=nltk.NaiveBayesClassifier.train(train_set)

classifier_f=open("/home/chirag/Desktop/NLTK/naive_bias","rb")
classifier=pickle.load(classifier_f)
classifier_f.close()



print("NLTK NAIVE BASED CLASSIFIER ACCURACY:",nltk.classify.accuracy(classifier,train_set))
classifier.show_most_informative_features(15) 

# save_classifier=open("naive_bias","wb")
# pickle.dump(classifier,save_classifier)
# save_classifier.close()

MNB_classifier=SklearnClassifier(MultinomialNB())
MNB_classifier.train(train_set)
print("MNB_classifier ACCURACY:",(nltk.classify.accuracy(MNB_classifier,test_set))*100)

Ber_classifier=SklearnClassifier(BernoulliNB())
Ber_classifier.train(train_set)
print("Ber_classifier ACCURACY:",(nltk.classify.accuracy(Ber_classifier,test_set))*100)


# Gauss_classifier=SklearnClassifier(GaussianNB())
# Gauss_classifier.train(train_set)
# print("Gauss_classifier ACCURACY:",(nltk.classify.accuracy(Gauss_classifier,test_set))*100)


# classifiersList=[SVC,LinearSVC,NuSVC,SGDClassifier,LogisticRegression]

SVC_classifier=SklearnClassifier(SVC())
SVC_classifier.train(train_set)
print("SVC_classifier ACCURACY:",(nltk.classify.accuracy(SVC_classifier,test_set))*100)

LinearSvc_classifier=SklearnClassifier(LinearSVC())
LinearSvc_classifier.train(train_set)
print("LinearSvc_classifier ACCURACY:",(nltk.classify.accuracy(LinearSvc_classifier,test_set))*100)

# Nu_classifier=SklearnClassifier(NuSVC())
# Nu_classifier.train(train_set)
# print("Nu_classifier ACCURACY:",(nltk.classify.accuracy(Nu_classifier,test_set))*100)

# SGD_Classifier=SklearnClassifier(SGDClassifier())
# SGD_Classifier.train(train_set)
# print("SGD_Classifier ACCURACY:",(nltk.classify.accuracy(SGD_Classifier,test_set))*100)

Logistic_classifier=SklearnClassifier(LogisticRegression())
Logistic_classifier.train(train_set)
print("Logistic_classifier ACCURACY:",(nltk.classify.accuracy(Logistic_classifier,test_set))*100)

voted_classifier=VoteClassifier(classifier,
                                MNB_classifier,
                                Ber_classifier,
                                Logistic_classifier,
                                SVC_classifier,
                                LinearSvc_classifier)
print("Voted_classifier ACCURACY:",(nltk.classify.accuracy(voted_classifier,test_set))*100)

print("Classificaton: ",voted_classifier.classify(test_set[0][0]),"Confidence %:",voted_classifier.confidence(test_set[0][0]))
print("Classificaton: ",voted_classifier.classify(test_set[1][0]),"Confidence %:",voted_classifier.confidence(test_set[1][0]))
print("Classificaton: ",voted_classifier.classify(test_set[2][0]),"Confidence %:",voted_classifier.confidence(test_set[2][0]))
print("Classificaton: ",voted_classifier.classify(test_set[3][0]),"Confidence %:",voted_classifier.confidence(test_set[3][0]))
print("Classificaton: ",voted_classifier.classify(test_set[4][0]),"Confidence %:",voted_classifier.confidence(test_set[4][0]))
print("Classificaton: ",voted_classifier.classify(test_set[5][0]),"Confidence %:",voted_classifier.confidence(test_set[5][0]))