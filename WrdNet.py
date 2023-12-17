import nltk
from nltk.corpus import wordnet
syns=wordnet.synsets("program")
print(syns)
print(syns[0].definition())

antonyms=[]
synonyms=[]

for syn in wordnet.synsets("happy"):
    for i in syn.lemmas():
        synonyms.append(i.name())
        if i.antonyms():
            antonyms.append(i.antonyms()[0].name())
print(set(synonyms))
print(set(antonyms))

w1=wordnet.synset("happy.a.01")
w2=wordnet.synset("joyous.a.01")
print(w1.wup_similarity(w2))    