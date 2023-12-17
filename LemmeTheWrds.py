from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ps=PorterStemmer()
example=["python","pythonner","watch","watcher","watchful"]
for w in example:
    print(ps.stem(w))