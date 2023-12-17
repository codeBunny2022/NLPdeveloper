from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

example="This is an example showing stopwords filtration"
stop_words = set(stopwords.words("english"))

words=word_tokenize(example)
print("The filterd output is:")
for w in words:
    if w not in stop_words:
        print(w)