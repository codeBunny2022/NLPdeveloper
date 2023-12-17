"""
Tokenizers: 
1. word tokenizers -> seperates by words
2. sentence tokenizers -> seperates by sentences

lexicons-> words and meanings (dictionary)
corporals-> it is a body of text
"""
from nltk.tokenize import sent_tokenize, word_tokenize

exampleText = "Hello there, How are you doing today, Weather is great! and python is awesome.The sky is pinkish blue , You should not eat cardbord."
print(sent_tokenize(exampleText))
print(word_tokenize(exampleText))