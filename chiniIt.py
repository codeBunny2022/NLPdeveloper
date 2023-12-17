import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
import matplotlib.pyplot as plt

sample_text=state_union.raw("2005-GWBush.txt")
sample=state_union.raw("2006-GWBush.txt")


custom_sent_tokenizer=PunktSentenceTokenizer(sample)
tokenized=custom_sent_tokenizer.tokenize(sample_text)

for i in tokenized:
    words=nltk.word_tokenize(i)
    tagged=nltk.pos_tag(words)

    chunkGram=r"""Chunk:{<.*>+}
                }<VB.?|IN|DT>{"""
    
    chunkParser=nltk.RegexpParser(chunkGram)
    chunked=chunkParser.parse(tagged)

    chunked.draw()