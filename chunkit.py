import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
import matplotlib.pyplot as plt

train_text=state_union.raw("2005-GWBush.txt")
sample=state_union.raw("2006-GWBush.txt")

custom_sent_tokenizer=PunktSentenceTokenizer(sample)

tokenized=custom_sent_tokenizer.tokenize(sample)
def process_it():
    try:
        for i in tokenized:
            words=nltk.word_tokenize(i)
            tagged=nltk.pos_tag(words)

            chunkGram=r"""Chunk:{<RB.?>*<VB.?>*<NNP><NN>?}"""

            chunkParser=nltk.RegexpParser(chunkGram)
            chunked=chunkParser.parse(tagged)
        chunked.draw()
    except Exception as e:
        print(str(e))

        

process_it()