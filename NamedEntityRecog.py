import nltk
nltk.download('maxent_ne_chunker')
nltk.download('words')
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
import matplotlib.pyplot as plt

train_text=state_union.raw("2005-GWBush.txt")
sample=state_union.raw("2006-GWBush.txt")

custom_sent_tokenizer=PunktSentenceTokenizer(sample)

tokenized=custom_sent_tokenizer.tokenize(sample)

for i in tokenized:
    words=nltk.word_tokenize(i)
    tagged=nltk.pos_tag(words)
    named_ent=nltk.ne_chunk(tagged,binary=True)

    named_ent.draw()