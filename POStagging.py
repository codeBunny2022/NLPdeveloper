import nltk
# nltk.download('state_union')
# nltk.download('averaged_perceptron_tagger')
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
train_text=state_union.raw("2005-GWBush.txt")
sample=state_union.raw("2006-GWBush.txt")
custom_sent_tokenizer=PunktSentenceTokenizer(sample)

tokenized=custom_sent_tokenizer.tokenize(sample)
def process_it():
    try:
        for i in tokenized:
            words=nltk.word_tokenize(i)
            tagged=nltk.pos_tag(words)
            print(tagged)
    except Exception as e:
        print(str(e))

process_it()