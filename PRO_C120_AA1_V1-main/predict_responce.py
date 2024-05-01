import nltk
import json
import pickle
import numpy as np
import random

ignore_words=["?","!" ",","." " 's' 'm' "]
import tensorflow

from data._preprossing  import get_stem_words
model= tensorflow.keras.models.load_model('./chatbot.h5')

intens=json.loads(open('./intents.json').read())
words=pickle.load(open('./words.pkl','rb '))
classes=pickle.load(open('./classes.pkl','rb'))

def preprocess_user_input(user_input):
    input_word_token_1=nltk.word_tokenize(user_input)
     input_word_token_2=get_stem_words(input_word_token_1,ignore_words)
    input_word_token_2=sorted(list(set(input_word_token_2)))
    bag=[]
    bag_of_words=[]