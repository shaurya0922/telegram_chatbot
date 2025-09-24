import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

sentence = "I will be running for fifteen kilometers today!"
tokens = word_tokenize(sentence)
print(tokens)
# ['Chatbots', 'are', 'becoming', 'smarter', 'every', 'day', '.']
