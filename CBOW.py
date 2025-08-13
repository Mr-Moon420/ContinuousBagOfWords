import numpy as np
import random
import DeepLearning as dl
import re

class Vocabulary:
    def __init__(self, words):
        self.w2i = {}
        self.i2w = {}

        for word in (words or []):
            self.add(word)

    def size(self):
        return len(self.w2i)

    def add(self, word):
        if word not in self.w2i:
            word_id = len(self.w2i)
            self.w2i[word] = word_id
            self.i2w[word_id] = word

    def getID(self, word):
        return self.w2i[word]
    
    def getWord(self, id):
        return self.i2w[id]
    
    def OHE(self, word):
        return [1 if self.w2i[word] == i else 0 for i in range(len(self.w2i))]
    

class cbow:
    embedding_dim = 5
    sentences = []
    
    def __init__(self, sentences, embedding_dim = 5):
        self.embedding_dim = embedding_dim
        self.sentences = sentences
        self.tokenized_sentences = [re.findall('[a-z]+|[.]', sentence.lower()) for sentence in sentences]
        self.vocab = Vocabulary(word for sentence in self.tokenized_sentences for word in sentence)

        self.embedding = dl.MultiWordEmbedding(self.vocab, self.embedding_dim)

        self.Network = dl.Sequential([
            self.embedding,
            dl.Sum(),
            dl.Linear(self.embedding_dim, self.vocab.size())
        ])

    def train(self, epochs = 100, optimizer = None, loss = None):
        if optimizer == None:
            optimizer = dl.GradientDescent(learning_rate=0.01)
        
        if loss == None:
            loss = dl.SoftMaxCrossEntropy()

        for _ in range(epochs):
            for sentence in self.tokenized_sentences:
                for i, word in enumerate(sentence):
                    inpWords = [s_word for s_word in sentence if s_word != word]
                    input = np.array([self.vocab.w2i[w] for w in inpWords])
                    target = self.vocab.OHE(word)

                    predicted = self.Network.forward(input)
                    gradient = loss.gradient(predicted, target)
                    self.Network.backward(gradient)
                    optimizer.step(self.Network)

    def predict(self, words, k = 5):
        input = np.array([self.vocab.w2i[word] for word in words])
        predicted = self.Network.forward(input)

        sorted = np.argsort(-predicted)[:k]
        sorted_words = [self.vocab.i2w[id] for id in sorted]
        return sorted_words
