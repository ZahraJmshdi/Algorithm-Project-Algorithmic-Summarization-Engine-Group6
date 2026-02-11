import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class TextRankSummarizer:
    def __init__(self, word_embeddings_path='glove.6B.100d.txt'):

        self.word_embeddings = self.load_embeddings(word_embeddings_path)
        self.stop_words = set(stopwords.words('english'))

    def load_embeddings(self, path):
        embeddings = {}
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                embeddings[word] = vector
        return embeddings

    def sentence_to_vector(self, sentence):
        words = word_tokenize(sentence.lower())
        words = [w for w in words if w.isalpha() and w not in self.stop_words]
        
        if len(words) == 0:
            return np.zeros((100,)) 
        
        vector_sum = np.zeros((100,))
        for w in words:
            if w in self.word_embeddings:
                vector_sum += self.word_embeddings[w]
        return vector_sum / len(words)

    def summarize(self, text, num_sentences=3):
        sentences = sent_tokenize(text)
        if len(sentences) <= num_sentences:
            return text
        
        sentence_vectors = [self.sentence_to_vector(sent) for sent in sentences]
        
        sim_matrix = np.zeros((len(sentences), len(sentences)))
        for i in range(len(sentences)):
            for j in range(len(sentences)):
                if i != j:
                    sim = cosine_similarity(
                        sentence_vectors[i].reshape(1, -1),
                        sentence_vectors[j].reshape(1, -1)
                    )
                    sim_matrix[i][j] = sim[0][0]
        
        nx_graph = nx.from_numpy_array(sim_matrix)
        scores = nx.pagerank(nx_graph)
        
        ranked_sentences = sorted(((scores[i], i, sent) for i, sent in enumerate(sentences)), reverse=True)
        top_sentence_indices = sorted([ranked_sentences[i][1] for i in range(num_sentences)])
        
        summary = ' '.join([sentences[i] for i in top_sentence_indices])
        return summary
