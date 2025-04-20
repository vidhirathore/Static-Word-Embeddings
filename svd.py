import nltk
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from nltk.corpus import brown
from collections import Counter
from tqdm import tqdm

nltk.download("brown")

class WordEmbeddingSVD:
    def __init__(self, context_window=2, embedding_dim=100, vocab_size=5000):
        self.context_window = context_window
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.word_to_index = {}
        self.index_to_word = {}
        self.cooc_matrix = None

    def load_corpus(self):
        """Load and preprocess the Brown corpus."""
        print("Loading Brown corpus...")
        brown_words = [word.lower() for word in brown.words()]
        word_counts = Counter(brown_words)
        most_common_words = [word for word, _ in word_counts.most_common(self.vocab_size)]

        self.word_to_index = {word: i for i, word in enumerate(most_common_words)}
        self.index_to_word = {i: word for word, i in self.word_to_index.items()}

        return brown_words

    def build_cooccurrence_matrix(self, brown_words):
        """Construct the co-occurrence matrix."""
        print("Building co-occurrence matrix...")
        self.cooc_matrix = torch.zeros((self.vocab_size, self.vocab_size), dtype=torch.float32, device=self.device)

        for i, word in enumerate(tqdm(brown_words)):
            if word not in self.word_to_index:
                continue
            word_idx = self.word_to_index[word]

            for j in range(1, self.context_window + 1):
                if i - j >= 0:
                    left_word = brown_words[i - j]
                    if left_word in self.word_to_index:
                        left_idx = self.word_to_index[left_word]
                        self.cooc_matrix[word_idx, left_idx] += 1

                if i + j < len(brown_words):
                    right_word = brown_words[i + j]
                    if right_word in self.word_to_index:
                        right_idx = self.word_to_index[right_word]
                        self.cooc_matrix[word_idx, right_idx] += 1

    def compute_svd(self):
        """Perform Singular Value Decomposition (SVD) to extract embeddings."""
        print("Performing SVD on GPU...")
        U, S, Vt = torch.linalg.svd(self.cooc_matrix)
        word_embeddings = U[:, :self.embedding_dim] * S[:self.embedding_dim].unsqueeze(0)

        # Convert embeddings to a dictionary {word: vector}
        embeddings_dict = {self.index_to_word[i]: word_embeddings[i] for i in range(self.vocab_size)}
        return embeddings_dict

    def save_embeddings(self, word_embeddings, filename="svd.pt"):
        """Save the learned word embeddings."""
        torch.save(word_embeddings, filename)
        print(f"Saved embeddings to {filename}")

    def train_and_save(self):
        """Pipeline to train embeddings and save them."""
        words = self.load_corpus()
        self.build_cooccurrence_matrix(words)
        embeddings = self.compute_svd()
        self.save_embeddings(embeddings)


if __name__ == "__main__":
    svd_model = WordEmbeddingSVD()
    svd_model.train_and_save()
