import nltk
import torch
import torch.nn as nn
import torch.optim as optim
import random
from nltk.corpus import brown
from collections import Counter
from tqdm import tqdm
import numpy as np

nltk.download("brown")

class SkipGramNegativeSampling(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramNegativeSampling, self).__init__()
        self.target_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.context_embedding = nn.Embedding(vocab_size, embedding_dim)

        nn.init.xavier_uniform_(self.target_embedding.weight)
        nn.init.xavier_uniform_(self.context_embedding.weight)

    def forward(self, target_words, context_words, negative_samples):
        target_embeds = self.target_embedding(target_words)  
        context_embeds = self.context_embedding(context_words)  
        negative_embeds = self.context_embedding(negative_samples)  

        positive_score = (target_embeds * context_embeds).sum(dim=1)  

        negative_score = torch.bmm(negative_embeds, target_embeds.unsqueeze(2)).squeeze(2)  

        return positive_score, negative_score

class SkipGramTrainer:
    def __init__(self, context_window=2, embedding_dim=100, vocab_size=5000, num_negative_samples=5, batch_size=1024, epochs=100, learning_rate=0.01):
        self.context_window = context_window
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.num_negative_samples = num_negative_samples
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.word_to_index = {}
        self.index_to_word = {}
        self.training_data = []
        self.word_freqs = []

    def load_corpus(self):
        print("Loading Brown corpus...")
        brown_words = [word.lower() for word in brown.words()]
        word_counts = Counter(brown_words)
        most_common_words = [word for word, _ in word_counts.most_common(self.vocab_size)]

        self.word_to_index = {word: i for i, word in enumerate(most_common_words)}
        self.index_to_word = {i: word for word, i in self.word_to_index.items()}
        self.word_freqs = [word_counts[word] for word in most_common_words]

        word_freqs = np.array([word_counts[word] for word in most_common_words], dtype=np.float32)
        unigram_dist = word_freqs ** 0.75
        unigram_dist /= unigram_dist.sum()
        self.uniform_dist = unigram_dist

        return brown_words

    def build_skipgram_training_data(self, words):
        print("Building Skip-Gram training data...")
        data = []
        for i in tqdm(range(len(words))):
            target_word = words[i]
            if target_word not in self.word_to_index:
                continue
            target_idx = self.word_to_index[target_word]

            for j in range(1, self.context_window + 1):
                if i - j >= 0:
                    context_word = words[i - j]
                    if context_word in self.word_to_index:
                        data.append((target_idx, self.word_to_index[context_word]))

                if i + j < len(words):
                    context_word = words[i + j]
                    if context_word in self.word_to_index:
                        data.append((target_idx, self.word_to_index[context_word]))

        self.training_data = data

    def get_negative_samples(self, true_context, num_samples):
        negatives = []
        while len(negatives) < num_samples:
            sample = np.random.choice(self.vocab_size, p=self.uniform_dist)
            if sample != true_context:
                negatives.append(sample)
        return negatives
    
    def train(self):
        model = SkipGramNegativeSampling(self.vocab_size, self.embedding_dim).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        loss_fn = nn.BCEWithLogitsLoss()

        print("Training Skip-Gram model...")
        for epoch in range(self.epochs):
            random.shuffle(self.training_data)
            total_loss = 0
            for i in tqdm(range(0, len(self.training_data), self.batch_size)):
                batch = self.training_data[i: i + self.batch_size]
                if len(batch) == 0:
                    continue

                target_words, context_words = zip(*batch)
                target_words = torch.tensor(target_words, dtype=torch.long, device=self.device)
                context_words = torch.tensor(context_words, dtype=torch.long, device=self.device)

                negative_samples = self.get_negative_samples(len(target_words))

                optimizer.zero_grad()

                positive_score, negative_score = model(target_words, context_words, negative_samples)
                positive_loss = loss_fn(positive_score, torch.ones_like(positive_score, device=self.device))
                negative_loss = loss_fn(negative_score, torch.zeros_like(negative_score, device=self.device))

                loss = positive_loss + negative_loss
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {total_loss:.4f}")

        return model.target_embedding.weight.data

    def save_embeddings(self, embeddings, filename="skipgram.pt"):
        """Save the trained word embeddings as a dictionary with word mappings."""
        embedding_dict = {self.index_to_word[i]: embeddings[i].cpu().numpy() for i in range(self.vocab_size)}
        
        torch.save(embedding_dict, filename)  # Save as dictionary, NOT tensor
        print(f"Saved embeddings as a dictionary to {filename}")

    def train_and_save(self):
        words = self.load_corpus()
        self.build_skipgram_training_data(words)
        embeddings = self.train()
        self.save_embeddings(embeddings)

if __name__ == "__main__":
    skipgram_trainer = SkipGramTrainer()
    skipgram_trainer.train_and_save()
