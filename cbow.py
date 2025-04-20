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

PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'

class CBOW_NegativeSampling(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_negative_samples):
        super(CBOW_NegativeSampling, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.context_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.num_negative_samples = num_negative_samples

        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.context_embedding.weight)

    def forward(self, context_words, negative_words):
        """
        Forward pass for CBOW with Negative Sampling.
        - context_words: Tensor of shape (batch_size, 2 * context_window)
        - negative_words: Tensor of shape (batch_size, num_negative_samples)
        """
        context_embeds = self.embedding(context_words)  # (batch_size, 2*context_window, embedding_dim)
        target_embeds = self.context_embedding(negative_words)  # (batch_size, num_negative_samples, embedding_dim)

        context_mean = context_embeds.mean(dim=1)  # (batch_size, embedding_dim)
        positive_score = torch.bmm(target_embeds, context_mean.unsqueeze(2)).squeeze(2)  # (batch_size, num_negative_samples)

        return positive_score


class CBOWTrainer:
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
        """Load and preprocess the Brown corpus."""
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

    def build_cbow_training_data(self, words):
        """Generate CBOW training samples (context → target)."""
        print("Building CBOW training data...")
        data = []
        for i in tqdm(range(self.context_window, len(words) - self.context_window)):
            target_word = words[i]
            if target_word not in self.word_to_index:
                continue
            context = [
                words[i - j] for j in range(1, self.context_window + 1)
            ] + [
                words[i + j] for j in range(1, self.context_window + 1)
            ]

            if all(w in self.word_to_index for w in context):
                target_idx = self.word_to_index[target_word]
                context_indices = [self.word_to_index[w] for w in context]
                data.append((context_indices, target_idx))

        self.training_data = data

    def get_negative_samples(self, true_context, num_samples):
        negatives = []
        while len(negatives) < num_samples:
            sample = np.random.choice(self.vocab_size, p=self.uniform_dist)
            if sample != true_context:
                negatives.append(sample)
        return negatives
    
    def train(self):
        """Train the CBOW model with Negative Sampling."""
        model = CBOW_NegativeSampling(self.vocab_size, self.embedding_dim, self.num_negative_samples).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        loss_fn = nn.BCEWithLogitsLoss()

        print("Training CBOW model...")
        for epoch in range(self.epochs):
            random.shuffle(self.training_data)
            total_loss = 0
            for i in tqdm(range(0, len(self.training_data), self.batch_size)):
                batch = self.training_data[i: i + self.batch_size]
                if len(batch) == 0:
                    continue

                context_words, target_words = zip(*batch)
                context_words = torch.tensor(context_words, dtype=torch.long, device=self.device)
                target_words = torch.tensor(target_words, dtype=torch.long, device=self.device).unsqueeze(1)

                negative_samples = self.get_negative_samples(len(target_words))

                optimizer.zero_grad()

                positive_score = model(context_words, target_words)
                positive_loss = loss_fn(positive_score, torch.ones_like(positive_score, device=self.device))

                negative_score = model(context_words, negative_samples)
                negative_loss = loss_fn(negative_score, torch.zeros_like(negative_score, device=self.device))

                loss = positive_loss + negative_loss
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {total_loss:.4f}")

        embeddings_dict = {self.index_to_word[i]: model.embedding.weight[i].detach().cpu() for i in range(self.vocab_size)}
        return embeddings_dict  

    def save_embeddings(self, embeddings, filename="cbow.pt"):
        """Save embeddings as a dictionary {word: vector}."""
        torch.save(embeddings, filename) 
        print(f"Saved embeddings to {filename}")


    def train_and_save(self):
        """Pipeline to train CBOW embeddings and save them."""
        words = self.load_corpus()
        self.build_cbow_training_data(words)
        embeddings = self.train()
        self.save_embeddings(embeddings)


if __name__ == "__main__":
    cbow_trainer = CBOWTrainer()
    cbow_trainer.train_and_save()
