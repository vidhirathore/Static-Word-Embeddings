import torch
import pandas as pd
import scipy.stats as stats
import argparse

def load_embeddings(embedding_path):
    """Load word embeddings from a .pt file."""
    embeddings = torch.load(embedding_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
    return embeddings

def load_wordsim_dataset(dataset_path="wordsim353.csv"):
    """Load WordSim-353 dataset and return word pairs with human similarity scores."""
    df = pd.read_csv(dataset_path)
    return df["Word 1"].str.lower().tolist(), df["Word 2"].str.lower().tolist(), df["Human"].tolist()

def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two word vectors."""
    return torch.nn.functional.cosine_similarity(vec1, vec2, dim=0).item()

# def compute_similarity(embeddings, word_to_index, words1, words2, scores):
#     """Compute cosine similarity for word pairs and return valid results."""
#     predicted_similarities, human_scores = [], []

#     for word1, word2, human_score in zip(words1, words2, scores):
#         if word1 in word_to_index and word2 in word_to_index:
#             vec1 = embeddings[word_to_index[word1]]
#             vec2 = embeddings[word_to_index[word2]]

#             similarity = cosine_similarity(vec1, vec2)
#             predicted_similarities.append(similarity)
#             human_scores.append(human_score)

#     return predicted_similarities, human_scores

def compute_similarity(embeddings, word_to_index, words1, words2, scores):
    """Compute cosine similarity for word pairs and return valid results."""
    predicted_similarities, human_scores = [], []

    for word1, word2, human_score in zip(words1, words2, scores):
        if word1 in word_to_index and word2 in word_to_index:
            vec1 = torch.tensor(embeddings[word1], dtype=torch.float32)
            vec2 = torch.tensor(embeddings[word2], dtype=torch.float32)

            similarity = cosine_similarity(vec1, vec2)
            predicted_similarities.append(similarity)
            human_scores.append(human_score)

    return predicted_similarities, human_scores


def evaluate(embedding_path, dataset_path="wordsim353.csv"):
    """Evaluate embeddings using WordSim-353 and compute Spearman correlation."""
    print(f"Loading embeddings from {embedding_path}...")
    embeddings = load_embeddings(embedding_path)

    print("Loading WordSim-353 dataset...")
    words1, words2, scores = load_wordsim_dataset(dataset_path)

    print("Computing word similarities...")
    word_to_index = {word: word for word in embeddings.keys()} 

    predicted_similarities, human_scores = compute_similarity(embeddings, word_to_index, words1, words2, scores)

    if not predicted_similarities:
        print("No valid word pairs found in the dataset.")
        return

    spearman_corr, _ = stats.spearmanr(predicted_similarities, human_scores)
    print(f"Spearman’s Rank Correlation: {spearman_corr:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Word Similarity using Pretrained Embeddings")
    parser.add_argument("embedding_path", type=str, help="Path to the .pt embedding file")
    args = parser.parse_args()

    evaluate(args.embedding_path)
    embeddings = torch.load("skipgram.pt")  