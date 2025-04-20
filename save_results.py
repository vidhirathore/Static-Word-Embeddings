import torch
import pandas as pd
import scipy.stats as stats
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

def load_embeddings(embedding_path):
    embeddings = torch.load(embedding_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
    return embeddings

def load_wordsim_dataset(dataset_path="wordsim353.csv"):
    df = pd.read_csv(dataset_path)
    return df["Word 1"].str.lower().tolist(), df["Word 2"].str.lower().tolist(), df["Human"].tolist()

def cosine_similarity(vec1, vec2):
    return torch.nn.functional.cosine_similarity(vec1, vec2, dim=0).item()

def compute_similarity(embeddings, word_to_index, words1, words2, scores):
    predicted_similarities, human_scores = [], []
    valid_word1, valid_word2 = [], []

    for word1, word2, human_score in zip(words1, words2, scores):
        if word1 in word_to_index and word2 in word_to_index:
            vec1 = torch.tensor(embeddings[word1], dtype=torch.float32)
            vec2 = torch.tensor(embeddings[word2], dtype=torch.float32)

            similarity = cosine_similarity(vec1, vec2)
            predicted_similarities.append(similarity)
            human_scores.append(human_score)
            valid_word1.append(word1)
            valid_word2.append(word2)

    return valid_word1, valid_word2, human_scores, predicted_similarities

def plot_scatter(human_scores, predicted_similarities, output_path="scatter_plot.png"):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=human_scores, y=predicted_similarities)
    plt.xlabel("Human Similarity Score")
    plt.ylabel("Predicted Cosine Similarity")
    plt.title("Human vs Predicted Similarity")
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def evaluate(embedding_path, dataset_path="wordsim353.csv", output_csv="similarity_results.csv"):
    print(f"Loading embeddings from {embedding_path}...")
    embeddings = load_embeddings(embedding_path)

    print("Loading WordSim-353 dataset...")
    words1, words2, scores = load_wordsim_dataset(dataset_path)

    print("Computing word similarities...")
    word_to_index = {word: word for word in embeddings.keys()} 

    valid_word1, valid_word2, human_scores, predicted_similarities = compute_similarity(
        embeddings, word_to_index, words1, words2, scores
    )

    if not predicted_similarities:
        print("No valid word pairs found in the dataset.")
        return

    spearman_corr, _ = stats.spearmanr(predicted_similarities, human_scores)
    print(f"Spearman’s Rank Correlation: {spearman_corr:.4f}")

    print(f"Saving results to {output_csv}...")
    result_df = pd.DataFrame({
        "Word 1": valid_word1,
        "Word 2": valid_word2,
        "Human Score": human_scores,
        "Predicted Similarity": predicted_similarities
    })
    result_df.to_csv(output_csv, index=False)
    print("Saved successfully.")

    print("Saving scatter plot...")
    plot_scatter(human_scores, predicted_similarities)
    print("Plot saved as scatter_plot.png.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Word Similarity using Pretrained Embeddings")
    parser.add_argument("embedding_path", type=str, help="Path to the .pt embedding file")
    args = parser.parse_args()

    evaluate(args.embedding_path)
