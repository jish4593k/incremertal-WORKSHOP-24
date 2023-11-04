import random
import math
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, normalized_mutual_info_score

K = 3

def load_data():
    data = np.genfromtxt("iris.txt", delimiter=',', usecols=(0, 1, 2, 3))
    return data

def kmeans_clustering(data):
    kmeans = KMeans(n_clusters=K, random_state=0)
    kmeans.fit(data)
    labels = kmeans.labels_
    return labels

def calculate_entropy(labels):
    value_counts = np.bincount(labels)
    probabilities = value_counts / len(labels)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def main():
    data = load_data()
    predicted_labels = kmeans_clustering(data)
    true_labels = np.genfromtxt("iris.txt", delimiter=',', usecols=(4), dtype=int)

    accuracy = accuracy_score(true_labels, predicted_labels)
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)
    entropy = calculate_entropy(predicted_labels)

    print("Accuracy: {:.2f}%".format(accuracy * 100))
    print("Normalized Mutual Information: {:.4f}".format(nmi))
    print("Entropy: {:.4f}".format(entropy))

if __name__ == '__main__':
    main()
