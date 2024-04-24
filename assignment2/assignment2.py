import re
import numpy as np
import pandas as pd

# Function to preprocess tweets
def preprocess_tweet(tweet):
    # Remove tweet id and timestamp
    tweet = tweet.split('|')[-1]
    # Remove any word that starts with @
    tweet = re.sub(r'@\w+', '', tweet)
    # Remove hashtag symbols
    tweet = re.sub(r'#', '', tweet)
    # Remove any URL
    tweet = re.sub(r'http\S+', '', tweet)
    # Convert every word to lowercase
    tweet = tweet.lower()
    return tweet

# Function to calculate Euclidean distance between two points
def euclidean_distance(point1, point2):
    return sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)) ** 0.5

# Function to initialize centroids
def initialize_centroids(data, k):
    centroids = []
    for _ in range(k):
        centroids.append(data[_ % len(data)])
    return centroids


# Function to perform K-means clustering
def k_means(data, k, max_iterations=100):
    centroids = initialize_centroids(data, k)
    for _ in range(max_iterations):
        clusters = [[] for _ in range(k)]
        for point in data:
            distances = [euclidean_distance(point, centroid) for centroid in centroids]
            cluster_index = distances.index(min(distances))
            clusters[cluster_index].append(point)
        new_centroids = [tuple(sum(y) / len(y) for y in zip(*cluster)) if cluster else (0,) * len(data[0]) for cluster in clusters]
        if new_centroids == centroids:
            break
        centroids = new_centroids
    return clusters

# Function to read dataset from a text file
def read_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        dataset = file.readlines()
    return dataset

# Dataset
file_path = 'Health-Tweets/bbchealth.txt'  # Replace with the path to dataset file in Health-Tweets
tweets = read_dataset(file_path)

preprocessed_tweets = [preprocess_tweet(tweet) for tweet in tweets]

numerical_data = [(len(tweet),) for tweet in preprocessed_tweets]

# Perform K-means clustering for different values of K
K_values = [1, 5, 10, 50, 100]
results = []
for K in K_values:
    clusters = k_means(numerical_data, K)
    sse = sum(sum(euclidean_distance(point, centroid) ** 2 for point in cluster) for centroid, cluster in zip(initialize_centroids(numerical_data, K), clusters))
    tmp = []
    for i, cluster in enumerate(clusters):
        tmp.append(f"{i+1}: {len(cluster)} tweets")
    results.append([K, sse] + [tmp])

# Convert results to DataFrame
results_df = pd.DataFrame(results, columns=['Value of K', 'SSE', 'Size of each Cluster'])

# Export results to CSV
results_df.to_csv('output.csv', index=False)