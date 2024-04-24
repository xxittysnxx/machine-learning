# Tweets Clustering using K-Means
This repository contains a Python script for training and evaluating a k-means clustering about different health news in Twitter. The classifier aims to calculate the Sum of Sqoare Errors to predict the k-means clustering model.

## Prerequisites
Before running the script, ensure you have the following dependencies installed:

- Python 3.x
- NumPy
- pandas

You can install these dependencies using pip:

```
pip install numpy pandas
```

## Usage
1. Clone the Repository

```
git clone https://github.com/xxittysnxx/machine-learning.git
```

2. Navigate to the Directory

```
cd machine-learning/assignment2
```

3. Run the Script
Execute the Python script assignment2.py:

```
python assignment2.py
```

4. View Results
After running the script, the results will be saved to a file named output.csv. You can view these results to analyze the performance of the k-meaans clustering under different configurations.

## Dataset
The dataset used in this script (bbchealth.txt) contains information about different health news in Twitter, including features such as tweet id|date and time|tweet, and the separator is '|'.

## Customization
You can customize the k-means model by adjusting the following parameters:

- Datasets: Choose from folder 'Health-Tweets'.
- K values: Experiment with different K values clustering (e.g., 1, 5, 10, 50, 100).
- Iterations: Set the number of k-means iterations (e.g., 1, 10, 100, 1000).

Feel free to modify these parameters to optimize the performance of the k-means clustering.
