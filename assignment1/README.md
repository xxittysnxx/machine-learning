# Neural Network Classifier for Rice Dataset
This repository contains a Python script for training and evaluating a neural network classifier on the Rice dataset. The classifier aims to predict the variety of rice (Cammeo or Osmancik) based on various features.

## Prerequisites
Before running the script, ensure you have the following dependencies installed:

- Python 3.x
- NumPy
- pandas
- scikit-learn

You can install these dependencies using pip:

```
pip install numpy pandas scikit-learn
```

## Usage
1. Clone the Repository

```
git clone https://github.com/xxittysnxx/machine-learning.git
```

2. Navigate to the Directory

```
cd machine-learning/assignment1
```

3. Run the Script
Execute the Python script neural_network_classifier.py:

```
python assignment1.py
```

4. View Results
After running the script, the results will be saved to a file named output.csv. You can view these results to analyze the performance of the neural network classifier under different configurations.

## Dataset
The dataset used in this script (rice.csv) contains information about different rice varieties, including features such as area, perimeter, major axis length, minor axis length, eccentricity, convex area, and extent. The target variable 'Class' specifies the rice variety (Cammeo or Osmancik).

## Customization
You can customize the neural network model by adjusting the following parameters:

- Activation Functions: Choose from 'sigmoid', 'tanh', or 'relu'.
- Learning Rates: Experiment with different learning rates (e.g., 0.01, 0.1, 1, 10).
- Iterations: Set the number of training epochs (e.g., 1, 10, 100, 1000).
- Hidden Layer Sizes: Define the size of the hidden layers (e.g., 1, 10, 100, 1000).

Feel free to modify these parameters to optimize the performance of the neural network classifier for the Rice dataset.
