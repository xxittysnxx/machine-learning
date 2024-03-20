import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, activation='sigmoid'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.activation = activation
        
        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        self.bias_output = np.zeros((1, self.output_size))
        
        # Activation function
        if activation == 'sigmoid':
            self.activation_func = self.sigmoid
            self.activation_derivative = self.sigmoid_derivative
        elif activation == 'tanh':
            self.activation_func = np.tanh
            self.activation_derivative = self.tanh_derivative
        elif activation == 'relu':
            self.activation_func = self.relu
            self.activation_derivative = self.relu_derivative
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def tanh_derivative(self, x):
        return 1 - np.tanh(x)**2
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return np.where(x <= 0, 0, 1)
    
    def forward_pass(self, X):
        self.hidden_output = self.activation_func(np.dot(X, self.weights_input_hidden) + self.bias_hidden)
        self.output = self.activation_func(np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output)
    
    def backward_pass(self, X, y):
        output_error = y - self.output
        output_delta = output_error * self.activation_derivative(self.output)
        
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.activation_derivative(self.hidden_output)
        
        self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * self.learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * self.learning_rate
        self.weights_input_hidden += X.T.dot(hidden_delta) * self.learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * self.learning_rate
    
    def train(self, X_train, y_train, epochs=1000):
        for epoch in range(epochs):
            self.forward_pass(X_train)
            self.backward_pass(X_train, y_train)
        # Calculate training accuracy
        self.forward_pass(X_train)
        predictions = np.round(self.output)
        accuracy = np.mean(predictions == y_train)
        return accuracy
    
    def predict(self, X_test):
        self.forward_pass(X_test)
        return np.round(self.output)

# Load data
data = pd.read_csv('https://raw.githubusercontent.com/xxittysnxx/machine-learning/main/assignment1/rice.csv')
data['Class'] = data['Class'].map({'Cammeo': 1, 'Osmancik': 0})
# Preprocess data
X = data.drop('Class', axis=1).values
y = data['Class'].values.reshape(-1, 1)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


activations = ['sigmoid', 'tanh', 'relu']
learning_rates = [0.01, 0.1, 1, 10]
iterations = [1, 10, 100, 1000]
hidden_sizes = [1, 10, 100, 1000]

results = []

for act in activations:
    for lr in learning_rates:
        for iter in iterations:
            for hidden in hidden_sizes:
                nn = NeuralNetwork(input_size=X_train.shape[1], hidden_size=hidden, output_size=1, learning_rate=lr, activation=act)
                train_accuracy = nn.train(X_train, y_train, epochs=iter)
                test_predictions = nn.predict(X_test)
                test_accuracy = np.mean(test_predictions == y_test)
                results.append([act, hidden, lr, iter, test_accuracy])

# Convert results to DataFrame
results_df = pd.DataFrame(results, columns=['activations', 'hidden', 'learningrates', 'iterations', 'accuracy'])

# Format accuracy as percentage
results_df['accuracy'] = (results_df['accuracy'] * 100).astype(str) + '%'

# Export results to CSV
results_df.to_csv('output.csv', index=False)
