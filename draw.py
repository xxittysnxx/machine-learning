import matplotlib.pyplot as plt

# Results data
learning_rates = [0.0001, 0.0001, 0.0001, 0.0001]
hidden_layers = [7, 9, 11, 13]
train_losses = [5.621421432834362, 4.459886281146871, 9.123426210633959, 7.861226617171531]
validation_losses = [32.892977309358734, 30.00859365366771, 24.674620024613922, 23.782733212865452]
validation_rmse = [2931.85, 2861.13, 1995.80, 1857.61]

# Plotting
plt.figure(figsize=(12, 6))

# Plot train and validation losses
plt.subplot(1, 2, 1)
plt.plot(hidden_layers, train_losses, marker='o', label='Train Loss')
plt.plot(hidden_layers, validation_losses, marker='o', label='Validation Loss')
plt.xlabel('Hidden Layers')
plt.ylabel('Loss')
plt.title('Train and Validation Losses')
plt.legend()

# Plot validation RMSE
plt.subplot(1, 2, 2)
plt.plot(hidden_layers, validation_rmse, marker='o', color='green')
plt.xlabel('Hidden Layers')
plt.ylabel('RMSE')
plt.title('Validation RMSE')

plt.tight_layout()

# Save the plots
plt.savefig('losses_and_rmse.png')

# Show the plots
plt.show()