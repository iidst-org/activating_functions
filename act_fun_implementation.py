import numpy as np
import matplotlib.pyplot as plt

# Define activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)  # NumPy has a built-in tanh function

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

# Generate input values
x = np.linspace(-10, 10, 400)

# Compute activation function values
y_sigmoid = sigmoid(x)
y_tanh = tanh(x)
y_relu = relu(x)
y_leaky_relu = leaky_relu(x)

# Create plots
plt.figure(figsize=(12, 8))

# Sigmoid
plt.subplot(2, 2, 1)
plt.plot(x, y_sigmoid, label="Sigmoid", color="blue")
plt.title("Sigmoid Activation Function")
plt.axhline(0, color='black', linewidth=0.5, linestyle="--")
plt.axvline(0, color='black', linewidth=0.5, linestyle="--")
plt.legend()

# Tanh
plt.subplot(2, 2, 2)
plt.plot(x, y_tanh, label="Tanh", color="red")
plt.title("Tanh Activation Function")
plt.axhline(0, color='black', linewidth=0.5, linestyle="--")
plt.axvline(0, color='black', linewidth=0.5, linestyle="--")
plt.legend()

# ReLU
plt.subplot(2, 2, 3)
plt.plot(x, y_relu, label="ReLU", color="green")
plt.title("ReLU Activation Function")
plt.axhline(0, color='black', linewidth=0.5, linestyle="--")
plt.axvline(0, color='black', linewidth=0.5, linestyle="--")
plt.legend()

# Leaky ReLU
plt.subplot(2, 2, 4)
plt.plot(x, y_leaky_relu, label="Leaky ReLU", color="purple")
plt.title("Leaky ReLU Activation Function")
plt.axhline(0, color='black', linewidth=0.5, linestyle="--")
plt.axvline(0, color='black', linewidth=0.5, linestyle="--")
plt.legend()

# Show the plots
plt.tight_layout()
plt.show()
