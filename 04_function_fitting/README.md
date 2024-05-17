

# 1. 正弦函数拟合

### 1. [0,4π] 300点3个隐藏层1500周期

<p align="center">
<img src="https://19640810.xyz/05_image/01_imageHost/20240517-164955.png" alt="Image Description" width="700">
</p>

```py
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Generate the dataset for [0, 4π]
x = np.linspace(0, 4 * np.pi, 300)
y = np.sin(x)

# Define a neural network model with adjusted parameters
def build_model():
    model = Sequential()
    model.add(Dense(64, input_dim=1, activation='relu'))  # Input layer with 64 neurons
    model.add(Dense(128, activation='relu'))  # First hidden layer with 128 neurons
    model.add(Dense(128, activation='relu'))  # Second hidden layer with 128 neurons
    model.add(Dense(64, activation='relu'))  # Third hidden layer with 64 neurons
    model.add(Dense(1, activation='linear'))  # Output layer with linear activation
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001))
    return model

# Create the model
nn_model = build_model()

# Fit the model to the data
nn_model.fit(x[:, np.newaxis], y, epochs=1500, verbose=0)

# Predict the values on a dense grid for smooth plotting
x_fit = np.linspace(0, 4 * np.pi, 4000)
y_fit = nn_model.predict(x_fit[:, np.newaxis])

# Plot the original data and the neural network fit for the new interval
plt.figure(figsize=(12, 8))
plt.plot(x, y, 'o', label='Original Data [0, 4π]')
plt.plot(x_fit, y_fit, '-', label='Optimized Neural Network Fit [0, 4π]')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Optimized Neural Network Fit for y = sin(x) over [0, 4π]')
plt.legend()
plt.show()
```

<p align="center">
<img src="https://19640810.xyz/05_image/01_imageHost/20240517-170028.png" alt="Image Description" width="700">
</p>


### 2. [0,4π] 500点5个隐藏层2000周期

<p align="center">
<img src="https://19640810.xyz/05_image/01_imageHost/20240517-170453.png" alt="Image Description" width="700">
</p>

```py
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Generate the dataset for [0, 4π] with more points
x = np.linspace(0, 4 * np.pi, 500)  # Increase the number of data points
y = np.sin(x)

# Define a neural network model with more hidden layers and neurons
def build_model():
    model = Sequential()
    model.add(Dense(128, input_dim=1, activation='relu'))  # Input layer with 128 neurons
    model.add(Dense(256, activation='relu'))  # First hidden layer with 256 neurons
    model.add(Dense(256, activation='relu'))  # Second hidden layer with 256 neurons
    model.add(Dense(128, activation='relu'))  # Third hidden layer with 128 neurons
    model.add(Dense(128, activation='relu'))  # Fourth hidden layer with 128 neurons
    model.add(Dense(64, activation='relu'))   # Fifth hidden layer with 64 neurons
    model.add(Dense(1, activation='linear'))  # Output layer with linear activation
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001))
    return model

# Create the model
nn_model = build_model()

# Fit the model to the data
nn_model.fit(x[:, np.newaxis], y, epochs=2000, verbose=0)  # Increase the number of epochs

# Predict the values on a dense grid for smooth plotting
x_fit = np.linspace(0, 4 * np.pi, 5000)
y_fit = nn_model.predict(x_fit[:, np.newaxis])

# Plot the original data and the neural network fit for the new interval
plt.figure(figsize=(12, 8))
plt.plot(x, y, 'o', label='Original Data [0, 4π]')
plt.plot(x_fit, y_fit, '-', label='Enhanced Neural Network Fit [0, 4π]')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Enhanced Neural Network Fit for y = sin(x) over [0, 4π]')
plt.legend()
plt.show()
```


### 3. [0,4π] 500点7个隐藏层2500周期_128神经元数

<p align="center">
<img src="https://19640810.xyz/05_image/01_imageHost/20240517-171109.png" alt="Image Description" width="700">
</p>

```py
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Generate the dataset for [0, 4π] with more points
x = np.linspace(0, 4 * np.pi, 500)  # Number of data points
y = np.sin(x)

# Define a neural network model with more hidden layers
def build_model():
    model = Sequential()
    model.add(Dense(128, input_dim=1, activation='relu'))  # Input layer with 128 neurons
    model.add(Dense(256, activation='relu'))  # First hidden layer with 256 neurons
    model.add(Dense(256, activation='relu'))  # Second hidden layer with 256 neurons
    model.add(Dense(128, activation='relu'))  # Third hidden layer with 128 neurons
    model.add(Dense(128, activation='relu'))  # Fourth hidden layer with 128 neurons
    model.add(Dense(64, activation='relu'))   # Fifth hidden layer with 64 neurons
    model.add(Dense(64, activation='relu'))   # Sixth hidden layer with 64 neurons
    model.add(Dense(32, activation='relu'))   # Seventh hidden layer with 32 neurons
    model.add(Dense(1, activation='linear'))  # Output layer with linear activation
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001))
    return model

# Create the model
nn_model = build_model()

# Fit the model to the data
nn_model.fit(x[:, np.newaxis], y, epochs=2500, verbose=0)  # Increased epochs for deeper network

# Predict the values on a dense grid for smooth plotting
x_fit = np.linspace(0, 4 * np.pi, 5000)
y_fit = nn_model.predict(x_fit[:, np.newaxis])

# Plot the original data and the neural network fit for the new interval
plt.figure(figsize=(12, 8))
plt.plot(x, y, 'o', label='Original Data [0, 4π]')
plt.plot(x_fit, y_fit, '-', label='Deep Neural Network Fit [0, 4π]')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Deep Neural Network Fit for y = sin(x) over [0, 4π]')
plt.legend()
plt.show()
```

### 4. [0,4π] 500点7个隐藏层2500周期_100神经元数

- 保持了原有的网络层数，但改变了每层的神经元数量

<p align="center">
<img src="https://19640810.xyz/05_image/01_imageHost/20240517-171648.png" alt="Image Description" width="700">
</p>

```py
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Generate the dataset for [0, 4π] with more points
x = np.linspace(0, 4 * np.pi, 500)  # Number of data points
y = np.sin(x)

# Define a neural network model with adjusted neurons per layer
def build_model():
    model = Sequential()
    model.add(Dense(64, input_dim=1, activation='relu'))  # Input layer with 64 neurons
    model.add(Dense(100, activation='relu'))  # First hidden layer with 100 neurons
    model.add(Dense(100, activation='relu'))  # Second hidden layer with 100 neurons
    model.add(Dense(50, activation='relu'))   # Third hidden layer with 50 neurons
    model.add(Dense(50, activation='relu'))   # Fourth hidden layer with 50 neurons
    model.add(Dense(25, activation='relu'))   # Fifth hidden layer with 25 neurons
    model.add(Dense(1, activation='linear'))  # Output layer with linear activation
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001))
    return model

# Create the model
nn_model = build_model()

# Fit the model to the data
nn_model.fit(x[:, np.newaxis], y, epochs=2000, verbose=0)  # Increased epochs for deeper network

# Predict the values on a dense grid for smooth plotting
x_fit = np.linspace(0, 4 * np.pi, 5000)
y_fit = nn_model.predict(x_fit[:, np.newaxis])

# Plot the original data and the neural network fit for the new interval
plt.figure(figsize=(12, 8))
plt.plot(x, y, 'o', label='Original Data [0, 4π]')
plt.plot(x_fit, y_fit, '-', label='Adjusted Neuron Network Fit [0, 4π]')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Adjusted Neuron Network Fit for y = sin(x) over [0, 4π]')
plt.legend()
plt.show()
```






