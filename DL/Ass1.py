# Step 1: Load the dataset
import pandas as pd
import numpy as np

data = pd.read_csv("bostonHousing.csv")
print(data.head())
print(data.columns)

# Step 2: Preprocess the dataset
# EDA - optional but good practice
import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(data["medv"])
plt.title("Distribution of MEDV (House Prices)")
plt.show()

sns.boxplot(x=data["medv"])
plt.title("Boxplot of MEDV")
plt.show()

# Handling missing values
print("Missing values:\n", data.isnull().sum())

# Separate features and target
X = data.drop('medv', axis=1)   #all column except medv is stored in x
y = data['medv']

# Standardize features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)  #convert numeric data to have mean=0 and sd=1

# Step 3: Split the dataset
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

print('Training set shape:', X_train.shape, y_train.shape)
#x_train.shape show no of R & C  in i/n data
#y_train.shape shoe no of R in o/p data
print('Testing set shape:', X_test.shape, y_test.shape)

# Step 4: Define the model architecture
from keras.models import Sequential
from keras.layers import Dense, Dropout

model = Sequential()  #layers r stacked one after another
model.add(Dense(128, activation='relu', input_dim=X_train.shape[1])) #input layer
model.add(Dense(64, activation='relu')) #hidden layer 1
model.add(Dense(32, activation='relu'))  #hidden layer 2
model.add(Dense(16, activation='relu')) #hidden layer 3
model.add(Dense(1))  # Output layer

print(model.summary())

# Step 5: Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Step 6: Train the model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2
)

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#An epoch refers to one complete cycle through the entire training dataset during the training process of a machine learning model.

# Step 7: Evaluate the model
loss, mae = model.evaluate(X_test, y_test)
print("Mean Absolute Error on Test Set:", mae)
