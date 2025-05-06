from keras.datasets import imdb
from keras import models
from keras import layers
import numpy as np

# Load the IMDB dataset
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# word_index is a dictionary mapping words to an integer index
word_index = imdb.get_word_index()
# We reverse it, mapping integer indices to words
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# We decode the review; note that our indices were offset by 3
# because 0, 1 and 2 are reserved indices for "padding", "start of sequence", and "unknown".
decoded_review = ' '.join([reverse_word_index.get(i - 3 , '.') for i in train_data[0]])

decoded_review

def vectorize_sequences(sequences, dimension=10000):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.  # set specific indices of results[i] to 1s
    return results

# Our vectorized training data
x_train = vectorize_sequences(train_data)
# Our vectorized test data
x_test = vectorize_sequences(test_data)

# Our vectorized labels
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

# Define the model architecture
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model with validation data
history = model.fit(x_train, y_train,
                    epochs=20,
                    batch_size=512,
                    validation_split=0.2)

# Evaluate the model on test data
results = model.evaluate(x_test, y_test)

# Print test loss and accuracy
print("Test loss:", results[0])
print("Test accuracy:", results[1])

"""
Classification evaluation metrics are used to assess the performance of machine learning models in classification tasks, where the goal is to predict the class label of instances based on their features. These metrics provide insights into how well the model is performing and help in comparing different models or tuning their parameters. Here are some common classification evaluation metrics:

1. **Accuracy**: Accuracy measures the proportion of correctly classified instances out of the total number of instances. It is calculated as the ratio of the number of correct predictions to the total number of predictions.

   \[ \text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}} \]

   Accuracy is a widely used metric when classes are balanced, meaning that each class occurs with similar frequency in the dataset. However, it can be misleading in imbalanced datasets, where one class dominates the others.

2. **Precision**: Precision measures the proportion of true positive predictions out of all positive predictions. It quantifies the model's ability to avoid false positives.

   \[ \text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}} \]

3. **Recall (Sensitivity)**: Recall, also known as sensitivity or true positive rate, measures the proportion of true positive predictions out of all actual positive instances. It quantifies the model's ability to capture all positive instances.

   \[ \text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}} \]

4. **F1 Score**: The F1 score is the harmonic mean of precision and recall. It provides a balanced measure of the model's performance, considering both false positives and false negatives.

   \[ \text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} \]

5. **Specificity**: Specificity measures the proportion of true negative predictions out of all actual negative instances. It quantifies the model's ability to correctly identify negative instances.

   \[ \text{Specificity} = \frac{\text{True Negatives}}{\text{True Negatives} + \text{False Positives}} \]

6. **False Positive Rate (FPR)**: FPR measures the proportion of false positive predictions out of all actual negative instances. It is complementary to specificity.

   \[ \text{FPR} = \frac{\text{False Positives}}{\text{False Positives} + \text{True Negatives}} \]

7. **Receiver Operating Characteristic (ROC) Curve**: The ROC curve is a graphical representation of the trade-off between true positive rate (sensitivity) and false positive rate (1 - specificity) for different threshold values. It helps in assessing the model's performance across various thresholds.

8. **Area Under the ROC Curve (AUC-ROC)**: AUC-ROC measures the area under the ROC curve. It provides a single value to summarize the model's performance across all possible threshold values. A higher AUC-ROC indicates better overall performance.

These are some of the key classification evaluation metrics used to assess the performance of machine learning models. The choice of metrics depends on the specific characteristics of the dataset and the objectives of the classification task.

Binary classification is a type of supervised learning task where the goal is to classify instances into one of two possible classes or categories. The two classes are typically referred to as the positive class and the negative class. Examples of binary classification tasks include:

1. Email spam detection: Classify emails as either spam (positive class) or not spam (negative class).
2. Disease diagnosis: Classify patients as either having a particular disease (positive class) or not having the disease (negative class).
3. Sentiment analysis: Classify text as either positive sentiment (positive class) or negative sentiment (negative class).

In binary classification, each instance in the dataset is represented by a set of features or attributes, and the task is to learn a model that can accurately predict the class label of new instances based on their features. The model learns patterns or relationships in the training data and uses them to make predictions on unseen data.

There are several algorithms commonly used for binary classification tasks, including:

1. **Logistic Regression**: Despite its name, logistic regression is a classification algorithm suitable for binary classification tasks. It models the probability that an instance belongs to a particular class using the logistic function, which maps the input features to the range [0, 1].

2. **Support Vector Machines (SVM)**: SVM is a powerful classification algorithm that works by finding the optimal hyperplane that separates the two classes in feature space. It aims to maximize the margin between the classes while minimizing classification errors.

3. **Decision Trees**: Decision trees partition the feature space into regions and assign a class label to each region. Each internal node of the tree represents a decision based on the feature values, and each leaf node represents a class label.

4. **Random Forests**: Random forests are an ensemble learning method that combines multiple decision trees to improve classification accuracy. Each tree in the forest is trained on a random subset of the training data and features, and the final prediction is made by aggregating the predictions of all trees.

Evaluation of binary classification models is typically done using various metrics, including accuracy, precision, recall, F1 score, ROC curve, and AUC-ROC. These metrics provide insights into different aspects of the model's performance, such as its ability to correctly classify positive instances (sensitivity) and negative instances (specificity), as well as its overall predictive accuracy.

Binary classification is a fundamental task in machine learning and has applications in various domains, including healthcare, finance, marketing, and natural language processing.
"""
