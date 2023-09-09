# Sentiment Analysis on IMDb Movie Reviews

This repository contains the code and resources for performing sentiment analysis on IMDb movie reviews using a classification model implemented with TensorFlow and Keras. The dataset consists of 25,000 training reviews and 25,000 testing reviews, making it a binary classification task to predict whether a review is positive or negative.

## Dataset

The dataset used for this sentiment analysis task is the IMDb movie reviews dataset. It contains 50,000 movie reviews, with 25,000 in the training set and 25,000 in the testing set. Each review is labeled as either positive or negative, making it suitable for a binary classification task.

## Preprocessing

Before training the classification model, the following preprocessing steps were applied to the text data:

1. **Text Cleaning**: Removal of HTML tags, special characters, and punctuation.
2. **Tokenization**: Splitting the text into individual words or tokens.
3. **Text Lowercasing**: Converting all text to lowercase to ensure uniformity.
4. **Stopword Removal**: Eliminating common stop words that don't carry meaningful sentiment.
5. **Padding and Sequence Truncation**: Ensuring that all sequences have the same length by either padding or truncating.

## Model Architecture

The sentiment classification model is built using TensorFlow and Keras. It typically consists of the following layers:

1. **Embedding Layer**: Converts words into numerical embeddings.
2. **LSTM (Long Short-Term Memory) or GRU (Gated Recurrent Unit) Layers**: These recurrent layers capture sequential dependencies in the text.
3. **Dense Layers**: One or more dense layers for classification.
4. **Output Layer**: A single output neuron with a sigmoid activation function for binary classification.

## Training

The model is trained on the training dataset using the binary cross-entropy loss function and the Adam optimizer. The training process involves iterating over the training data for a fixed number of epochs while updating the model weights to minimize the loss.

## Evaluation Metrics

To assess the performance of the sentiment analysis model, the following evaluation metrics are calculated on the test dataset:

1. **Confusion Matrix**: A table showing true positive, true negative, false positive, and false negative values.
2. **Accuracy**: The ratio of correctly predicted samples to the total number of samples.
3. **Precision**: The ratio of true positives to the total predicted positives.
4. **Recall**: The ratio of true positives to the total actual positives.
5. **F1-Score**: The harmonic mean of precision and recall, providing a balanced measure of model performance.

## Usage

1. Clone this repository to your local machine.
2. Ensure you have the necessary libraries and dependencies installed.
3. Run the provided Jupyter Notebook or Python script to train and evaluate the sentiment analysis model.
4. Explore the evaluation metrics and model predictions on the test dataset.

## Conclusion

Sentiment analysis on IMDb movie reviews is a common NLP task, and this repository provides a framework for building and evaluating a classification model using TensorFlow and Keras. By preprocessing the data, training the model, and calculating key evaluation metrics, you can gain insights into the model's performance in classifying movie reviews as positive or negative. Feel free to experiment with different model architectures and hyperparameters to improve the results.
