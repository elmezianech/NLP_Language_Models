# NLP_Language_Models

This repository contains a collection of NLP experiments conducted using PyTorch library. The project explores various techniques such as regression, text generation, and BERT embeddings.
The project is divided into three main parts.

## Parts Overview:

### Part 1: Regression
This project aims to perform regression on Arabic text data collected from various websites. The goal is to develop models that can effectively predict their relevance scores.
- Dataset

The dataset consists of Arabic proverbs and sayings scraped from this website : https://arabpoems.com/حكم-وأمثال/ using Scrapy. Each text is assigned a relevance score between 0 and 10, representing its importance or significance.

- Preprocessing
1. Text Cleaning: Special characters, diacritics, and stop words are removed from the text data.
2. Tokenization and Lemmatization: The text is tokenized into words, and then lemmatized to normalize the words to their base form.

- Models Used
1.  Recurrent Neural Network (RNN): A basic RNN architecture is trained on the preprocessed text data. R2 score, Mean Squared Error (MSE), and Mean Absolute Error (MAE) are used to evaluate the model's performance.

2.  Long Short-Term Memory (LSTM): An LSTM architecture is utilized to capture long-term dependencies in the text data. Performance metrics similar to the RNN model are used for evaluation.

3.  Bidirectional GRU: A bidirectional GRU model is employed to capture information from both past and future contexts. Evaluation metrics are used to assess the model's effectiveness.
  
### Part 2: Transformer (Text generation)
### Part 3: BERT

