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
This part of the project focuses on text generation using the Transformer model, specifically the GPT-2 model. Text generation involves predicting the next word or sequence of words given an input prompt. In this case, we fine-tune the GPT-2 model on a dataset of quotes to generate new quotes.
- Dataset

For fine-tuning the GPT2 model, this Quotes dataset is used : https://www.kaggle.com/datasets/alihasnainch/quotes-dataset

- Model Configuration

1. Model: GPT2LMHeadModel, a pre-trained GPT-2 model, is utilized for text generation tasks.
2. Optimizer and Scheduler: AdamW optimizer with linear scheduler and warmup steps is employed for fine-tuning the model.

- Inference and Text Generation

1. Loading Pre-trained Model: Loads the fine-tuned model from the specified epoch for generating quotes.
2. Generation Process: Generates subsequent words starting with a prompt ("QUOTE:") until predicting an end-of-text token.
3. Output Handling: Decodes generated quotes from token IDs and saves to an output file.

### Part 3: BERT
This part focuses on implementing BERT-based model for sentiment analysis on Amazon Fashion reviews dataset 
- Dataset

For this Part, the Amazon Fashion dataset is used : https://nijianmo.github.io/amazon/index.html.

- Preprocessing
1.  Text preprocessing includes lowercase conversion, special character removal, and stopword elimination.

- Model Training
1.  BERT model (BertForSequenceClassification) is loaded.
2.  Data is split into training and testing sets.
3.  Custom dataset class (ReviewDataset) is created for tokenization and padding.
4.  Training is conducted using the Trainer class with defined parameters.

- Model Evaluation
1. Evaluation metrics include accuracy, F1 score, confusion matrix, BLEU score, and BERTScore.
2. Metrics are computed using standard libraries such as scikit-learn and NLTK.

### Conclusion
This project explores regression, text generation with GPT-2, and sentiment analysis using BERT embeddings. It covers Arabic text regression, quote generation, and sentiment analysis on Amazon Fashion reviews. Each part involves data preprocessing, model training, and evaluation.
