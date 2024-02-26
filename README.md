# Japanese-newspapers-classification 📰 🗞️
## This repository contains a Natural Language Processing project to classify articles from multiple Japanese newspapers 🈯️ 🈂️

## Dataset 📊
There are two datasets, one containing approximately 312,000 records with text in Japanese, the other containing approximately 36,000 records with text in English, but always referring to Japanese newspapers.
The columns are the same for both datasets and are as follows:
- Text: Newspaper article in text format, in Japanese and English
- Date: Date of publication of the article
- Author: Author of the article
- Title: Title of the article
- Source: Newspaper that published the article

## Process of Analysis ⚙️
After loading the dataset, the steps were as follows:
- **Data preprocessing** 👀

Check the number of records, number of columns, number of classes and number of records for each class.
Next, check for null values for each column and check for duplicate records.
- **Exploratory Data Analysis** 🔍

After eliminating duplicate records and those with null values in the columns of interest (text and source), an exploratory analysis process was conducted using graphs to understand the distribution of the classes. Additionally, graphs were printed for both datasets to see the number of null values in the author and title columns. Finally, two other graphs show the average length of articles for each newspaper class.
- **Data preparation** 📈
  
The data_cleaning function allows you to clean the text, both for Japanese and English, done separately for linguistic reasons. In detail, the stop words were eliminated, the text was standardized in lowercase, the sentences were split into single words and lemmatization was applied to bring the verbs back into basic form.
Subsequently, the datasets were concatenated maintaining only two columns, text and source.

### Rebalancing ⚖️
The complete, clean and merged dataset has class balancing issues.
To solve this problem, several techniques were tested, oversampling, undersampling, weights per class and a mix of oversampling and undersampling.
Finally, we opted for an undersampling of the majority classes, i.e. all those who had a number of records higher than the average number of class records, which is 15,000.
The undersampling was carried out taking into account the sentences that contained the most frequent characters, considered most representative of the class itself.
Subsequently, to balance the dataset, oversampling was done using the SMOTE technique, creating synthetic data whose characteristics are similar to the real ones, but not the same.
In order to apply SMOTE, it was first necessary to transform the text into numerical vectors and there were two techniques used:
- **TF-IDF**: Term frequency- Inverse Document Frequency, a famous technique capable of assigning weights to frequent words and rare words.
- **BERT**: Bidirectional Encoder Representation from Transformers, a more advanced technique based on Transformers technology capable of better learning semantic relationships in data in textual format.

### Model Creation ⚒️
After processing the data in textual format and transforming it into numerical vectors, the dataset is ready to be used in Machine Learning and Deep Learning models to create a classifier.
Regarding Machine Learning models, the following were tested:
- **Logistic Regression**
- **Multi Layer Perceptrons**
- **Random Forest**

The approach was as follows:
- Subdivision of data into train and test sets
- Model training on the train set
- Predictions on the test set
- Classification report with accuracy, precision, recall and F1-Score metrics.

In summary, the results are different from each other:
- Logistic Regression: 0.58 accuracy in the train set and 0.41 in the test set, the model that performs worst of all.
- MLP: 0.72 accuracy in the train set and 0.45 in the test set, using 500 neurons and a single hidden layer
- Random Forest: 0.98 accuracy in the train but 0.40 in the test set, therefore showing overfitting problems.

Some suggestions 💡

It is a good idea to do hyperparameter tuning to understand which is the best set of parameters to use, in particular:
- MLP: test with at least two layers and fewer neurons per layer. Test different activation functions.
- Random Forest: test a random search by trying different configurations, such as the number of trees and the depth of the trees.
Use cross-validation to compare results across different folds and prevent overfitting.


Regarding deep learning models, the following were tested:
- **GRU**: Special Recurrent Neural Network that uses memory gates to store the most important information
- **LSTM** : Long-Short-Term-Memory, type of neural network equipped with long-term memory

In summary, the results are good and better than the machine learning approach, in particular:
- GRU reaches 0.53 accuracy in the train set, 0.57 in the validation set and 0.57 in the test set, with 6 epochs, 3 layers, 64 neurons for the first two layers and 23 in the last layer.
-LSTM reaches 0.70 accuracy in the train set, 0.60 in the validation set and 0.60 in the test set, with 10 epochs, 3 layers, 128 neurons for the first two layers and 23 in the last layer.

Some suggestions 💡

It is a good idea to do hyperparameter tuning to understand which is the best set of parameters to use, in particular:
- GRU: test with layers of 128 neurons and at least 10 training epochs
- LSTM: Try adding a further layer of 64 neurons, trying to hyperparameter tune the other parameters such as dropout and activation function.

