# Japanese-newspapers-classification 📰 🗞️
## This repository contains a Natural Language Processing project to classify articles from multiple Japanese newspapers 🈯️ 🈂️

## Dataset 📊
There are two datasets, one containing approximately 312,000 records with text in Japanese, the other containing approximately 36,000 records with text in English, but always referring to Japanese newspapers.
The columns are the same for both datasets and are as follows:
- Text: Newspaper article in text format, in Japanese and English
- Date: Date of publication of the article
- Author: Author of the article
- Title: Title of the article
- Source: Newspaper that published the article. There are 23 unique source.

you can find the two datasets used in the project in my google drive at the following [Link](https://drive.google.com/drive/folders/1polqOeG7XF0TiTZvdvFM_z-BDRhQARYT).
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
Regarding **Machine Learning** models, the following were tested:
- **Logistic Regression**
- **Multi Layer Perceptrons**
- **Random Forest**

The approach was as follows:
- Subdivision of data into train and test sets
- Model training on the train set
- Predictions on the test set
- Classification report with accuracy, precision, recall and F1-Score metrics
- Confusion matrix

In summary, the results are different from each other:
- **Logistic Regression**
  
Dataset | Accuracy
-|-
Train-set | 0.58     
Test-set | 0.41   
 

- **MLP**

Dataset | Accuracy
-|-
Train-set | 0.72     
Test-set | 0.45 

- **Random Forest**

Dataset | Accuracy
-|-
Train-set | 0.98     
Test-set | 0.40



Some suggestions 💡

It is a good idea to do hyperparameter tuning to understand which is the best set of parameters to use, in particular:
- MLP: test with at least two layers and fewer neurons per layer. Test different activation functions.
- Random Forest: test a random search by trying different configurations, such as the number of trees and the depth of the trees.
Use cross-validation to compare results across different folds and prevent overfitting.
- Test with machines that do not have RAM limitations and use more available data to grasp hidden relationships

Regarding **Deep Learning** models, the following were tested:
- **GRU**: Special Recurrent Neural Network that uses memory gates to store the most important information
  
- **LSTM**: Long-Short-Term-Memory, type of neural network equipped with long-term memory

- **BILSTM**: Bidirectional Long-Short-Term-Memory, is a type of neural network that uses two layers of LSTM to capture information on both sides

In summary, the results are good and better than the machine learning approach, in particular:
- **GRU**

Dataset | Accuracy
-|-
Train-set | 0.69    
Test-set | 0.58

-  Hyperparameters: 6 epochs, 3 layers, 128 neurons for the first layer, 64 neurons the second one and 23 in the last layer.
  
- **LSTM**

Dataset | Accuracy
-|-
Train-set | 0.60    
Test-set | 0.56

-   Hyperparameters: 10 epochs, 3 layers, 128 neurons for the first layer, 64 neurons the second one, 32 neurons for the third layer and 23 in the last layer.
  
- **BILSTM**

Dataset | Accuracy
-|-
Train-set | 0.70    
Test-set | 0.60

-   Hyperparameters: 10 epochs, 3 layers, 128 neurons for the first layer, 64 neurons the second one and 23 in the last layer.




Some suggestions 💡

It is a good idea to do hyperparameter tuning to understand which is the best set of parameters to use, in particular:
- GRU: testing with an additional layer, different dropout values and epochs, different activation functions, different numbers of neroni for each layers
- LSTM: Try adding a further layer of 64 neurons, trying to hyperparameter tune the other parameters such as dropout and activation function.


To conclude, the next steps to create more accurate classifiers could be the following:
- Transformers, neural network architectures that use the attention mechanism to identify the most important parts from the less important ones
- pretrained models such as GPT or LLAMA which have been found to be very efficient in processing natural language text
- Work in environments that allow you to make greater use of computing power such as RAM and GPU to speed up and streamline operations


With these latest implementations, accuracies in the test set of around 70-80% could be achieved.
