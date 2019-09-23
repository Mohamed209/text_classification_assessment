#### Text Classifier API

this repo may be considered as text classification API

it describes text classification pipeline from text preprocessing until model evaluation
##### folder description
```
src >>> contain all source code / class definitions fro text preprocessing model training and evaluation
model_weights >>> contain weights of trained models and vectorizer objects
data >>>
├── model_predictions
│   └── predictions.txt # taskA results
├── processed # processed data ready for training
│   ├── train.csv
│   ├── x_testrandom_forest.txt
│   └── y_testrandom_forest.txt
└── raw # received non processed data
    ├── test_data.txt
    └── train_set.txt
taskA.ipynb and taskB.ipynb >>> driver code for training and evaluating models 
for task A and B

``` 
##### hyparameters tuning
all models parameters fitted in this repo are found using grid search technique
we will discuss them in detail for every task
###### Task A
* text preprocessing pipeline :

    1. remove punctuation , special chars , extra spaces and numbers
    
    2. augment text to increase data set variance
    
    3. text stemming
* model training 
    1. TF-IDF vectorizer used to represent text where best parameters
    found by grid search are :
    
    ```
    ngram_range=(1, 2)
    max_features=3000
  keeping other parameters to default
  all stop words are removed / excluded from training
    ```
  2. random forest classifier used to train the model
  ```
  n_estimators=200
  ```
 * model evaluation 
 check accuracy , confusion matrix and classification report for trained classifiers

    also find predections from "test_data.txt" in
    ```
   data --> model_predections --> predictions.txt
   ``` 
##### Note
in taskA , a text augmentation technique was used as original dataset 
contains only about 1750 training examples , after text augmentation i got 7500 training example 
this trick increases the model accuracy with about 14%

###### Task B
I used a LSTM neural network with hyparameters tuned using also gridsearch
and with architecture as 
```
Embedding (500,32) >> LSTM (100) >> Dense (1)
```
embedding is an advanced technique for word representations , as it makes the model knows words meaning and similarity between them 
so it is clear that why neural networks outperform classical machine learning algorithms 
