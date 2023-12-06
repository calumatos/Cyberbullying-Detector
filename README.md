# Final_project DA_OCT23
## Detect Cyber bullying

#### Members of the group:
Carmen Matos, 
Juliane Petersen

### Problem Definition:
The phenomenon of cyberbullying has been growing at alarming rates. 65% of global adults say their kids or kids in their community have been cyberbullied over social media. 45% say their child was bullied through text or messaging apps (Source: https://explodingtopics.com/blog/cyberbullying-stats). Technology can be used to help identify cyberbullying content.

#### Goal of the project:
This project will use Machine Learning methods to detect whether or not a social media message contains cyberbullying content. In addition, we will identify general risk factors for bullying.  
#### Brief description of the dataset:

Dataset 1: 
This dataset is a collection of datasets from different sources related to the automatic detection of cyber-bullying.
The data is from different social media platforms like Kaggle, Twitter, Wikipedia Talk pages and YouTube. The data 
contain text and labeled as bullying or not. The data contains different types of cyber-bullying like hate speech, 
aggression, insults and toxicity.
Source: Kaggle (https://www.kaggle.com/datasets/saurabhshahane/cyberbullying-dataset?select=aggression_parsed_dataset.csv)
Number of rows: 159686 rows
Number of features: 5 columns

Dataset 2: 
This dataset includes various indicators of bullying at school, such as feelings of loneliness, lack of close friends, poor communication with parents, absence from classes. 
Source: Kaggle (https://www.kaggle.com/datasets/leomartinelli/bullying-in-schools/)
Number of rows: 56981 rows
Number of features: 18 columns


#### Project Plan:

# Steps for dataset 1
1. Text Preprocessing:
Cleaning and preprocessing the text data. This involves tasks like removing stop words, handling emojis, and converting text to lowercase.
Tokenizing the text to convert it into a format suitable for machine learning algorithms.

2. Feature Extraction:
Extracting relevant features from the text data. This might include bag-of-words representations, 
TF-IDF (Term Frequency-Inverse Document Frequency), or word embeddings.
Exploring advanced techniques like word embeddings (Word2Vec, GloVe) to capture semantic relationships between words.

3. Model Selection:
Choosing a machine learning model that suits your problem. Common models for text classification include Naive Bayes, 
Support Vector Machines (SVM), and deep learning models like recurrent neural networks (RNNs) or transformers (e.g., BERT).
Considering ensemble methods or fine-tuning pre-trained models for improved performance.

4. Model Training and Evaluation:
Splitting the dataset into training and testing sets to train and evaluate your model.
Identifying appropriate evaluation metrics such as precision, recall, F1-score, and accuracy to assess the model's performance.

5. Addressing Class Imbalance:
Our reserach showed that cyberbullying datasets often suffer from class imbalance, where non-cyberbullying instances significantly outnumber 
cyberbullying instances. Identifying the best way to address these imbalances using techniques like oversampling, undersampling, or using more advanced 
methods like Synthetic Minority Over-sampling Technique (SMOTE).

6. Hyperparameter Tuning:
Fine-tuning the hyperparameters of our model to optimize its performance. Using grid search or random search as useful techniques for finding optimal hyperparameter values.

7. Deployment:
Once satisfied with our model's performance, we aim to deploy it to a platform where it can be used for real-time or batch processing of messages and comments.


# Steps for dataset 2

1. Data Wrangling
2. EDA
3. Drawing conclusions


# Time line

Wed: Data wrangling and Elevator pitch

Thu: Finalise data wrangling and EDA for dataset 1

Fri: Text preprocessing for dataset 2

Mo: Feature extraction

Tue: Model selection, training and evaluation

Wed: Improving the model: Addressing imbalances and finetuning hyperparameters

Thu: Preparing the presentation

Fri: Presentation


