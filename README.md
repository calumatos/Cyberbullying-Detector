# Final_project DA_OCT23
Detect Cyber bullying

#### Members of the group:
Carmen Matos, 
Juliane Petersen

#### Goal of the project:

Our main objective is to develop a machine learning project to address cyberbullying based on 
internet messages, tweets, and social media comments. 

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
This dataset includes various indicators of bullying such as feelings of loneliness, 
lack of close friends, poor communication with parents, absence from classes. 
Source: Kaggle (https://www.kaggle.com/datasets/leomartinelli/bullying-in-schools/)
Number of rows: 56981 rows
Number of features: 18 columns



#### Project Plan:

##### Problem Definition:
Clearly define what constitutes cyberbullying in the context of your project. This might include offensive language, 
personal attacks, or any form of online harassment. Consider the different forms of cyberbullying, such as direct 
attacks, impersonation, or the spread of harmful content.

##### Data Collection:
Gather a diverse and representative dataset of internet messages, tweets, and social media comments that cover various platforms.
Annotate the dataset with labels indicating whether each instance contains cyberbullying or not. This may involve manual 
labeling or leveraging pre-existing labeled datasets.

##### Text Preprocessing:
Clean and preprocess the text data. This involves tasks like removing stop words, handling emojis, and converting text to lowercase.
Tokenize the text to convert it into a format suitable for machine learning algorithms.

##### Feature Extraction:
Extract relevant features from the text data. This might include bag-of-words representations, 
TF-IDF (Term Frequency-Inverse Document Frequency), or word embeddings.
Explore advanced techniques like word embeddings (Word2Vec, GloVe) to capture semantic relationships between words.

##### Model Selection:
Choose a machine learning model that suits your problem. Common models for text classification include Naive Bayes, 
Support Vector Machines (SVM), and deep learning models like recurrent neural networks (RNNs) or transformers (e.g., BERT).
Consider ensemble methods or fine-tuning pre-trained models for improved performance.

##### Model Training and Evaluation:
Split your dataset into training and testing sets to train and evaluate your model.
Use appropriate evaluation metrics such as precision, recall, F1-score, and accuracy to assess the model's performance.

##### Addressing Class Imbalance:
Cyberbullying datasets often suffer from class imbalance, where non-cyberbullying instances significantly outnumber 
cyberbullying instances. Address this imbalance using techniques like oversampling, undersampling, or using more advanced 
methods like Synthetic Minority Over-sampling Technique (SMOTE).

##### Hyperparameter Tuning:
Fine-tune the hyperparameters of your model to optimize its performance. Grid search or random search can be useful 
techniques for finding optimal hyperparameter values.

### Deployment:
Once satisfied with your model's performance, deploy it to a platform where it can be used for real-time or batch 
processing of messages and comments.

