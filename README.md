# Cyberbullying Detector
## Final_project DA_OCT23



#### Members of the group:
Carmen Matos, 
Juliane Petersen

#### Problem Definition:
The phenomenon of cyberbullying has been growing at alarming rates. 65% of global adults say their kids or kids in their community have been cyberbullied over social media. 45% say their child was bullied through text or messaging apps.  

(Source: https://explodingtopics.com/blog/cyberbullying-stats).

#### Goal of the project:
The primary goal is to develop a machine learning model that can accurately identify whether a given social media message contains cyberbullying content. In addition, the project aims to identify  common risk factors associated with bullying and to improve our understanding of the broader context. 

[View the final presentation](https://github.com/calumatos/Cyberbullying-Detector/blob/main/Detect_Cyberbullying%2C%20Final_presentation.pdf)

#### Brief description of the dataset:

Dataset 1: 
This dataset includes various indicators of bullying at school, such as feelings of loneliness, lack of close friends, poor communication with parents, absence from classes. 

Source: Kaggle (https://www.kaggle.com/datasets/leomartinelli/bullying-in-schools/)

Number of rows: 56981 rows

Number of features: 18 columns


Dataset 2: 
This dataset is a collection of datasets from different sources related to the automatic detection of cyber-bullying.
The data is from different social media platforms like Kaggle, Twitter, Wikipedia Talk pages and YouTube. The data 
contain text and labeled as bullying or not. The data contains different types of cyber-bullying like hate speech, 
aggression, insults and toxicity.

Source: Kaggle (https://www.kaggle.com/datasets/saurabhshahane/cyberbullying-dataset?select=aggression_parsed_dataset.csv)

Number of rows: 159686 rows

Number of features: 5 columns


## Project Plan:

### Steps for dataset 1
1. Data Wrangling
2. EDA
3. Drawing conclusions

### Steps for dataset 2

1. Text Preprocessing:
Cleaning and preprocessing the text data. This involves tasks like removing stop words, handling emojis, and converting text to lowercase.
Tokenizing the text to convert it into a format suitable for machine learning algorithms. 

2. Feature Extraction:
Extracting relevant features from the text data. Exploring advanced techniques like word embeddings (Word2Vec) to capture semantic relationships between words. This step improves the model's understanding of the textual content. 

3. Model Selection:
The project uses the Sequential Model with the TensorFlow Keras framework, a popular choice for text classification and NLP (natural language processing) tasks. This selection is based on the model's ability to effectively process sequential data. 

4. Model Training and Evaluation:
The dataset is divided into training and testing sets to train and evaluate the model. Evaluation metrics such as precision, recall, and accuracy are employed to assess the model's performance and ensure its effectiveness in identifying cyberbullying content. 

5. Test Interface:
Use a Streamlit interface to display the results.

 <div>
    <img src="https://github.com/calumatos/Cyberbullying-Detector/blob/main/Cyberbullying_Interface.png">         
  </div>

By combining technological advances in machine learning with a comprehensive approach to data analysis, this project aims to contribute to the prevention and identification of cyberbullying and to  make online spaces safer for users, in particular for the younger generation.

 

