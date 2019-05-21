#################################### Import Data ######################################
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

data = pd.read_csv('TextClassification/MainFunction/booklisting.csv', encoding='ISO-8859-1')

columns = ['id', 'image', 'image_link', 'books', 'author', 'bclass', 'genre']
data.columns = columns

books = pd.DataFrame(data['books'])
author = pd.DataFrame(data['author'])
genre = pd.DataFrame(data['genre'])
bclass = pd.DataFrame(data['bclass'])

################################ Data Preprocessing ##################################
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Clean stopwords function
def clean_stopwords(syn):
    stop_words = set(stopwords.words('english'))
    clean = []
    for sent in syn:
        new = []
        token = word_tokenize(sent)
        for word in token:
            if word not in stop_words:
                new.append(word)
        clean.append(' '.join(new))

    return (clean)

import re

# Get books description
book = data.books

# Get books categorization
category = data.genre

# Create empty array to store processed description
syn = []

# Data preprocessing for description
for sen in range(0, len(book)):
    #     Remove special characters and punctuation
    document = re.sub(r'\W', ' ', str(book[sen]))

    #     Set description into lower case
    document = document.lower()

    #     remove single character
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

    #     remove double space
    document = re.sub(r'\s+', ' ', document, flags=re.I)

    #     save processed description into array
    syn.append(document)

# clean stopwords
clean = clean_stopwords(syn)

############################### Convert Text into Vector ######################################
# REMOVE ENGLISH STOPWORDS
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(clean)

# SAVE VECTOR INTO MODEL
pickle.dump(count_vect.vocabulary_, open("count_vector.pkl", "wb"))

######################################### Term Frequency and Invers Document Frequency (TF-IDF) ##############################################
from sklearn.feature_extraction.text import TfidfTransformer

# TRANSFORM WORD VECTOR TO TF IDF
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# SAVE TF-IDF
pickle.dump(tfidf_transformer, open("tfidf.pkl", "wb"))

############################ Split Data into Train and Test Set ####################################
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_train_tfidf, category, test_size=0.2, random_state=0)

########################## Training Classification Model with Naive Bayes Algorithm ###################################
from sklearn.naive_bayes import MultinomialNB

# Training model with Naive Bayes Algorithm
classifierNB = MultinomialNB().fit(X_train, y_train)
classifierNB.fit(X_train, y_train)

# save model
pickle.dump(classifierNB, open("nb.pkl", "wb"))

############################## Predicting using Naive Bayes Model #################################################
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


def main(input_text):
    docs_new = input_text
    docs_new = [docs_new]

    # Load model
    loaded_vec = CountVectorizer(vocabulary=pickle.load(open("Test/count_vector.pkl", "rb")))
    loaded_tfidf = pickle.load(open("Test/tfidf.pkl", "rb"))
    loaded_model = pickle.load(open("Test/nb.pkl", "rb"))
    X_new_counts = loaded_vec.transform(docs_new)
    X_new_tfidf = loaded_tfidf.transform(X_new_counts)
    # Predict
    predicted = loaded_model.predict(X_new_tfidf)
    return predicted[0]


########################## Training Classification Model with Supervised Vector Machine Algorithm ###################################
from sklearn import svm

# Training model with SVM algorithm
classifierSVM = svm.LinearSVC()
classifierSVM.fit(X_train, y_train)

# Save model
pickle.dump(classifierSVM, open("svm.pkl", "wb"))

############################## Predicting using Supervised Vector Machine Algorithm #################################################
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

def svmFunction(input_text2):
    docs_new = input_text2
    docs_new = [docs_new]

    # Load model
    loaded_vec2 = CountVectorizer(vocabulary=pickle.load(open("Test/count_vector.pkl", "rb")))
    loaded_tfidf2 = pickle.load(open("Test/tfidf.pkl","rb"))
    loaded_model2 = pickle.load(open("Test/svm.pkl","rb"))

    X_new_counts2 = loaded_vec2.transform(docs_new)
    X_new_tfidf2 = loaded_tfidf2.transform(X_new_counts2)

    # Predict
    predicted2 = loaded_model2.predict(X_new_tfidf2)

    return (predicted2[0])