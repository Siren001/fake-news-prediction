#importing the required libraries
from sklearn.utils import shuffle
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from nltk import tokenize
import seaborn as sns
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import sklearn.metrics as metrics
import numpy as np
from sklearn.metrics import plot_confusion_matrix


fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

#adding a flag to track fake and real

fake['target'] = 'fake'
true['target'] = 'true'

#concatenating the datasets

data = pd.concat([fake, true]).reset_index(drop = True)

#shuffling of data to avoid any bias

data = shuffle(data)
data = data.reset_index(drop=True)

#Data cleansing
#Removing the date
#Removing the title
#Convert the text to lowercase

data.drop(["date"],axis=1,inplace=True)
data.drop(["title"],axis=1,inplace=True)
data['text'] = data['text'].apply(lambda x: x.lower())
#Remove punctuation:

def punctuation_removal(text):
    all_list = [char for char in text if char not in string.punctuation]
    clean_str = ''.join(all_list)
    return clean_str
data['text'] = data['text'].apply(punctuation_removal)

#Remove stopwords:
stop = stopwords.words('english')
data['text'] = data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))


#getting to know our dataset
print("\nHead________________________________________________________\n")
print(data.head())
print("\nDesription________________________________________________________\n")
print(data.describe())

#vizualizing How many articles are there per subject?
print("\nNumber of articles per subject\n")
print(data.groupby(['subject'])['text'].count())
data.groupby(['subject'])['text'].count().plot(kind="bar")
plt.show()

#vizualizing the amount of fake and real articles
print("\nNumber of fake and real articles\n")
print(data.groupby(['target'])['text'].count())
data.groupby(['target'])['text'].count().plot(kind='bar')
plt.show()

# Function to vizualize the most frequently occuring news articles
token_space = tokenize.WhitespaceTokenizer()
def counter(text, column_text, quantity):
    all_words = ' '.join([text for text in text[column_text]])
    token_phrase = token_space.tokenize(all_words)
    frequency = nltk.FreqDist(token_phrase)
    df_frequency = pd.DataFrame({"Word": list(frequency.keys()),
                                   "Frequency": list(frequency.values())})
    df_frequency = df_frequency.nlargest(columns = "Frequency", n = quantity)
    plt.figure(figsize=(12,8))
    ax = sns.barplot(data = df_frequency, x = "Word", y = "Frequency", color = 'blue')
    ax.set(ylabel = "Count")
    plt.xticks(rotation='vertical')
    plt.show()
#Most frequent words in fake news

counter(data[data['target'] == 'fake'], 'text', 20)
#Most frequent words in real news

counter(data[data['target'] == 'true'], 'text', 20)

#Function to plot

from sklearn import metrics
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    if normalize==True:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
        thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')

#Splitting data
X_train,X_test,y_train,y_test = train_test_split(data['text'], data.target, test_size=0.2, random_state=42)

# Logistic regression:
# Vectorizing and applying TF-IDF
import time

start = time.time()
from sklearn.linear_model import LogisticRegression
pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', LogisticRegression())])
# Fitting the model
model = pipe.fit(X_train, y_train)
# Accuracy
prediction = model.predict(X_test)
print("accuracy for LR: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))
end = time.time()
print('Time Taken by LR:',end - start)

cm = metrics.confusion_matrix(y_test, prediction)
plot_confusion_matrix(cm, classes=['Fake', 'Real'])
plt.show()


#Decision Tree Classifier
import time

start = time.time()
from sklearn.tree import DecisionTreeClassifier
# Vectorizing and applying TF-IDF
pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', DecisionTreeClassifier(criterion= 'entropy',
                                           max_depth = 20,
                                           splitter='best',
                                           random_state=42))])
# Fitting the model
model = pipe.fit(X_train, y_train)
# Accuracy
prediction = model.predict(X_test)
print("accuracy for DTC: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))
end = time.time()
print('Time taken by DTC: ', end - start)

cm = metrics.confusion_matrix(y_test, prediction)
plot_confusion_matrix(cm, classes=['Fake', 'Real'])
plt.show()

#Random Forest Classifier
import time

start = time.time()

from sklearn.ensemble import RandomForestClassifier
pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', RandomForestClassifier(n_estimators=50, criterion="entropy"))])
model = pipe.fit(X_train, y_train)
prediction = model.predict(X_test)
print("accuracy for RFC: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))
end = time.time()
print('Time taken by RFC:', end - start)

cm = metrics.confusion_matrix(y_test, prediction)
plot_confusion_matrix(cm, classes=['Fake', 'Real'])
plt.show()
