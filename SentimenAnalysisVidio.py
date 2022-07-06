import pandas as pd
import string, re
import csv
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# text = pd.read_csv('data.csv', usecols=['at','score','content'])
# text.head()

stopword_dict = pd.read_csv('dict_stopword.csv', header =None)
stopword_dict = stopword_dict.rename(columns={0: 'stopword'})
stopword_dict.head()

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def lowercase(text):
    return text.lower()

def cleansing(text):
    remove = string.punctuation
    translator = str.maketrans(remove, ' '*len(remove))
    text = text.translate(translator)
    text = text.encode('ascii', 'ignore').decode('utf-8')
    text = re.sub(r'[^\x00-\x7f]',r'', text)
    text = re.sub(r'[0-9]+', '', text)   
    text = text.replace('\n', ' ')
    return text

def remove_stopword(text):
    text = ' '.join([' ' if word in stopword_dict.stopword.values else word for word in text.split(' ')])
    text =  re.sub(' +',' ',text) #remove extra space
    text = text.strip()
    return text

def stemming(text):
    return stemmer.stem(text)

def preprocess(text):
    text = lowercase(text)
    text = cleansing(text)
    text = remove_stopword(text)
    text = stemming(text)
    return text

text['content_clear'] = text['content'].apply(preprocess)
text.head()

def tokenize(prepro):
    tokens = re.split('\W+', prepro) #W+ means that either a word character (A-Za-z0-9_) or a dash (-) can go there.
    return tokens

text['content_token'] = text['content_clear'].apply(lambda x: tokenize(x.lower())) 
text.head()

lexicon_positive = dict()
with open('Kamus Negative.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        lexicon_positive[row[0]] = int(row[1])

lexicon_negative = dict()
with open('Kamus Positive.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        lexicon_negative[row[0]] = int(row[1])
        
# Function to determine sentiment polarity of tweets        
def sentiment_analysis_lexicon_indonesia(text):
    #for word in text:
    score = 0
    for word in text:
        if (word in lexicon_positive):
            score = score + lexicon_positive[word]
    for word in text:
        if (word in lexicon_negative):
            score = score + lexicon_negative[word]
    polarity=''
    if (score > 0):
        polarity = 'positif'
    elif (score < 0):
        polarity = 'negatif'
    else:
        polarity = 'netral'
    return score, polarity

results = text['content_token'].apply(sentiment_analysis_lexicon_indonesia)
results = list(zip(*results))
text['polarity_score'] = results[0]
text['polarity'] = results[1]
print(text['polarity'].value_counts())

text = text[text['polarity'] != 'netral']
print(text['polarity'].value_counts())

X_train, X_test, y_train, y_test = model_selection.train_test_split(text['content_clear'],text['polarity'], test_size=0.2)
Encoder = LabelEncoder()
y_train =  Encoder.fit_transform(y_train)
y_test = Encoder.fit_transform(y_test)

#TF
Tf_vect = CountVectorizer()
Tf_vect.fit(text["content_clear"])

Train_X_Tf = Tf_vect.transform(X_train)
Test_X_Tf =  Tf_vect.transform(X_test)
Train_X_Tf.shape

#TF-IDF
Tfidf_vect = TfidfVectorizer()
Tfidf_vect.fit(text["content_clear"])

Train_X_Tfidf = Tfidf_vect.transform(X_train)
Test_X_Tfidf =  Tfidf_vect.transform(X_test)
Train_X_Tfidf.shape

SVM = SVC()
SVM.fit(Train_X_Tfidf,y_train)

prediction_SVM = SVM.predict(Test_X_Tfidf)
print("SVM Accuracy Score:", round(accuracy_score(prediction_SVM, y_test)*100,2))
print("SVM Recall Score:", round(recall_score(prediction_SVM, y_test)*100,2))
print("SVM Precision Score:", round(precision_score(prediction_SVM, y_test)*100,2))
print("SVM F1 Score:", round(f1_score(prediction_SVM, y_test)*100,2))

hyperparameters = {'kernel': ('linear', 'rbf', 'poly', 'sigmoid')}
svm=SVC()
svm_tuned = GridSearchCV(svm, hyperparameters, cv=5)
svm_tuned.fit(Train_X_Tfidf, y_train)

print("Best C:", svm_tuned.best_estimator_.C)
print("Best Kernel:", svm_tuned.best_estimator_.kernel)
print("Best Score:", svm_tuned.best_score_)

svm = SVC(C=1, kernel='linear')
svm.fit(Train_X_Tfidf,y_train)

predict_test = svm.predict(Test_X_Tfidf)
print("SVM Accuracy Score:", round(accuracy_score(predict_test, y_test)*100,2),"%")
print("SVM Recall Score:", round(recall_score(predict_test, y_test)*100,2),"%")
print("SVM Precision Score:", round(precision_score(predict_test, y_test)*100,2),"%")
print("SVM F1 Score:", round(f1_score(predict_test, y_test)*100,2),"%")
print(f'\nConfusion Matrix:\n{confusion_matrix(predict_test,y_test)}')


count_vect_df = pd.DataFrame(Train_X_Tfidf.todense(), columns=Tfidf_vect.get_feature_names())
listKata = pd.DataFrame({"Kata" : count_vect_df.columns, "Frekuensi" : 0})

import joblib
joblib.dump(svm.fit(Train_X_Tfidf,y_train),'modelsvm.pkl')
joblib.dump(Tfidf_vect,'listkata.pkl')