from flask import Flask, render_template, request
import joblib
import pandas as pd
import string, re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
factory = StemmerFactory()
stemmer = factory.create_stemmer()
import warnings
warnings.filterwarnings(action='ignore')

app = Flask(__name__)
model = joblib.load("modelsvm.pkl")
listkata = joblib.load("listkata.pkl")
stopword_dict = pd.read_csv('dict_stopword.csv', header =None)
stopword_dict = stopword_dict.rename(columns={0: 'stopword'})
slangword = pd.read_csv('slangword.csv', sep=';')

def cleansing(text):
    remove = string.punctuation
    translator = str.maketrans(remove, ' '*len(remove))
    text = text.translate(translator)
    text = text.encode('ascii', 'ignore').decode('utf-8')
    text = re.sub(r'[^\x00-\x7f]',r'', text)
    text = re.sub(r'[0-9]+', '', text)   
    text = text.replace('\n', ' ')
    return text

slangword_map = dict(zip(slangword['slang'], slangword['formal']))
def normalize(text):
   return ' '.join([slangword_map[word] if word in slangword_map else word for word in text.split(' ')])

def remove_stopword(text):
    text = ' '.join([' ' if word in stopword_dict.stopword.values else word for word in text.split(' ')])
    text =  re.sub(' +',' ',text) #remove extra space
    text = text.strip()
    return text

def stemming(text):
    return stemmer.stem(text)

def preprocess(text):
    text = text.lower()
    text = cleansing(text)
    text = normalize(text)
    text = remove_stopword(text)
    text = stemming(text)
    return text

@app.route('/')
def home():
    return render_template('predict.html')

@app.route('/predict', methods=["POST"])    
def predict():
    new_review = request.form["Review"]
    text = preprocess(new_review)
    Train_X_Tfidf = listkata.transform([text])
    count_vect_df = pd.DataFrame(Train_X_Tfidf.todense(), columns=listkata.get_feature_names())
    prediction = model.predict(count_vect_df)[0]  
    if prediction == "positif":
        return render_template('predict.html', prediction_text='Positive')
    else:
        return render_template('predict.html', prediction_text='Negative')

if __name__ == "__main__":
    app.run(debug=True)
