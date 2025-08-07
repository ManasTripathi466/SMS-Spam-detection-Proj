import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

### yha se uper vala function copy kiye h ##
# Data Preprocessing
def text_transform(x):
  x = x.lower()          #Lowercase
  x = nltk.word_tokenize(x)       #Tokenization

  y = []
  for i in x:         #Removing special characters
    if i.isalnum():
      y.append(i)

  x = y[:]
  y.clear()

  for i in x:
    if i not in stopwords.words('english') and i not in string.punctuation:       # Removing stop words and production
      y.append(i)

  x = y[:]
  y.clear()

  for i in x:
    y.append(ps.stem(i))       #Stemming


  return " ".join(y)

###yha tak uper vala function copy kiye!##


model = pickle.load(open('model.pkl','rb'))
vectorizer = pickle.load(open('vectorizer.pkl','rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter your message")


#Predict button for my project

if st.button("Predict"):

    transformed_sms = text_transform(input_sms)  # Preprocessing
    vector_input = vectorizer.transform([transformed_sms])  # Vectorize
    prediction = model.predict(vector_input)  # Predict

    if prediction[0] == 1:
        st.error("🔴 SPAM")
    else:
        st.success("🟢 NOT SPAM")


# slider with instructions
st.sidebar.markdown("## SMS")
st.sidebar.info("This is for spam classifier project created by manas tripathi.")
