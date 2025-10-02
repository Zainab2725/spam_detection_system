import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

ps = PorterStemmer()
stopwords = set(stopwords.words('english'))

def transform(text):
  text = text.lower()
  text = nltk.word_tokenize(text)
  output = []

  for word in text:
    if word.isalnum():
       output.append(word)

  text = output[:]
  output.clear()

  for word in text:
    if word not in nltk.corpus.stopwords.words('english') and word not in string.punctuation:
      output.append(word)

  text = output[:]
  output.clear()

  for word in text:
     output.append(ps.stem(word))

  return " ".join(output)


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))


st.title("Email/SMS Spam Classifier")
input_mail = st.text_area("Enter the email content")

if "history" not in st.session_state:
    st.session_state.history = []


if st.button("Predict"):
  transformed_mail = transform(input_mail)
  vector_input = tfidf.transform([transformed_mail])
  result = model.predict(vector_input)[0]
  prob = model.predict_proba(vector_input)[0]
  if result == 1:
        st.error(f"🚨 Spam (confidence: {prob[1]:.2f})")
  else:
        st.success(f"✅ Not Spam (confidence: {prob[0]:.2f})")

    # Save history
  st.session_state.history.append(
        {"Email": input_mail, "Prediction": "Spam" if result == 1 else "Not Spam", "Confidence": max(prob)}
    )

st.subheader("📂 Upload file for batch classification")
uploaded_file = st.file_uploader("Upload a .txt or .csv file", type=["txt", "csv"])

if uploaded_file:
    if uploaded_file.name.endswith(".txt"):
        emails = uploaded_file.read().decode("utf-8").split("\n")
        df = pd.DataFrame(emails, columns=["Message"])
    else:
        df = pd.read_csv(uploaded_file, encoding='latin1')
        df = df.rename(columns={"v1": "Label", "v2": "Message"})

    df["Transformed"] = df["Message"].apply(transform)
    vectors = tfidf.transform(df["Transformed"])

    df["Prediction"] = model.predict(vectors)
    df["Confidence"] = model.predict_proba(vectors).max(axis=1)

    df["Prediction"] = df["Prediction"].map({0: "Not Spam", 1: "Spam"})

    st.dataframe(df[["Message", "Prediction", "Confidence"]])

    # Download results
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Results as CSV", csv, "classified_emails.csv", "text/csv")

# ---- Prediction history ----
st.subheader("📜 Prediction History")
if len(st.session_state.history) > 0:
    hist_df = pd.DataFrame(st.session_state.history)
    st.dataframe(hist_df)
else:
    st.write("No predictions yet.")