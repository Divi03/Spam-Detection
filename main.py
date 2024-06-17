import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load the vectorizer and model
tfidf = pickle.load(open('models/model.pkl', 'rb'))
model = pickle.load(open('models/model.pkl', 'rb'))

# Get input SMS
print("Enter SMS: ")
input_sms = input()

# 1. Preprocess the input SMS
transformed_sms = transform_text(input_sms)

# 2. Vectorize the transformed SMS
vector_input = tfidf.transform([transformed_sms])

# 3. Predict using the model
result = model.predict(vector_input)[0]

# 4. Display the result in a box
if result == 1:
    print(" ________")
    print("|  Spam  |")
    print(" ‾‾‾‾‾‾‾‾")
else:
    print(" ___________")
    print("| Not Spam  |")
    print(" ‾‾‾‾‾‾‾‾‾‾‾")
