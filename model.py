import re
import torch
import torch.nn as nn
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.linear(x)
        out = self.sigmoid(out)
        return out

def predict_email(email_text, model, vectorizer):
    processed_text = re.sub(r'[^a-zA-Z\s]', '', email_text.lower())
    tokens = word_tokenize(processed_text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    processed_text = ' '.join(tokens)
    features = vectorizer.transform([processed_text]).toarray()
    features = torch.FloatTensor(features)
    model.eval()
    with torch.no_grad():
        output = model(features)
        prediction = 'Spam' if output.item() >= 0.5 else 'Ham'
    return prediction, output.item()

def load_and_infer(email_text):
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    input_dim = vectorizer.max_features
    model = LogisticRegression(input_dim)
    model.load_state_dict(torch.load('spam_classifier_model.pth', weights_only=True))
    model.eval()

    prediction, probability = predict_email(email_text, model, vectorizer)
    return prediction, probability

# sample_email = "Win a free iPhone now! Click here to claim your prize!"
# prediction, probability = load_and_infer(sample_email)
# print(f'\nSample email: "{sample_email}"')
# print(f'Prediction: {prediction}')
# print(f'Spam Probability: {probability:.4f}')