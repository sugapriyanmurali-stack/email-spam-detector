# email-spam-detector
email spam detector using Ai
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

messages=["Win money now",
          "Free prize offer",
          "Call me tomorrrow",
          "Lets meet for lunch",
          "Congratulations you have won",
          "Hello how are you"]
labels = [1, 1, 0, 0, 1, 0]  # 1 = Spam, 0 = Not Spam

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(messages)

model = MultinomialNB()
model.fit(X, labels)

test_message = ["hi how are you .....you have an exciting offer"]
test_vector = vectorizer.transform(test_message)

prediction = model.predict(test_vector)

if prediction[0] == 1:
    print("Spam Message")
else:
    print("Not Spam Message")
