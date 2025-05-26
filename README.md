# 📧 Spam_Shield

**Spam_Shield** is a simple and effective spam detection system that uses machine learning to classify SMS and email messages as **Spam** or **Ham** (not spam). It leverages natural language processing (NLP) techniques and a **Multinomial Naive Bayes** classifier to make accurate predictions based on message content.

---

## 🚀 Features

- Text preprocessing with cleaning, tokenization, stopword removal, and stemming.
- Spam detection using Multinomial Naive Bayes.
- High accuracy on both training and test data.
- Predicts custom message input for real-time spam detection.

---

## 🧠 Model Summary

| Metric           | Score      |
|------------------|------------|
| Train Accuracy   | 99.21%     |
| Test Accuracy    | 97.67%     |

---

## 🛠️ How It Works

### 1. **Text Cleaning Function**

```python
def textcleaner(data):
    corpus_list = []
    for i in range(len(data)):
        rp = re.sub('[^a-zA-Z]', " ", data['message'][i])
        rp = rp.lower()
        rp = rp.split()
        rp = [ps.stem(word) for word in rp if word not in stop_words]
        rp = " ".join(rp).strip()
        corpus_list.append(rp)
    return corpus_list
```
- Removes special characters
- Converts text to lowercase
- Removes stopwords
- Applies stemming

### 2. **Model Training**
```python
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(X_train, y_train)
```
### 3. **Prediction Example**
```python
check_data = "Congratulations! You've won a $1000 gift card to Amazon. Click here to claim"
check_data = pd.DataFrame({"message": [check_data]})

check_corpus = textcleaner(check_data)
check_corpus = cv.transform(check_corpus).toarray()

check_pred = model.predict(check_corpus)

print("Spam" if check_pred[0] else "Ham")

```

## 📊 Dataset

The project uses a labeled dataset of SMS messages marked as **spam** or **ham**, preprocessed into a **Bag-of-Words** model for training.

---

## 📌 Future Improvements

- Incorporate **TF-IDF** vectorization for more informative features.
- Use deep learning models (e.g., **LSTM**, **BERT**) for better contextual understanding.
- Deploy as a **web** or **mobile application**.

---

