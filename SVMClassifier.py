import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Sample dataset
data = {
    "Waste": ["banana peel", "plastic bottle", "broken glass", "apple core", "newspaper", "old battery"],
    "Category": ["Organic", "Recyclable", "Hazardous", "Organic", "Recyclable", "Hazardous"]
}

df = pd.DataFrame(data)

# Convert text labels to numerical values
category_mapping = {category: idx for idx, category in enumerate(df["Category"].unique())}
df["Category"] = df["Category"].map(category_mapping)

# Feature extraction (convert text data to numerical format)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["Waste"])
y = df["Category"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train an SVM classifier
model = SVC(kernel="linear")
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Predict a new waste item
new_waste = ["paper cup"]
new_waste_vector = vectorizer.transform(new_waste)
predicted_category = model.predict(new_waste_vector)
reverse_mapping = {idx: category for category, idx in category_mapping.items()}
print("Predicted Category:", reverse_mapping[predicted_category[0]])
