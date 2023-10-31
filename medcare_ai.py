import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression

# Load the data
data = pd.read_csv("updated_drg.csv")

# Create a Streamlit app
st.title("Healthcare Data Analysis")

# Display the data
st.header("Dataset")
st.write(data)

# Split the data into training and testing sets
features = ["drg_code", "drg_severity", "drg_mortality"]
target = "Risk"
X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)

# Train a logistic regression model
st.header("Logistic Regression Model")
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_report_text = classification_report(y_test, y_pred)

st.subheader("Model Evaluation")
st.write(f"Accuracy: {accuracy}")
st.write("Classification Report:")
st.code(classification_report_text)

# Select relevant features for clustering and segmentation
cluster_features = ["drg_severity", "drg_mortality"]

# Choose the number of clusters (K) based on your data and objectives
n_clusters = st.slider("Select the number of clusters", min_value=2, max_value=10, value=4)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=n_clusters)
data['cluster'] = kmeans.fit_predict(data[cluster_features])

# Visualize the clusters
st.header("K-Means Clustering for Severity of Illness")
fig, ax = plt.subplots()
scatter = ax.scatter(data["drg_severity"], data["drg_mortality"], c=data["cluster"], cmap='viridis')
ax.set_xlabel("DRG Severity")
ax.set_ylabel("DRG Mortality")
ax.set_title("K-Means Clustering for Severity of Illness")
st.pyplot(fig)

# Interpret the clusters
cluster_centers = kmeans.cluster_centers_

st.subheader("Cluster Centers")
st.write(cluster_centers)

# Segment patients based on the clusters
segmented_patients = [data[data['cluster'] == i] for i in range(n_clusters)]

st.subheader("Segments Based on Clusters")
fig, ax = plt.subplots()
for i, segment in enumerate(segmented_patients):
    ax.scatter(segment["drg_severity"], segment["drg_mortality"], label=f"Segment {i}")
ax.set_xlabel("DRG Severity")
ax.set_ylabel("DRG Mortality")
ax.set_title("Segments Based on Clusters")
ax.legend()

st.title("Data Visualizations")

# Histogram of drg_severity
st.header("Histogram of Severity of Illness")
fig, ax = plt.subplots()
ax.hist(data['drg_severity'], bins=20, color='blue', alpha=0.5)
ax.set_xlabel('Severity of Illness (DRG Severity)')
ax.set_ylabel('Frequency')
st.pyplot(fig)

# Create a TF-IDF vectorizer to convert text descriptions to numerical features
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # Adjust max_features as needed
X = data['description']
y_severity = data['drg_severity']
y_mortality = data['drg_mortality']
X_tfidf = tfidf_vectorizer.fit_transform(X)

# Train a Linear Regression model for severity
model_severity = LinearRegression()
model_severity.fit(X_tfidf, y_severity)

# Train a Linear Regression model for mortality
model_mortality = LinearRegression()
model_mortality.fit(X_tfidf, y_mortality)

# Add a section for prediction
st.header("Patient Severity and Mortality Prediction")

# Input for patient description
patient_description = st.text_area("Enter the patient description:")

# Button to trigger prediction
if st.button("Predict"):
    # Transform the patient description to a TF-IDF vector
    patient_description_tfidf = tfidf_vectorizer.transform([patient_description])

    # Make predictions
    prediction_severity = model_severity.predict(patient_description_tfidf)[0]
    prediction_mortality = model_mortality.predict(patient_description_tfidf)[0]

    # Display the predictions
    st.subheader("Predictions:")
    st.write(f"Predicted Severity: {prediction_severity:.2f}")
    st.write(f"Predicted Mortality: {prediction_mortality:.2f}")
