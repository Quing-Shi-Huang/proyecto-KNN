from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os


app = Flask(__name__)

# Load the dataset and preprocess
data = pd.read_csv('data.csv')  # Make sure the CSV is in the same folder as app.py

# Encode Gender to numeric (0 for Male, 1 for Female)
label_encoder = LabelEncoder()
data['Gender'] = label_encoder.fit_transform(data['Gender'])  # 0 = Male, 1 = Female

# Separate the features and the target variable (Gender for gender prediction)
X = data.drop(columns=['CustomerID', 'Gender'])
y_gender = data['Gender']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans for clustering
optimal_clusters = 5
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
kmeans.fit(X_scaled)

# Train KNN classifiers for both gender and cluster predictions
X_train = X_scaled
y_train_cluster = kmeans.predict(X_train)

knn_cluster = KNeighborsClassifier(n_neighbors=5)
knn_cluster.fit(X_train, y_train_cluster)

knn_gender = KNeighborsClassifier(n_neighbors=5)
knn_gender.fit(X_train, y_gender)

# Route for the home page (index.html)
@app.route('/')
def index():
    return render_template('index.html')

# Route to process the form and show the result
@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    age = float(request.form['age'])
    annual_income = float(request.form['annual_income'])
    spending_score = float(request.form['spending_score'])

    # Prepare the user input for prediction
    user_data = np.array([[age, annual_income, spending_score]])
    user_data_scaled = scaler.transform(user_data)

    # Make predictions
    predicted_cluster = knn_cluster.predict(user_data_scaled)[0]
    predicted_gender = knn_gender.predict(user_data_scaled)[0]
    predicted_gender_label = 'Female' if predicted_gender == 1 else 'Male'

    # Generate the cluster visualization with the user's point
    plt.figure(figsize=(8, 6))

    # Plot the data points and color them by cluster
    plt.scatter(X_scaled[:, 1], X_scaled[:, 2], c=kmeans.labels_, cmap='viridis', alpha=0.6, s=100)

    # Highlighting the user's input data
    plt.scatter(user_data_scaled[:, 1], user_data_scaled[:, 2], c='red', marker='+', s=200, label='User Input', edgecolors='black')

    # Adding titles and labels
    plt.title('Customer Segmentation Based on Annual Income and Spending Score')
    plt.xlabel('Annual Income (Standardized)')
    plt.ylabel('Spending Score (Standardized)')
    plt.legend()

    # Save the graph as an image
    image_path = os.path.join('static', 'cluster_plot.png')
    plt.savefig(image_path)
    plt.close()  # Close the plot to free memory


    # Pass the results to the result.html template
    return render_template('result.html',
                           age=age,
                           annual_income=annual_income,
                           spending_score=spending_score,
                           cluster=predicted_cluster,
                           gender=predicted_gender_label,
                           image_file='cluster_plot.png')

if __name__ == '__main__':
    app.run(debug=True)