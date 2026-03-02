import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv("Crimes_-_2022.csv")

# Select important columns
df = df[['Primary Type', 'Arrest', 'Domestic',
         'District', 'Latitude', 'Longitude', 'Date']]

df = df.dropna()

# Convert Date
df['Date'] = pd.to_datetime(df['Date'])
df['Hour'] = df['Date'].dt.hour
df['Month'] = df['Date'].dt.month

# Encode categorical columns
le = LabelEncoder()
df['Primary Type'] = le.fit_transform(df['Primary Type'])
df['Domestic'] = le.fit_transform(df['Domestic'])
df['Arrest'] = le.fit_transform(df['Arrest'])

# =========================
# SUPERVISED LEARNING
# =========================

X = df[['Primary Type', 'Domestic', 'District', 'Hour', 'Month']]
y = df['Arrest']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print("Random Forest Results")
print(confusion_matrix(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)

print("XGBoost Results")
print(confusion_matrix(y_test, xgb_pred))
print(classification_report(y_test, xgb_pred))

# =========================
# UNSUPERVISED LEARNING
# =========================

location_data = df[['Latitude', 'Longitude']]
scaler = StandardScaler()
scaled_location = scaler.fit_transform(location_data)

kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster_KMeans'] = kmeans.fit_predict(scaled_location)

plt.figure()
plt.scatter(df['Longitude'], df['Latitude'],
            c=df['Cluster_KMeans'])
plt.title("Crime Hotspots - KMeans")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

dbscan = DBSCAN(eps=0.3, min_samples=10)
df['Cluster_DBSCAN'] = dbscan.fit_predict(scaled_location)

plt.figure()
plt.scatter(df['Longitude'], df['Latitude'],
            c=df['Cluster_DBSCAN'])
plt.title("Crime Hotspots - DBSCAN")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()