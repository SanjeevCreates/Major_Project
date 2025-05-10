import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from feature_selection import wolf_optimizer

train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

X_train = train_df.drop(['subject', 'Activity'], axis=1)
y_train = train_df["Activity"]
X_test = test_df.drop(['subject', 'Activity'], axis=1)
y_test = test_df["Activity"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
selected_features = wolf_optimizer(X_scaled, y_train, num_features=20)
print("Selected Feature Indices:", selected_features)

X_selected = X_scaled[:, selected_features]
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_selected, y_train)

y_pred = clf.predict(X_selected)
accuracy = accuracy_score(y_train, y_pred)
print("Accuracy with selected features:", accuracy)

selected_feature_names = X_train.columns[selected_features]
print("Selected Feature Names:", selected_feature_names)

Xtest_selected = X_test.iloc[:, selected_features]

joblib.dump(clf, 'models/classification_model.h5')
joblib.dump(scaler, 'models/scaler.pkl')