import os
import zipfile
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

train_zip_path = "features_640.zip"
test_zip_path = "features_val_640.zip"
train_extract_path = "features_640/features_640"
test_extract_path = "features_val_640/features_val_640"

if not os.path.exists("features_640"):
    with zipfile.ZipFile(train_zip_path, "r") as zip_ref:
        zip_ref.extractall("features_640")

if not os.path.exists("features_val_640"):
    with zipfile.ZipFile(test_zip_path, "r") as zip_ref:
        zip_ref.extractall("features_val_640")

X_train, y_train = [], []
label_mapping = {}
for idx, file_name in enumerate(sorted(os.listdir(train_extract_path))):
    label_mapping[file_name] = idx
    file_path = os.path.join(train_extract_path, file_name)
    data = loadmat(file_path)
    X_train.append(data['feature'])
    y_train.append(np.full((data['feature'].shape[0],), idx))

X_train = np.vstack(X_train)
y_train = np.hstack(y_train)

X_test, y_test = [], []
for file_name in sorted(os.listdir(test_extract_path)):
    file_path = os.path.join(test_extract_path, file_name)
    idx = label_mapping[file_name]
    data = loadmat(file_path)
    X_test.append(data['feature'])
    y_test.append(np.full((data['feature'].shape[0],), idx))

X_test = np.vstack(X_test)
y_test = np.hstack(y_test)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf = SGDClassifier(
    loss="log_loss", penalty="l2", max_iter=1000, tol=1e-4,
)
clf.fit(X_train, y_train)

y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

train_error = 1 - accuracy_score(y_train, y_train_pred)
test_error = 1 - accuracy_score(y_test, y_test_pred)

print(f"Training Misclassification Error: {train_error:.4f}")
print(f"Test Misclassification Error: {test_error:.4f}")
