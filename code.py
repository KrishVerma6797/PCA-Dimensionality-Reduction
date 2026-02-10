import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

digits=load_digits()
x=digits.data
y=digits.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)

#PCA
components=[2,10,30,50]
pca_models={}

for n in components:
    pca=PCA(n_components=n)
    pca.fit(x_train_scaled)
    pca_models[n]=pca

#Explained Variance & Cumulative Variance Plot
pca_full=PCA()
pca_full.fit(x_train_scaled)
cum_var=np.cumsum(pca_full.explained_variance_ratio_)

plt.figure(figsize=(8,5))
plt.plot(cum_var,marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance vs Number of Components')
plt.grid()
plt.show()

#Dataset Transformation (Dimensionality Reduction)
pca_30=PCA(n_components=30)
x_train_pca=pca_30.fit_transform(x_train_scaled)
x_test_pca=pca_30.transform(x_test_scaled)

#Logistic Regression (Original vs Reduced)
lr_original=LogisticRegression(max_iter=1000)
lr_original.fit(x_train_scaled,y_train)
y_pred_original=lr_original.predict(x_test_scaled)
acc_original=accuracy_score(y_test,y_pred_original)

lr_pca=LogisticRegression(max_iter=1000)
lr_pca.fit(x_train_pca,y_train)
y_pred_pca=lr_pca.predict(x_test_pca)
acc_pca=accuracy_score(y_test,y_pred_pca)

## Accuracy Comparison Report
print("Accuracy (Original Data):", acc_original)
print("Accuracy (PCA Reduced Data - 30 components):", acc_pca)

#2D PCA Visualization
pca_2=PCA(n_components=2)
x_2d=pca_2.fit_transform(x_train_scaled)

plt.figure(figsize=(8,6))
scatter=plt.scatter(x_2d[:,0],x_2d[:,1],c=y_train,cmap='tab10',s=15)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2D PCA Visualization of Digits Dataset')
plt.colorbar(scatter)
plt.show()