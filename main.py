import pandas as pd
df = pd.read_csv("data/data.csv")
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())
# Remove unnecessary columns
df.drop(["id", "Unnamed: 32"], axis=1, inplace=True)

# Convert diagnosis to numbers
df["diagnosis"] = df["diagnosis"].map({
    "M": 1,
    "B": 0
})

print(df.head())

#Seprating datasets into X and Y

x = df.drop("diagnosis", axis=1)
y = df["diagnosis"]
print(df.info())
print(df["diagnosis"].value_counts())
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,  
    test_size=0.2,
    random_state=42
)

print(x_train.shape)
print(x_test.shape)

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=10,
    random_state=42
)

model.fit(x_train, y_train)
y_pred = model.predict(x_test)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 10))
plot_tree(
    model,
    feature_names=x.columns,
    class_names=["Benign", "Malignant"],
    filled=True,
    rounded=True,
    fontsize=8
)
plt.show()