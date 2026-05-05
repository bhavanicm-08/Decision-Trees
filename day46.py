import pandas as pd

data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak'],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes']
}
df = pd.DataFrame(data)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for column in df.columns:
    df[column] = le.fit_transform(df[column])

X = df.drop('PlayTennis', axis=1)
y = df['PlayTennis']

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(X, y)

from sklearn import tree
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
tree.plot_tree(model, feature_names=X.columns, class_names=['No','Yes'], filled=True)
plt.show()