import pandas as pd
import numpy as np

# ----- Part A: Manual Bayes' Theorem example (classic "Play Tennis" concept learning) -----
data = {
    'Outlook':    ['Sunny','Sunny','Overcast','Rain','Rain','Rain','Overcast','Sunny','Sunny','Rain','Sunny','Overcast','Overcast','Rain'],
    'Temperature':['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild'],
    'Humidity':   ['High','High','High','High','Normal','Normal','Normal','High','Normal','Normal','Normal','High','Normal','High'],
    'Wind':       ['Weak','Strong','Weak','Weak','Weak','Strong','Strong','Weak','Weak','Weak','Strong','Strong','Weak','Strong'],
    'PlayTennis': ['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']
}
df = pd.DataFrame(data)

def calc_prior(df, target_col):
    return df[target_col].value_counts(normalize=True)

def calc_likelihood(df, feature, value, target_col, target_value):
    subset = df[df[target_col] == target_value]
    return (subset[feature] == value).sum() / len(subset)

# New instance to predict: Outlook=Sunny, Temperature=Cool, Humidity=High, Wind=Strong
new_instance = {'Outlook': 'Sunny', 'Temperature': 'Cool', 'Humidity': 'High', 'Wind': 'Strong'}
priors = calc_prior(df, 'PlayTennis')

posteriors = {}
for cls in df['PlayTennis'].unique():
    prob = priors[cls]
    for feature, value in new_instance.items():
        prob *= calc_likelihood(df, feature, value, 'PlayTennis', cls)
    posteriors[cls] = prob

print("Priors:\n", priors)
print("\nUnnormalized posteriors:", posteriors)

predicted_class = max(posteriors, key=lambda k: posteriors[k])
print(f"\nPrediction for {new_instance}: PlayTennis = {predicted_class}")

# ----- Part B: Using sklearn's GaussianNB on a real dataset -----
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42) # pyright: ignore[reportAttributeAccessIssue]

nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)

print("\nGaussianNB Accuracy on Iris:", accuracy_score(y_test, y_pred))