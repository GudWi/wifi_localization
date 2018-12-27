import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn import linear_model, metrics

df = pd.read_csv(r'C:\Users\GudWin\Desktop\wifi_localization.txt', sep="\t", header=None, names=["s1", "s2", "s3", "s4", "s5", "s6", "s7", "room"])
X = df.iloc[:, :7]
y = df.iloc[:, 7]
X_train, X_test, y_train, y_test = train_test_split(X,y, shuffle=True, stratify=y, test_size=0.1)
log_full = linear_model.LogisticRegression()
log_full.fit(X_train, y_train)
full_predict = log_full.predict(X_test)
print(y_test)
print(full_predict)
print("Accuracy", metrics.accuracy_score(y_test, full_predict))
