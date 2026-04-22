from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def evaluate_ml_utility(X_real_train, X_real_test, y_train, y_test, X_synthetic):

    min_len = min(len(X_synthetic), len(y_train))

    X_syn = X_synthetic.iloc[:min_len]
    y_syn = y_train.iloc[:min_len]

    model = LogisticRegression()
    model.fit(X_syn, y_syn)

    y_pred = model.predict(X_real_test)

    return accuracy_score(y_test, y_pred)