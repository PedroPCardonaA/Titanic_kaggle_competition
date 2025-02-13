from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

class RandomForestModel:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def fit(self):
        self.model = RandomForestClassifier()
        self.model.fit(self.X_train, self.y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
        

class XGBoostModel:

    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def fit(self):
        self.model = xgb.XGBClassifier()
        self.model.fit(self.X_train, self.y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)