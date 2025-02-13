from utils.utils import *
from model.classifier_model import RandomForestModel, XGBoostModel
import pandas as pd

def main():
    df = pd.read_csv('data/train.csv')
    passenger_ids, X, y = preprocess_data(df)
    model = XGBoostModel(X, y)
    model.fit()
    df_test = pd.read_csv('data/test.csv')
    passenger_ids_test, X_test = preprocess_train_data(df_test)
    predictions = model.predict(X_test)
    save_solution(passenger_ids_test, predictions, 'data/solution.csv')



if __name__ == '__main__':
    main()