from utils.utils import *
from model.classifier_model import RandomForestModel, XGBoostModel
import pandas as pd

def main():
    df = pd.read_csv('data/train.csv')
    passenger_ids, X, y = preprocess_data(df)
    print(X)
    X = feature_reduction(X, 2)
    plot_2d_figure(X, y, 'plots/plot.png')



if __name__ == '__main__':
    main()