import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(df: pd.DataFrame) -> (np.array, pd.DataFrame, np.array ):
    df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)
    label_encoder = LabelEncoder()
    df['Sex'] = label_encoder.fit_transform(df['Sex'])
    df['Embarked'] = label_encoder.fit_transform(df['Embarked'].astype(str))
    passenger_ids = df['PassengerId'].to_numpy()
    df = df.drop('PassengerId', axis=1)
    y = df['Survived'].to_numpy()
    df = df.drop('Survived', axis=1)
    imputer = KNNImputer(n_neighbors=5)
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    scaler = MinMaxScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    
    return (passenger_ids, df, y)

def preprocess_train_data(df: pd.DataFrame) -> (np.array, pd.DataFrame):
    df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)
    label_encoder = LabelEncoder()
    df['Sex'] = label_encoder.fit_transform(df['Sex'])
    df['Embarked'] = label_encoder.fit_transform(df['Embarked'].astype(str))
    passenger_ids = df['PassengerId'].to_numpy()
    df = df.drop('PassengerId', axis=1)
    imputer = KNNImputer(n_neighbors=5)
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    return (passenger_ids, df)

def save_solution(passenger_ids: np.array, predictions: np.array, output_path: str):
    solution = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': predictions})
    solution.to_csv(output_path, index=False)
    
def feature_reduction(X: pd.DataFrame, n_components: int) -> pd.DataFrame:
    tsne = TSNE(n_components=n_components)
    X = tsne.fit_transform(X)
    return X

def plot_2d_figure(X: pd.DataFrame, y: np.array, plot_path: str):
    tsne = TSNE(n_components=2)
    X = tsne.fit_transform(X)
    plt.scatter(X[y==0, 0], X[y==0, 1], color='red', label='0')
    plt.scatter(X[y==1, 0], X[y==1, 1], color='blue', label='1')
    plt.savefig(plot_path)