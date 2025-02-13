import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
import numpy as np

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
    