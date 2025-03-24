import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def clean_data(df):
    df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)
    return df

def encode_features(df):
    le_gender = LabelEncoder()
    df['Gender'] = le_gender.fit_transform(df['Gender'])
    df = pd.get_dummies(df, columns=['Geography'], drop_first=True)
    return df

def scale_features(df):
    scaler = StandardScaler()
    scaled_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    df[scaled_cols] = scaler.fit_transform(df[scaled_cols])
    return df
