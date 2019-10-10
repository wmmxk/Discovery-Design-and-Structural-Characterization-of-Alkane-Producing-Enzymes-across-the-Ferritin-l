import numpy as np
import pandas as pd


def bootstrapping(data, seed=2019):
    np.random.seed(seed)
    idx = np.random.choice(data.index, data.shape[0]*3)
    return data.loc[idx, :]


def prepare_data(file_path):
    df = pd.read_csv(file_path).drop(columns=['Gene_name'])
    df.iloc[:, :7] = df.iloc[:, :7] - df.iloc[:, :7].min()
    df.iloc[:, :7] /= df.iloc[:, :7].max()
    df_train = df[df.Category == "Training set"].drop(columns=["Category"])
    df_test = df[df.Category != "Training set"].drop(columns=["Category"])
    return df_train, df_test


