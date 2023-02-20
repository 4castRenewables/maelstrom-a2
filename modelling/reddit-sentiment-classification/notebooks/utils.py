import numpy as np


def convert_df_labels(df, num_labels):
    """
    convert array of labels as part of dataframe to one hot encoded format,
    [[2,3], [1]] -> [[0,0,1,1], [0,1,0,0]] for num_labels = 4
    """
    y = np.zeros((df.shape[0], num_labels))
    for index, row in enumerate(df["labels"].values):
        for r in row:
            y[index, r] = 1
    return y


def remove_ambiguous_data(df, y):
    """
    remove 'neutral' texts with additional emotion
    """
    mask = np.logical_and(y[:, -1] == 1, y.sum(axis=-1) > 1)
    df = df.iloc[np.where(np.logical_not(mask))[0]].reset_index(drop=True)
    return df


def binarize_labels_torch(labels):
    """
    returns labels in format [0, 1, 1, 0,.....]
    """
    y_binary = np.zeros(labels.shape[0])
    mask = labels[:, -1] == 1
    y_binary[mask] = 1
    return y_binary
