import pandas as pd
import numpy as np


def null_report(df):
    total = df.isnull().sum()
    perc = total / df.isnull().count() * 100
    tt = pd.concat([total, perc], axis=1, keys=['Total', 'Percent'])
    types = []
    for col in df.columns:
        dtypeimport pandas as pd
        import numpy as np
        
        class null_report():
            """This class provides dataframe NaN reporting functionality in tidy form
            """
            
            def generate_report(self):
                total = self.isnull().sum()
                perc = total / self.isnull().count() * 100
                new_frame = pd.concat([total, perc], axis=1, keys=['Total', 'Percent'])
                types = []
                for col in self.columns:
                    dtype = str(self[col].dtype)
                    types.append(dtype)
                new_frame['Types'] = types
                return np.transpose(new_frame)
        
        def train_val_test_split(df):
            train, val, test = np.split(df.sample(frac=1), [int(.6
                                          * len(df)), int(.8 * len(df))])
            return train, val, test
        
        def add_list_to_df(df, lst):
            """This function takes a dataframe and a list,
            then adds the list to the dataframe as a new column
            """
            s = pd.Series(lst)
            return pd.concat([df, s], axis=1)
        
        
        def simple_confusion_matrix(y_true, y_pred):
            y_true = pd.Series(y_true, name='True')
            y_pred = pd.Series(y_pred, name='Predicted')
            return pd.crosstab(y_true, y_pred, margins=True)
        
        
        def show_full_frames():
            pd.set_option('display.max_rows', 500)
            pd.set_option('display.max_columns', 500)
            pd.set_option('display.width', 1000)
        
        
        def split_datetime(df, col):
            df[col] = df[col].to_datetime()
            df['month'] = df[col].dt.month
            df['year'] = df[col].dt.year
            df['day'] = df[col].dt.day
         = str(df[col].dtype)
        types.append(dtype)
    tt['Types'] = types
    return np.transpose(tt)


def train_val_test_split(df):
    (train, val, test) = np.split(df.sample(frac=1), [int(.6
                                  * len(df)), int(.8 * len(df))])
    return (train, val, test)


class complex_number:

    def __init__(self, r=0, i=0):
        self.real = r
        self.imag = i

    def getData(self):
        print '{0}+{1}j'.format(self.real, self.image)


def add_list_to_df(df, lst):
    s = pd.Series(lst)
    return pd.concat(df, s)


def simple_confusion_matrix(y_true, y_pred):
    y_true = pd.Series(y_true, name='True')
    y_pred = pd.Series(y_pred, name='Predicted')
    return pd.crosstab(y_true, y_pred, margins=True)


def show_full_frames():
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)


def split_datetime(df, col):
    df[col] = df[col].to_datetime()
    df['month'] = df[col].dt.month
    df['year'] = df[col].dt.year
    df['day'] = df[col].dt.day
