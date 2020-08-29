import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
# from six.moves import urllib
import urllib

import create_feature_column as cfc

def read_dataset_as_dataFrame(lable_column_name,test_size_percent=0.3,shuffle = False):
    '''
    Read dataset from a csv file on the internet.
    Returns train and test dataset as Pandas.Dataframe objects. 
    In addition, returns the lable_column_name for furthur usage
    '''
    df_data = pd.read_csv('http://storage.googleapis.com/tf-datasets/titanic/train.csv')
    df_data = df_data.append (pd.read_csv('http://storage.googleapis.com/tf-datasets/titanic/eval.csv'))

    df_train_data, df_test_data = train_test_split(df_data,test_size=test_size_percent,shuffle=shuffle)

    # # An alternative way to get the dataset regarles of given test_size_percent
    # df_train_data = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
    # df_test_data = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')
    

    return df_train_data,df_test_data

# Convert Dataframe to tf.dataset
def df_to_tf_dataset(dataframe,lable_column_name,shuffle=False,batch_size=32):
    '''
    Converts the input 'dataFrame' object to Tensorflow Dataset object 
    If 'shuffle' is on, the returned dataset is shuffled befor batch operation
    '''
    shuffle_buffer_size = 1024
    if len(dataframe) < shuffle_buffer_size:
        shuffle_buffer_size = len(dataframe)

    targets = dataframe.pop(lable_column_name)

    tf_ds = tf.data.Dataset.from_tensor_slices ((dict(dataframe),targets))
    
    if shuffle:
        tf_ds = tf_ds.shuffle(buffer_size = shuffle_buffer_size)
    
    tf_ds = tf_ds.batch(batch_size = batch_size)
        
    return tf_ds

def get_ds_train_test_featureColumn(lable_column_name,train_batch_size=32,test_batch_size=1,shuffle=False,test_size_percent=0.3):
    '''
    Gets the final dataset (Train and Test in separated) as TF.dataset object, inaddition of created the feature_columns
    '''

    # Get dataset as Dataframe object 
    df_train_data,df_test_data = read_dataset_as_dataFrame(lable_column_name,test_size_percent=test_size_percent,shuffle=shuffle)

    # Get feature_columns
    feature_columns = cfc.create_feature_columns(df_train_data,lable_column_name)

    # Transfer Dataframe object to tf.dataset object to feed in to a tf.model as a DenseFeature
    ds_train = df_to_tf_dataset(df_train_data,lable_column_name, batch_size = train_batch_size, shuffle=shuffle)
    ds_test = df_to_tf_dataset(df_test_data,lable_column_name,batch_size = test_batch_size,shuffle=shuffle)

    return ds_train,ds_test,feature_columns

