import numpy as np
import pandas as pd
from tensorflow.keras import layers
import tensorflow.python.feature_column as fc
import tensorflow as tf

import df_dataset_overview as df_overview
from read_dataset import get_ds_train_test_featureColumn as get_dataset 

def main():
    lable_column_name = 'survived'

    # display some charectristic of data as dataFrame object
    df_overview.dataframe_overview(lable_column_name)

    # demonstrate the raw input data vs data after transforming to the feature_column as DenseFeatured
    demo_with_feature_column(lable_column_name)


def demo_with_feature_column(lable_column_name):
    '''
    Demonstrating of severaal types of feature columns after preprocessing step as the input of tf.model
    '''

    # Read dataset as tf.datatset and also the created Feature_columns
    ds_train,ds_test,feature_columns = get_dataset(lable_column_name=lable_column_name,train_batch_size= 5, test_batch_size=1, shuffle=False,test_size_percent=0.3)


    # Printing all created feature_columns
    print('\ntf.Feature Columns:')

    i=0
    for f_column in feature_columns:
        i+=1
        ## in case of having a Categorical_feature_column witch is wrapped with Embedding_column
        # if(not isinstance(f_column,tf.python.feature_column.EmbeddingColumn) and not isinstance(f_column,tf.python.feature_column.BucketizedColumn)):

        ## in case of having a Categorical_feature_column witch is wrapped with Indicator_column
        if(not isinstance(f_column,fc.BucketizedColumn) and not isinstance(f_column,fc.IndicatorColumn)):
            # For Numeric Feature Columns
            print('Feature Column['+str(i)+']-> ('+f_column.key+'): ',f_column)
        else:
            if(not isinstance(f_column,fc.BucketizedColumn)):
                #For Categorical_fearur_columns which is wrapped with IndicatorColumn
                print('Feature Column['+str(i)+']-> ('+f_column.categorical_column.key+'): ',f_column.categorical_column)
            else:
                #For BucketizedColumn feature columns
                print('Feature Column['+str(i)+']->  ('+f_column.source_column.key+'): ',f_column.source_column)


    # Start Demo: presenting input data as Feature_Columns
    sample_input_data = iter(ds_test)
    # print(data[0].keys())

    for i in range(10):
        data=next(sample_input_data)

        print('\nDenseFeature representation:')
        show_data_as_denseFeatures(feature_columns,data[0])

        print('Raw data represntation:')
        for f_column in feature_columns:
            ## in case of having a Categorical_feature_column witch is wrapped with Embedding_column
            # if(isinstance(f_column,fc.EmbeddingColumn)):
            if(isinstance(f_column,fc.IndicatorColumn)):
                ## in case of having a Categorical_feature_column witch is wrapped with Indicator_Column
                print(f_column.categorical_column.key,': ',data[0][f_column.categorical_column.key])
            else:
                if(isinstance(f_column,fc.BucketizedColumn)):
                    # BucketizedColumn_feature_columns
                    print(f_column.source_column.key,': ',data[0][f_column.source_column.key])
                else:
                    # Numerical_feature_column
                    print(f_column.key,': ',data[0][f_column.key])            


def show_data_as_denseFeatures(feature_column,example_batch):
    '''
    Gets a sample batch of data (as tf.datased object ) and shows the input data after applying feature_column.
    The result is the exact values for the example_batch, which is fed into the tf.model directly
    '''
    feature_layers= layers.DenseFeatures(feature_column)
    print(feature_layers(example_batch).numpy()[0])


if __name__ == "__main__":
    main()