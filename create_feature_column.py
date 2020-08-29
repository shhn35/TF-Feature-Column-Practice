import tensorflow as tf


def create_feature_columns(dataframe,lable_column_name):
    '''
    Creates Tensorflow Feature_column and returns the feature_columns from the Datafram dataset
    '''
    NUMERICAL_COLUMN_NAMES = ['age','fare']
    
    CATECORICAL_COLUMN_NAMES =list(dataframe.columns.unique())
    CATECORICAL_COLUMN_NAMES = [x for x in CATECORICAL_COLUMN_NAMES if x not in NUMERICAL_COLUMN_NAMES] 
    CATECORICAL_COLUMN_NAMES.remove(lable_column_name)

    feature_columns = []

    # Bucketized-column of a Numeric_column 'Age'
    age = tf.feature_column.numeric_column(NUMERICAL_COLUMN_NAMES.pop(0))
    age_bucketized = tf.feature_column.bucketized_column(age,boundaries=[20,25,40,55,80,100])
    feature_columns.append(age_bucketized)

    # Numerica Features Column
    for feature_name in NUMERICAL_COLUMN_NAMES:
        feature_columns.append(tf.feature_column.numeric_column(feature_name,dtype=tf.float32))
    
    # Categorical Features Column with Vocabulary
    for feature_name in CATECORICAL_COLUMN_NAMES:
        feature_vocabulary_list = dataframe[feature_name].unique()

        cat_feature_column_with_vocab = tf.feature_column.categorical_column_with_vocabulary_list(feature_name,feature_vocabulary_list)

        ## Wrap the caltegorical column with Embeding_column in case of having larg vocabulary list
        ## Finding the best value for 'dimention' is a challange here and it depends on the data itself
        # feature_columns.append(tf.feature_column.embedding_column(tf.feature_column.categorical_column_with_hash_bucket(feature_name,10),dimension=1))

        ## Wrap the categorical column with Indicator_column in case of having alimitted vocabulary list
        feature_columns.append(tf.feature_column.indicator_column(cat_feature_column_with_vocab))
    
    return feature_columns
