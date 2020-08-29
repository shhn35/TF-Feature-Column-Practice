# This a demo on Tensorflow Feature Column
In this demo a represtation of tf.feature_column is presented based on well known Titanic dataset, which [available on](http://storage.googleapis.com/tf-datasets/titanic/train.csv)

## List of contents
1. Visulizing the data set as a DataFrame
1. Presenting a demo on Featute_Column

### Visulizing the data set as a DataFrame
The first part of this demo focuses on dataset it self and gives some data visualization based on pandas.dataframe, which shows the main charectristics of the data itself.

### Presenting a demo on Featute_Column
The second part, tries to cover some main tf.feature_columns including [_**BucketizedColumn**_](https://www.tensorflow.org/api_docs/python/tf/feature_column/bucketized_column), [_**NumericColumn**_](https://www.tensorflow.org/api_docs/python/tf/feature_column/numeric_column), [_**CategoricalColumnWithVocabulary**_](https://www.tensorflow.org/api_docs/python/tf/feature_column/categorical_column_with_vocabulary_list), and wrapping _CaltegoricalColumn_ using [_**IndicatorColumn**_](https://www.tensorflow.org/api_docs/python/tf/feature_column/indicator_column). A [_**CrossFeatureColumn**_](https://www.tensorflow.org/api_docs/python/tf/feature_column/crossed_column) is also included to demonstrate how individual features can mixed together in order to privide a more depth knowledge of correlational features such as _age_ and _sex_.
Tolls like [_**FACET**_] (https://pair-code.github.io/facets/) and [_**Embeding Projector**_](https://projector.tensorflow.org) is out there to monitor the effecr of single individual featuter to an other one, and eventually identify the correlation between two individual features.
Moreover, the differences between the raw data and **DenseFeatures** is presented in the output to demonstrate how feature_column transform raw data in a form that is suitable for a tf.model

