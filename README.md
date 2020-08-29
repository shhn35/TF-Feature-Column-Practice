# This a demo on Tensorflow Feature Column
In this demo a represtation of tf.feature_column is presented based on well known Titanic dataset, which [available on](http://storage.googleapis.com/tf-datasets/titanic/train.csv)

## List of contents
1. Visulizing the data set as a DataFrame
1. Presenting a demo on Featute_Column

### Visulizing the data set as a DataFrame
The first part of this demo focuses on dataset it self and gives some data visualization based on pandas.dataframe, which shows the main charectristics of the data itself.

### Presenting a demo on Featute_Column
The second part, tries to cover some main tf.feature_column including _**BucketizedColumn**_, _**NumericColumn**_, _**CategoricalColumnWithVocabulary**_, and wrapping _CaltegoricalColumn_ using _**IndicatorColumn**_.
Moreover, the differences between the raw data and **DenseFeatures** is presented in the output to demonstrate how feature_column transform raw data in a form that is suitable for a tf.model

