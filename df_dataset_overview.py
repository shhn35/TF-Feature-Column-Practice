import pandas as pd
import matplotlib.pyplot as plt

import read_dataset as rd

def dataframe_overview():
    '''
    Give an onverview on the dataset
    '''

    df_train,df_test,lable_column_name = rd.read_dataset_as_dataFrame()

    # dataFrame functionalities on raw data
    print('Data overview on well-known "Titanic" dataset->')
    print(len(df_train),' Smaples for Training')
    print(len(df_test),' Smaples for Test')

    print('\nTrain data Head:')
    print(df_train.head())

    print('\nTrain data description based on DataFrame:')
    print(df_train.describe())

    print('\nHistogram of the column "age" for the Train data with 25 bins:')
    df_train.age.hist(bins=25)
    plt.show()

    # Chart representation of number of Male and Female in the Train data
    df_train.sex.value_counts().plot(kind='bar',title='Number of Male and Female in the Train data:')
    plt.show()

    # Chart representation of the "class" distribution among all passengers in the Train data
    df_train['class'].value_counts().plot(kind='pie',title='"class" distribution among all passengers in the Train data')
    plt.show()

    # The ration of all survived Female to all on board Females as well as for Males
    df_train.groupby('sex').survived.mean().plot(kind='bar',table=True,title='The ration of all survived Female to all on board Females as well as for Males').set_xlabel('survived')
    plt.show()
    
    # The number of alived and dead females and males
    df_train.groupby('sex').survived.value_counts().plot(kind='bar',legend='true',table=True,title='The number of alived and dead females and males')
    plt.show()

    # The ration of all survived Female to all on board Females as well as for Males based on theri "class"
    df_train.groupby(['sex','class']).survived.mean().plot(kind='bar',legend='true',table=True,title='The ration of all survived Female to all on board Females as well as for Males based on theri "class"')
    plt.show()

    # pd.concat([df_train,y_train],axis=1).groupby(['sex','class']).survived.value_counts().plot(kind='bar',legend='true',table=True)
    # plt.show()
