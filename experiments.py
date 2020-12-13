import classifier
import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pandas as pd
import time


clfs = [
    classifier.WeightClassifierCDF(),
    classifier.WeightClassifierABS(),
    KNeighborsClassifier(),
    LogisticRegression(),
    RandomForestClassifier(),
    SVC()
]

pipelines = [
    make_pipeline(StandardScaler(), clf)
    for clf in clfs
]

results = pd.DataFrame(
    columns=[
        'experiment_number',
        'dataset_name',
        'classifier_name',
        'accuracy',
        'confusion_matrix',
        'time (s)',
    ]
)

for experiment_iter in range(10):
    dataset_list = [
        data
        for data
        in datasets.datasets[:]
        if data['name'] != 'mushdata.mat'
    ]
    for d in dataset_list:
        dataset_list.set_description(d['name'])
        X = d['X']
        train_X, test_X, train_y, test_y = train_test_split(
            X,
            d['y'],
            train_size=0.7,
        )
        for pipeline in pipelines:
            pipeline[-1].class_ts = None
            try:
                start = time.time()
                pipeline.fit(train_X, train_y)
                predicted = pipeline.predict(test_X)
                end = time.time()
                results = results.append(
                    {
                        'experiment_number': experiment_iter,
                        'dataset_name': d['name'],
                        'classifier_name': type(pipeline[-1]).__name__,
                        'accuracy': (
                            (predicted == test_y).sum() /
                            test_y.shape[0]
                        ),
                        'confusion_matrix': [
                            confusion_matrix(test_y, predicted)
                        ],
                        'time (s)': end-start,
                    },
                    ignore_index=True,
                )
            except Exception as e:
                print(f'Exception: {e}')
                results = results.append(
                    {
                        'experiment_number': experiment_iter,
                        'dataset_name': d['name'],
                        'classifier_name': pipeline[-1].__name__,
                        'accuracy': None,
                        'confusion_matrix': None,
                        'time (s)': 0.,
                    },
                    ignore_index=True,
                )

table = results.drop(
    [
        'confusion_matrix',
        'experiment_number',
        'time (s)'
    ],
    axis=1
).groupby(
    [
        'dataset_name',
        'classifier_name',
    ]
).agg(lambda x: f'{np.mean(x):.2f} \u00B1 {np.std(x):.2f}').unstack().copy()

print(table.accuracy.drop('WeightClassifierCDF', axis=1).rename(
    columns={
        'KNeighborsClassifier': 'K-Neighbors',
        'LogisticRegression': 'Logistic Reg.',
        'RandomForestClassifier': 'Random Forest',
        'SVC': 'SVM',
        'WeightClassifierABS': 'Weight',
    }
).drop(
    ['checkdata.mat', 'bupadata.mat', 'pimadata.mat', 'wpbc60data.mat'],
    axis=0
).to_latex())

time_table = results.drop(
    [
        'confusion_matrix',
        'experiment_number',
        'accuracy'
    ],
    axis=1
).groupby(
    [
        'dataset_name',
        'classifier_name',
    ]
).agg(lambda x: f'{np.mean(x):.2f} \u00B1 {np.std(x):.2f}').unstack().copy()

print(time_table['time (s)'].drop('WeightClassifierCDF', axis=1).rename(
    columns={
        'KNeighborsClassifier': 'K-Neighbors',
        'LogisticRegression': 'Logistic Reg.',
        'RandomForestClassifier': 'Random Forest',
        'SVC': 'SVM',
        'WeightClassifierABS': 'Weight',
    }
).drop(
    ['checkdata.mat', 'bupadata.mat', 'pimadata.mat', 'wpbc60data.mat'],
    axis=0
).to_latex())
