"""
This is a boilerplate pipeline 'simple_classifier'
generated using Kedro 0.18.7
"""
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer

def select_data(data, min_confidence):
    """Discard low confidence"""
    return data.query('airline_sentiment_confidence >= @min_confidence')


def data_metrics(data):
    """Gather metrics for the dataset, for mlflow logging."""
    metrics = {
        'n_total': data.shape[0],
    }

    for klass, count in data['airline_sentiment'].value_counts().items():
        metrics[f'n_{klass}'] = count

    return {
        key: {'value': value, 'step': 1}
        for key, value in metrics.items()
    }


def split_train_test(data, test_size, random_state):
    """Split incoming data into train and test sets"""
    train, test = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state
    )
    return train, test


def train_naive_bayes(train, n_folds, param_grid):
    """Train the Naïve Bayes classifier with cross-validated grid search."""

    col_transformer = ColumnTransformer([
        ('tfidf', TfidfVectorizer(), 'text'),
    ])

    pipe = Pipeline(
        [
            ('ct', col_transformer),
            ('clf', MultinomialNB())
        ]
    )
    clf = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=n_folds),
        scoring=['f1_weighted', 'precision_weighted', 'recall_weighted'],
        refit='f1_weighted',
    )

    X = train
    y = train['airline_sentiment']
    clf.fit(X, y)
    return clf


def report_model(clf):

    idx = clf.best_index_

    metrics = {
        'f1_mean': clf.cv_results_['mean_test_f1_weighted'][idx],
        'f1_std': clf.cv_results_['std_test_f1_weighted'][idx],
        'precision_mean': clf.cv_results_['mean_test_precision_weighted'][idx],
        'precision_std': clf.cv_results_['std_test_precision_weighted'][idx],
        'recall_mean': clf.cv_results_['mean_test_recall_weighted'][idx],
        'recall_std': clf.cv_results_['std_test_recall_weighted'][idx],
    }

    return {
        key: {'value': float(value), 'step': 1}
        for key, value in metrics.items()
    }
