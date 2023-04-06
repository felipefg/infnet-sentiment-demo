"""
Pipeline for the simple naive-bayes based classifier.
"""
from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (select_data, data_metrics, split_train_test,
                    train_naive_bayes, report_model)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=select_data,
            name='select_data',
            inputs=['raw_tweets', 'params:min_confidence'],
            outputs='selected_tweets',
        ),
        node(
            func=data_metrics,
            name='data_metrics',
            inputs='selected_tweets',
            outputs='selected_tweets_metrics',
        ),
        node(
            func=split_train_test,
            name='split_train_test',
            inputs=[
                'selected_tweets',
                'params:test_size',
                'params:test_split_random_state'
            ],
            outputs=['data_train', 'data_test'],
        ),
        node(
            func=train_naive_bayes,
            name='train_naive_bayes',
            inputs=[
                'data_train',
                'params:n_folds',
                'params:param_grid',
            ],
            outputs='nb_model',
        ),
        node(
            func=report_model,
            name='report_model',
            inputs='nb_model',
            outputs='model_metrics',
        )
    ])
