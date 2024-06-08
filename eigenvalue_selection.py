from collections import OrderedDict
from datetime import datetime

import thr_students

from utils.plot_utils import plot_metrics as plot_metrics
from utils.experiments_utils import run_experiment as run_experiment_students


if __name__ == '__main__':

    # Approaches to test
    approaches = ['fairdas']


    verbose = True

    # Define data
    config = dict(
        historical_batches=30,
        n_batches=100,
        batch_dim=32,

        # alpha * Uniform + (1-alpha) * Sampling noise
        alpha=0.1,

        # Define actions
        actions='polynomial_fn', # 'group_weights' or 'polynomial_fn'

        scaling='IQR_normalization' #'None', or 'normalization' or 'IQR_normalization'
     )
    # To seed data generation process
    seeds = [1234, 3245, 4242, 5627,6785, 8282, 9864, 9921]

    # Run experiment with only fairness metric
    if config['actions'] == 'group_weights':
        exp0 = {'exp0': OrderedDict(
            GeDI=0.5,
            score_absolute_error=0.5
        )}
    elif config['actions'] == 'polynomial_fn':
        exp0 = {'exp0': OrderedDict(
            GeDI=0.5,
            mean_squared_error=0.5
        )}
    thresholds = [exp0[exp] for exp in exp0.keys()]
    list_metrics = list(exp0['exp0'].keys())
    eigenvalues = [0.01,0.1,0.2,0.5,1]
    for e in eigenvalues:
        config['eigen'] = e
        for thr in thresholds:
            config['threshold'] = thr
            # Run experiment
            experiment_name = str(datetime.now())[:-7]


            folder, thr_path = run_experiment_students(approaches=approaches,
                                                       config=config,
                                                       list_metrics=list_metrics,
                                                       experiment_name=experiment_name,
                                                       seeds=seeds,
                                                       scaling=config['scaling'],
                                                       verbose=verbose)

            plot_metrics(thr_path, folder, config['scaling'], calibration=True)

