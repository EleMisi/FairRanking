from collections import OrderedDict
from datetime import datetime
import argparse


import thr_students
from utils.plot_utils import plot_metrics as plot_metrics
from utils.experiments_utils import run_experiment as run_experiment_students



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run experiments')
    parser.add_argument('--actions', type=str,
                        help='Experiment class', default='discrete')

    args = parser.parse_args()


    if args.actions == 'discrete':
        actions = 'group_weights'
    elif args.actions == 'continuous':
        actions = 'polynomial_fn'
    else:
        argparse.ArgumentTypeError(f'invalid actions type: {args.actions}')


    # Approaches to test
    approaches = ['fairdas', 'baseline']


    verbose = True

    # Define data
    config = dict(
        historical_batches=30,
        n_batches=100,
        batch_dim=32,

        # alpha * Uniform + (1-alpha) * Sampling noise
        alpha=0.1,

        # Define dynamics
        eigen=0.2,

        # Define actions
        actions=actions,

        scaling='IQR_normalization'
     )
    # To seed data generation process
    seeds = [1234, 3245, 4242, 5627,6785, 8282, 9864, 9921]


    # Run experiment with only fairness metric
    if config['actions'] == 'group_weights':
         exp0 = {'exp0': OrderedDict(
             GeDI=0,
             score_absolute_error=2
         )}
    elif config['actions'] == 'polynomial_fn':
         exp0 = {'exp0': OrderedDict(
             GeDI=0.2,
             mean_squared_error=0.2
         )}
    thresholds = [exp0[exp] for exp in exp0.keys()]
    list_metrics = list(exp0['exp0'].keys())

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

         plot_metrics(thr_path, folder, config['scaling'])

    # Threshold configuration to test
    experiments = thr_students.thresholds
    thresholds = [experiments[config['actions']][exp] for exp in experiments[config['actions']].keys()]
    list_metrics = list(experiments[config['actions']]['exp2'].keys())


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
        plot_metrics(thr_path, folder, config['scaling'])

