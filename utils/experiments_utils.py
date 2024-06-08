import json
import os
import pickle
from collections import OrderedDict

import numpy as np

from utils.data_utils import Batch_Gen

np.seterr(all='raise')
from scipy import optimize
import matplotlib.pyplot as plt

import const_define as cd
from utils import metrics
from utils.dyn_systems import DynSystem
from utils.obj_functions import ObjFun_DynStateVar, ObjFun_BaselineSum, ObjFun_Normalization, ObjFun_Distancex0
from utils.constraints import poly_constraints, poly_constraints_thr, DEGREE




def store_records(metrics_record, rankings_record, actions_record, batches_records, modified_scores_records, opt_records, record_path):
    # Metrics
    fname = os.path.join(record_path, f'metrics_record.pkl')
    with open(fname, 'wb') as f:
        pickle.dump(metrics_record, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'Saved records {fname}', flush=True)

    # Ranking
    fname = os.path.join(record_path, f'ranking_record.pkl')
    with open(fname, 'wb') as f:
        pickle.dump(rankings_record, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'Saved records {fname}', flush=True)
    # Actions
    fname = os.path.join(record_path, f'actions_record.pkl')
    with open(fname, 'wb') as f:
        pickle.dump(actions_record, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'Saved records {fname}', flush=True)

    # Store batches
    fname = os.path.join(record_path, f'batches_record.pkl')
    with open(fname, 'wb') as f:
        pickle.dump(batches_records, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'Saved records {fname}', flush=True)

    # Store modified scores
    fname = os.path.join(record_path, f'modified_scores_record.pkl')
    with open(fname, 'wb') as f:
        pickle.dump(modified_scores_records, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'Saved records {fname}', flush=True)

    # Optimization
    fname = os.path.join(record_path, f'opt_record.pkl')
    with open(fname, 'wb') as f:
        pickle.dump(opt_records, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'Saved records {fname}', flush=True)
    f = open(os.path.join(record_path, f'opt_record.txt'), 'w')
    f.write(str(opt_records))
    f.close()

def optimizing_polynomial_fn(approach: str, approach_components: dict, thresholds: np.array, batch_scores: np.ndarray,
               batch_escs: np.ndarray, batch_escs_discr:np.ndarray, current_metrics: np.array, parameters: np.array, callback=None):

    solver = 'trust-constr'
    # Load current batch in the objective function
    if approach.lower() == 'fairdas':

        # Compute dynamical state
        x = approach_components['dyn_sys'](current_metrics)
        thr_check = x >= thresholds

        if thr_check.all():

            approach_components['obj_fun'].load_current_batch(batch_scores=batch_scores, batch_escs=batch_escs,
                                                              batch_escs_discr=batch_escs_discr, y=x)



            # Minimize
            res = optimize.minimize(approach_components['obj_fun'], x0=parameters,
                                    method=solver,
                                    constraints=poly_constraints,
                                    callback=callback,
                                    options={'maxiter': 100, 'disp': False})


            return res

        else:
            y = np.zeros_like(x)
            var_idx = []
            # Cycle over metrics
            for i in range(thr_check.shape[0]):
                # If state geq than threshold
                if thr_check[i]:
                    y[i] = x[i]
                # If state less than threshold
                else:
                    y[i] = -1
                    var_idx.append(i)

            # Load current batch in the objective function
            approach_components['obj_fun'].load_current_batch(batch_scores=batch_scores, batch_escs=batch_escs,
                                                              batch_escs_discr=batch_escs_discr, y=y)

            # Initial guess parameters
            initial_guess = np.zeros(len(parameters) + len(var_idx))
            initial_guess[:len(parameters)] = parameters
            for i, idx in enumerate(var_idx):
                initial_guess[len(parameters) + i] = x[idx]

            beta_bounds = ((-np.inf, np.inf),) * (DEGREE +1)
            # Add bound for y
            if approach_components['obj_fun'].scaling == 'normalization':
                bounds =  beta_bounds+ tuple([(0, thresholds[idx]) for idx in var_idx])
            elif approach_components['obj_fun'].scaling == 'standardization':
                bounds =  beta_bounds +tuple([(-5, thresholds[idx]) for idx in var_idx])
            elif approach_components['obj_fun'].scaling == 'IQR_normalization':
                bounds = beta_bounds+  tuple([(-1, thresholds[idx]) for idx in var_idx])
            else:
                bounds = beta_bounds + tuple([(-1, thresholds[idx]) for idx in var_idx])

            # Load var idx in obj fn
            approach_components['obj_fun'].var_idx = var_idx


            # Minimize
            res = optimize.minimize(approach_components['obj_fun'], x0=initial_guess,
                                    bounds=bounds, method=solver,
                                    constraints=poly_constraints_thr[len(var_idx)],
                                    callback=callback, options={'maxiter': 100, 'disp': False}, )

            return res


    elif approach.lower() == 'baseline':
        # Load current batch in the objective function
        approach_components['obj_fun'].load_current_batch(batch_scores=batch_scores, batch_escs=batch_escs,
                                                          batch_escs_discr=batch_escs_discr,
                                                          thresholds=thresholds)

        # Minimize
        res = optimize.minimize(approach_components['obj_fun'], x0=parameters, method=solver,
                                constraints=poly_constraints,
                                callback=callback, options={'maxiter': 100, 'disp': False})
        #assert res.success == True, res
        return res


def optimizing_group_weight(approach: str, approach_components: dict, thresholds: np.array, batch_scores: np.ndarray,
               batch_escs: np.ndarray, batch_escs_discr:np.ndarray, current_metrics: np.array, parameters: np.array, callback=None):

    solver = 'SLSQP'
    # Load current batch in the objective function
    if approach.lower() == 'fairdas':

        # Compute dynamical state
        x = approach_components['dyn_sys'](current_metrics)
        thr_check = x >= thresholds

        if thr_check.all():

            approach_components['obj_fun'].load_current_batch(batch_scores=batch_scores, batch_escs=batch_escs,
                                                              batch_escs_discr=batch_escs_discr, y=x)
            # Minimize
            res = optimize.minimize(approach_components['obj_fun'], x0=parameters,
                                    bounds=approach_components['bounds'], method=solver,
                                    constraints=approach_components['cons'],
                                    callback=callback, options={'maxiter': 100, 'disp': False})
            #assert res.success == True, res

        else:
            y = np.zeros_like(x)
            var_idx = []
            # Cycle over metrics
            for i in range(thr_check.shape[0]):
                # If state geq than threshold
                if thr_check[i]:
                    y[i] = x[i]
                # If state less than threshold
                else:
                    y[i] = -1
                    var_idx.append(i)

            # Load current batch in the objective function
            approach_components['obj_fun'].load_current_batch(batch_scores=batch_scores, batch_escs=batch_escs,
                                                              batch_escs_discr=batch_escs_discr, y=y)

            # Initial guess parameters
            initial_guess = np.zeros(len(parameters) + len(var_idx))
            initial_guess[:len(parameters)] = parameters
            for i, idx in enumerate(var_idx):
                initial_guess[len(parameters) + i] = x[idx]

            # Add bound for y
            if approach_components['obj_fun'].scaling == 'normalization':
                bounds = approach_components['bounds'] + tuple([(0, thresholds[idx]) for idx in var_idx])
            elif approach_components['obj_fun'].scaling == 'standardization':
                bounds = approach_components['bounds'] + tuple([(-5, thresholds[idx]) for idx in var_idx])
            elif approach_components['obj_fun'].scaling == 'IQR_normalization':
                bounds = approach_components['bounds'] + tuple([(-1, thresholds[idx]) for idx in var_idx])

            # Load var idx in obj fn
            approach_components['obj_fun'].var_idx = var_idx

            res = optimize.minimize(approach_components['obj_fun'], x0=initial_guess,
                                    bounds=bounds, method=solver,
                                    constraints=approach_components['cons'],
                                    callback=callback, options={'maxiter': 100, 'disp': False}, )
            #assert res.success == True, res


    elif approach.lower() == 'baseline':
        # Load current batch in the objective function
        approach_components['obj_fun'].load_current_batch(batch_scores=batch_scores, batch_escs=batch_escs,
                                                          batch_escs_discr=batch_escs_discr,
                                                          thresholds=thresholds)
        # Minimize
        res = optimize.minimize(approach_components['obj_fun'], x0=parameters,
                                bounds=approach_components['bounds'], method=solver,
                                constraints=approach_components['cons'],
                                callback=callback, options={'maxiter': 100, 'disp': False})
        #assert res.success == True, res
    return res


def build_approach(historical_data: np.ndarray, approach: str, matrix_A: np.ndarray, threshold: dict,
                   list_metrics: list, scaling: str, actions: str):
    """
    Grounds approach building blocks and normalizes metrics.
    """

    n_groups = 4

    approach_components = {}

    metrics_dict = OrderedDict()
    for metric_name in list_metrics:
        if metric_name == 'DIDI':
            # Creating DIDI metric object
            didi_metric = metrics.DIDI(actions=actions,impact_function=metrics.impact_function)
            metrics_dict['DIDI'] = didi_metric

        elif metric_name == 'GeDI':
            # Creating GeDI metric object
            gedi_metric = metrics.GeDI(actions=actions, degree=1) if actions == 'polynomial_fn' else metrics.GeDI(actions=actions)
            metrics_dict['GeDI'] = gedi_metric

        elif metric_name == 'score_absolute_error':
            # Create score distance object
            score_distance = metrics.MeanAbsoluteError(actions=actions)
            metrics_dict['score_absolute_error'] = score_distance

        elif metric_name == 'mean_squared_error':
            # Create score distance object
            score_distance = metrics.MeanSquaredError(actions=actions)
            metrics_dict['mean_squared_error'] = score_distance

        else:
            raise ValueError(f'Unknown metric name {metric_name}!')

    if actions == 'group_weights':
        # Define initial value of theta
        parameters = np.zeros(n_groups)
        # Define bounds for theta (theta is in (0,1))
        bounds = ((0., 1.),) * n_groups
        # Define theta constraint (theta sum = 1)
        cons = ({'type': 'eq', 'fun': lambda x: 1 - sum(x[:n_groups])})
        # Define n parameters
        n_params = n_groups
    elif actions == 'polynomial_fn':
        # Define initial value of beta as constant function
        parameters = np.array([1,0,0,0,0])[:DEGREE +1]
        # No bounds or constraints
        bounds = ()
        cons = ()
        # Define n parameters
        n_params = DEGREE+1


    # Find metrics minimum and maximum for normalization
    metrics_scaling_factors = metrics_scaling(metrics_dict=metrics_dict, historical_data=historical_data,
                                              parameters=parameters, bounds=bounds, cons=cons, scaling=scaling, actions=actions)

    if scaling == 'normalization':
        # Store min max values in Metric obj
        for name in metrics_scaling_factors:
            metrics_dict[name].max = metrics_scaling_factors[name]['max']
            metrics_dict[name].min = metrics_scaling_factors[name]['min']
            metrics_dict[name].scaling = scaling
    elif scaling == 'standardization':
        for name in metrics_scaling_factors:
            metrics_dict[name].mean = metrics_scaling_factors[name]['mean']
            metrics_dict[name].std = metrics_scaling_factors[name]['std']
            metrics_dict[name].scaling = scaling
    elif scaling == 'IQR_normalization':
        for name in metrics_scaling_factors:
            metrics_dict[name].q1 = metrics_scaling_factors[name]['q1']
            metrics_dict[name].q3 = metrics_scaling_factors[name]['q3']
            metrics_dict[name].scaling = scaling
    elif scaling == 'min_std':
        for name in metrics_scaling_factors:
            metrics_dict[name].min = metrics_scaling_factors[name]['min']
            metrics_dict[name].std = metrics_scaling_factors[name]['std']
            metrics_dict[name].scaling = scaling
    else:
        for name in metrics_dict:
            metrics_dict[name].scaling = False

    # The metrics weights are 1
    metrics_weight = {name: 1 for name in metrics_dict}


    # Define objective function & dynamics
    if approach.lower() == 'fairdas':
        thr = tuple(threshold[metric_name] for metric_name in list_metrics)
        dyn_sys = DynSystem(matrix_A, np.array(thr))
        obj_fun = ObjFun_DynStateVar(metrics_dict=metrics_dict, metrics_weight=metrics_weight, n_params=n_params,
                                     scaling=scaling)
        approach_components['dyn_sys'] = dyn_sys
    elif approach.lower() == 'baseline':
        obj_fun = ObjFun_BaselineSum(metrics_dict=metrics_dict, metrics_weight=metrics_weight, n_params=n_params,
                                     scaling=scaling)

    # Store

    approach_components['cons'] = cons
    approach_components['bounds'] = bounds
    approach_components['parameters'] = parameters
    approach_components['scaling_factors'] = metrics_scaling_factors
    approach_components['metrics_weight'] = metrics_weight
    approach_components['metrics_dict'] = metrics_dict
    approach_components['obj_fun'] = obj_fun

    return approach_components


def metrics_scaling(metrics_dict: dict, historical_data: dict, parameters: np.array,
                    bounds: tuple,
                    cons, scaling: str,
                    actions : str):
    """
    Extracts values for metrics scaling
    """

    # Compose a single historical batch
    # hist_batch = np.vstack(historical_data)
    # Metrics record
    metrics_record = {name: [] for name in metrics_dict}

    if scaling != 'None':
        # Obj fun for normalization
        obj_fun = ObjFun_Normalization(metrics_dict=metrics_dict,
                                       metrics_weight={name: 1 for name in metrics_dict},
                                       n_params=len(parameters),
                                       scaling='None')

        n_batches = len(historical_data['score'])
        for target_metric in metrics_dict:
            # Iterate over data
            for i in range(n_batches):
                scores = historical_data['score'][i]
                escs = historical_data['ESCS'][i]
                escs_discr = historical_data['ESCS_discretized'][i]
                # Compute current metrics value with previous step actions
                for name, metric_fn in metrics_dict.items():
                    metric = metric_fn(batch_scores=scores, batch_escs=escs,batch_escs_discr=escs_discr, parameters=parameters)
                    metrics_record[name].append(metric)

                # Optimize current target metric (if needed)
                if metrics_record[target_metric][-1] > 0.0:


                    obj_fun.load_current_batch(batch_scores=scores, batch_escs=escs, batch_escs_discr=escs_discr,target_metric=target_metric)

                    if actions == 'group_weights':
                        solver = 'SLSQP'
                        res = optimize.minimize(obj_fun, x0=parameters, bounds=bounds, method=solver, constraints=cons,
                                                callback=None, options={'maxiter': 100, 'disp': False})
                        #assert res.success == True, res
                    elif actions == 'polynomial_fn':
                        solver = 'trust-constr'
                        res = optimize.minimize(obj_fun, x0=parameters, method=solver, constraints=poly_constraints,
                                                callback=None, options={'maxiter': 100, 'disp': False, })

                        #assert res.success == True, res
                    # Update theta vector
                    parameters = res.x


        if scaling == 'normalization':
            # Extract minimum and maximum value for metrics
            metrics_min_max = {name: {'min': None, 'max': None} for name in metrics_record}
            for name in metrics_record:
                metrics_min_max[name]['min'] = min(metrics_record[name])
                metrics_min_max[name]['max'] = max(metrics_record[name])
                assert metrics_min_max[name]['min'] < metrics_min_max[name][
                    'max'], f"Max {name} ({metrics_min_max[name]['min']}) <= min {name} ({metrics_min_max[name]['max']})"

            return metrics_min_max

        elif scaling == 'standardization':
            # Extract mean and std value for metrics
            metrics_mean_std = {name: {'mean': None, 'std': None} for name in metrics_record}
            for name in metrics_record:
                metrics_mean_std[name]['mean'] = np.mean(metrics_record[name])
                metrics_mean_std[name]['std'] = np.std(metrics_record[name])

            if False:
                import matplotlib.pyplot as plt
                import seaborn as sns
                sns.histplot(metrics_record[name], bins=100)
                plt.axvline(np.mean(metrics_record[name]), 0, 60, color='green', linewidth=5)
                plt.axvline(np.mean(metrics_record[name]) + np.std(metrics_record[name]), 0, 60, color='green')
                plt.axvline(np.mean(metrics_record[name]) - np.std(metrics_record[name]), 0, 60, color='green')

            return metrics_mean_std

        elif scaling == 'IQR_normalization':
            # Extract q1 and q3 value for metrics
            metrics_q1_q3 = {name: {'q1': None, 'q3': None} for name in metrics_record}
            for name in metrics_record:
                metrics_q1_q3[name]['q1'] = np.quantile(metrics_record[name], 0.25)
                metrics_q1_q3[name]['q3'] = np.quantile(metrics_record[name], 0.75)

            if False:
                import matplotlib.pyplot as plt
                import seaborn as sns
                sns.histplot(metrics_record[name], bins=100)
                plt.axvline(np.quantile(metrics_record[name], 0.25), 0, 60, color='orange')
                plt.axvline(np.quantile(metrics_record[name], 0.75), 0, 60, color='orange')
            return metrics_q1_q3

        elif scaling == 'min_std':
            # Extract min and std value for metrics
            metrics_min_std = {name: {'min': None, 'std': None} for name in metrics_record}
            for name in metrics_record:
                metrics_min_std[name]['min'] = np.min(metrics_record[name])
                metrics_min_std[name]['std'] = np.std(metrics_record[name])

            return metrics_min_std

    else:
        print("No scaling performed!")
        return False


def create_records(approaches, list_metrics, seeds):

    # Records
    metrics_record = {a:
                          {m:
                               {seed: [] for seed in seeds}
                           for m in list_metrics + ['scaling_factors']}
                      for a in approaches}

    rankings_record = {a:
                           {seed: [] for seed in seeds}
                       for a in approaches}

    actions_record = {a:
                          {seed: [] for seed in seeds}
                      for a in approaches}

    modified_scores_records = {a:
                          {seed: [] for seed in seeds}
                      for a in approaches}

    batches_records = {seed: [] for seed in seeds}


    opt_records = {a:
                           {seed: [] for seed in seeds}
                       for a in approaches}

    return metrics_record, rankings_record, actions_record, modified_scores_records, batches_records, opt_records


def run_experiment(approaches: list, config: dict, list_metrics: list, experiment_name: str, seeds: list, scaling: str,
                   verbose: bool = False, ):
    alpha = config['alpha']
    eigen = config['eigen']
    matrix_A = np.eye(len(list_metrics), len(list_metrics)) * eigen
    threshold_dict = config['threshold']
    actions = config['actions']
    config['DEGREE'] = DEGREE

    # Define threshold
    current_thrs = tuple([threshold_dict[m] for m in list_metrics])

    record_path = os.path.join(cd.PROJECT_DIR, f"results_{config['actions']}",
                               'records', str(current_thrs), experiment_name)
    if not os.path.isdir(record_path):
        os.makedirs(record_path)

    # Save config file
    with open(os.path.join(record_path, 'config.json'), 'w',
              encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

    metrics_record, rankings_record, actions_record, modified_scores_records, batches_records, opt_records = create_records(
        approaches, list_metrics, seeds)



    # Iterate over seed
    for seed in seeds:
        print()
        print('*' * 40)
        print(f'Test with data generated with seed: {seed}')
        print('*' * 40)

        # Data Generator
        data_gen = Batch_Gen(batch_dim=config['batch_dim'], n_batches=config['n_batches'],
                             n_historical_batches=config['historical_batches'], seed=seed)
        n_groups = data_gen.n_groups
        historical_batches, batches = data_gen.generate_batches(data_gen.uniform_distr, data_gen.sampling_noise_distr,
                                                                alpha=alpha)

        print(f'----Test with thr: {current_thrs}')
        print(f'----Test with alpha: {alpha}')

        # Iterate over approaches
        for approach in approaches:
            print('----Approach:', approach, )

            # Set global seed
            cd.set_seed(4242)

            # Store queries and resources for current seed
            if batches_records[seed] == []:
                batches_records[seed] = batches

            # Otherwise check if the new queries and resources are equal to the stored ones
            else:
                assert (batches_records[seed]['score'] == batches['score']).all() and (
                        batches_records[seed]['ESCS'] == batches[
                    'ESCS']).all(), f'Mismatch between generated and stored resources for seed {seed}'

            # Build approach
            approach_components = build_approach(historical_batches, approach, matrix_A, threshold_dict, list_metrics,
                                                 scaling=scaling, actions=actions)
            parameters = approach_components['parameters']

            # Store scaling factor metrics
            metrics_record[approach]['scaling_factors'][seed] = approach_components['scaling_factors']

            # Iterate over queries batches
            n_batches = len(batches['score'])
            for i in range(n_batches):



                batch_scores = batches['score'][i]
                batch_escs = batches['ESCS'][i]
                batch_escs_discr = batches['ESCS_discretized'][i]


                for name in approach_components['metrics_dict']:
                    approach_components['metrics_dict'][name].folder = record_path

                # Compute metrics given actions found in previous step
                current_metrics = np.array([approach_components['metrics_dict'][name](batch_scores=batch_scores,
                                                                                      batch_escs=batch_escs,
                                                                                      batch_escs_discr=batch_escs_discr,
                                                                                      parameters=parameters) for
                                            name in approach_components['metrics_dict']])
                # Compute modified score for storage
                modified_scores = np.array([approach_components['metrics_dict'][name].modified_scores for
                                            name in approach_components['metrics_dict']])
                # Store modified scores
                modified_scores_records[approach][seed].append(modified_scores)

                # Store current metrics
                for idx, m in enumerate(list_metrics):
                    metrics_record[approach][m][seed].append(float(current_metrics[idx]))

                # Store current actions
                actions_record[approach][seed].append(parameters)


                if verbose and False:
                    print(f'\tMetrics batch {i}/{data_gen.n_batch_query - historical_batches}:', *current_metrics)

                # Check if metrics are above threshold
                thresholds_arr = np.array(current_thrs)
                check = np.all(current_metrics <= thresholds_arr)
                if not check:

                    callback = None

                    if actions == 'group_weights':
                        res = optimizing_group_weight(approach=approach, approach_components=approach_components,
                                                      thresholds=thresholds_arr, batch_scores=batch_scores,
                                                      batch_escs=batch_escs,
                                                      batch_escs_discr=batch_escs_discr,
                                                      current_metrics=current_metrics, parameters=parameters,
                                                      callback=callback)
                        parameters = res.x[:n_groups]
                    elif actions == 'polynomial_fn':
                        res = optimizing_polynomial_fn(approach=approach, approach_components=approach_components,
                                                       thresholds=thresholds_arr, batch_scores=batch_scores,
                                                       batch_escs=batch_escs,
                                                       batch_escs_discr=batch_escs_discr,
                                                       current_metrics=current_metrics, parameters=parameters,
                                                       callback=callback)
                        parameters = res.x[:DEGREE + 1]

                    if verbose:
                        opt_records[approach][seed].append(res)

            if actions == 'polynomial_fn':
                # plt.legend(range(n_batches))
                plt.title(f'{approach}')
                plt.savefig(os.path.join(record_path, f'{approach}_curves_{seed}.png'))
                plt.close('all')

    # Store results
    store_records(metrics_record, rankings_record, actions_record, batches_records, modified_scores_records, opt_records, record_path)


    return os.path.join(cd.PROJECT_DIR, f"results_{config['actions']}"), os.path.join(
        str(current_thrs), experiment_name)
