import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import const_define as cd

plt.rcParams['font.size'] = 10
sns.set_style('whitegrid')


def plot_metrics(experiment_name: str, folder:str, scaling: str, calibration=False):
    # Path to existing records
    record_path = os.path.join(folder, 'records', experiment_name)
    assert os.path.isdir(record_path), f'Dir {record_path} does not exist!'

    # Loading records
    metrics_record = pd.read_pickle(os.path.join(record_path, 'metrics_record.pkl'))
    actions_record = pd.read_pickle(os.path.join(record_path, 'actions_record.pkl'))
    with open(os.path.join(record_path, 'config.json')) as f:
        config = json.load(f)
    print('Loaded records with config:')
    print(config)

    assert config['scaling'] == scaling, f"{scaling}!={config['scaling']}"

    # Images path
    imgpath_png = os.path.join(folder, 'images', experiment_name, 'png_imgs')
    imgpath_eps = os.path.join(folder, 'images', experiment_name, 'eps_imgs')
    if not os.path.isdir(imgpath_png):
        os.makedirs(imgpath_png)
    if not os.path.isdir(imgpath_eps):
        os.makedirs(imgpath_eps)

    metrics_to_plot = list(config['threshold'].keys())
    threshold = config['threshold']
    actions = config['actions']
    degree = 2


    # Define statistics
    ma_windows = (10, 20, 50)
    # Iterate over approaches
    statistics = {}
    w_beta_record = {}
    for approach in metrics_record:
        w_beta_record[approach] = {}
        statistics[approach] = {}
        # Iterate over metrics to compute statistics
        for metric in metrics_to_plot:
            statistics[approach][metric] = {
                'mean': [],
                'std': [],
            }
            for w in ma_windows:
                statistics[approach][metric][f'MA({w})_mean'] = []
            # Iterate over seeds
            seeds = list(metrics_record[approach][metric].keys())
            for seed in seeds:
                # Compute overall mean and std
                statistics[approach][metric]['mean'].append(np.mean(metrics_record[approach][metric][seed]))
                statistics[approach][metric]['std'].append(np.std(metrics_record[approach][metric][seed]))

                # Compute smoothing MA
                for w in ma_windows:
                    values_MA = pd.DataFrame(metrics_record[approach][metric][seed]).rolling(window=w).mean()
                    statistics[approach][metric][f'MA({w})_mean'].append(np.mean(values_MA[0]))

        # Compute statistics on actions
        statistics[approach]['Action_Smoothness'] = {
            'mean': [],
            'std': [],
        }

        if actions == 'group_weights':
            for seed in seeds:
                actions_list = actions_record[approach][seed]
                norm_list = []
                # Iterate over actions, the first batch is not optimized so we do not consider it
                for b in range(1, len(actions_list) - 1):
                    va = actions_list[b]
                    vb = actions_list[b + 1]
                    #norm = np.linalg.norm(actions_list[b] - actions_list[b + 1])
                    norm = 1 - np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-7)
                    norm_list.append(norm)
                statistics[approach]['Action_Smoothness']['mean'].append(np.mean(norm_list))
                statistics[approach]['Action_Smoothness']['std'].append(np.std(norm_list))

        elif actions == 'polynomial_fn':
            # We need to compute a fine discretization of the polynomial function
            z = np.linspace(0, 1, 1000)  # Generate 1000 escs points (from 0 to 1)
            w_beta_record[approach] = {}
            for seed in seeds:
                w_beta_record[approach][seed] = []
                actions_list = actions_record[approach][seed]
                w_beta_list = []
                # Compute W_beta for each batch (skipping first batch that is not optimized)
                for b in range(0, len(actions_list) - 1):
                    # Compute g
                    g = np.stack(
                        [z ** d - np.mean(z ** d) for d in np.arange(degree) + 1],
                        axis=1)  # We want g polynomial with zero mean
                    w_beta = g @ actions_list[b][:degree]
                    w_beta_list.append(w_beta)
                w_beta_record[approach][seed].append(w_beta_list)

                norm_list = []
                # Compute norm
                for b in range(1, len(w_beta_list) - 1):
                    va = w_beta_list[b]
                    vb = w_beta_list[b+1]
                    norm = 1 - np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-7)
                    norm_list.append(norm)
                statistics[approach]['Action_Smoothness']['mean'].append(np.mean(norm_list))
                statistics[approach]['Action_Smoothness']['std'].append(np.std(norm_list))
    # Save statistics
    fname = os.path.join(record_path, f'statistics.json')
    with open(fname, 'w') as f:
        json.dump(statistics, f)
        print(f'Saved records {fname}', flush=True)

    fname = os.path.join(record_path, f'statistics.txt')
    with open(fname, 'w') as f:
        for approach in statistics:
            f.write(f'{approach}, {threshold}\n')
            for metric in statistics[approach]:
                f.write(f'\t{metric}\n')
                for stat in statistics[approach][metric]:
                    assert len(statistics[approach][metric][stat]) == len(seeds) or len(
                        statistics[approach][metric][stat]) == 0
                    try:
                        mean = np.mean(statistics[approach][metric][stat])
                        std = np.std(statistics[approach][metric][stat])
                        if metric == 'Action_Smoothness':
                            f.write(f'\t\t{stat}: {round(mean, 6)} $\pm$ {round(std, 6)}\n')
                        else:
                            f.write(f'\t\t{stat}: {round(mean, 3)} $\pm$ {round(std, 3)}\n')
                    except:
                        f.write(f"\t\t{stat}: cannot compute\n")
        f.close()

    if not calibration:
        # Plot
        for seed in seeds:
            plot_normalized(metrics=metrics_record, threshold=threshold, metrics_to_plot=metrics_to_plot, seed_to_plot=seed,
                            imgpath_png=imgpath_png, imgpath_eps=imgpath_eps, scaling=scaling)
            plot_actions_heatmap(actions_record, w_beta_record=w_beta_record,
                                 seed_to_plot=seed, imgpath_png=imgpath_png, imgpath_eps=imgpath_eps, actions=actions)


def plot_normalized(metrics: dict, threshold: dict, metrics_to_plot: list, seed_to_plot: int, imgpath_eps: str,
                    imgpath_png: str, scaling: str):
    # Iterate over metrics to compute statistics
    for metric in metrics_to_plot:
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 6), dpi=130)
        # Iterate over approaches
        for approach in metrics:

            # Set approaches color and symbols for plot
            colors = 'orange' if approach == 'baseline' else 'green'
            markers = '^' if approach == 'baseline' else 'x'

            # Define current data
            values = metrics[approach][metric][seed_to_plot]

            # Plot values
            axes.plot(range(len(values)),
                      values,
                      marker=markers,
                      linestyle='dotted',
                      color=colors,
                      label=approach)

            if scaling == 'standardization':
                axes.set_ylim(-1.5, 1.5)
            elif scaling == 'normalization':
                axes.set_ylim(-0.2, 1.2)
            elif scaling == 'IQR_normalization':
                axes.set_ylim(-1.0, 2)
            elif scaling == 'None':
                continue
            else:
                print('Unknown scaling')
                raise ValueError

            # Compute mean
            mean = np.mean(values)
            axes.axhline(y=mean, color=colors, linewidth=2.5, linestyle='-',
                         label=f'Mean {approach}')

            if approach == 'fairdas':
                # Plot threshold once
                axes.axhline(y=threshold[metric], color='blue', linewidth=2.5, linestyle='-.', label='Threshold')
            axes.set_xlabel('Batches', size=15)
            axes.set_ylabel(f'{metric}', size=15)
            legend = axes.legend(loc='lower right', frameon=1, shadow=False, edgecolor="black")
            legend.get_frame().set_facecolor('white')
            legend.get_frame().set_alpha(0.8)

            axes.tick_params(axis='both', which='major', labelsize=15)
            axes.tick_params(axis='both', which='minor', labelsize=15)

        fname = f'{metric}_normalized_seed_{seed_to_plot}'
        plt.savefig(os.path.join(imgpath_eps, fname + '_normalized.eps'), format='eps',
                    bbox_inches="tight")
        fig.suptitle(fname)
        plt.savefig(os.path.join(imgpath_png, fname + '_normalized.png'), format='png',
                    bbox_inches="tight")
        plt.close('all')


def plot_actions_heatmap(action_record, w_beta_record:dict, seed_to_plot: int, imgpath_eps: str, imgpath_png: str, actions:str, mode:str='split'):


    if actions == 'group_weights':
        # Define data
        data_fairdas = np.stack(action_record['fairdas'][seed_to_plot]).T
        data_baseline = np.stack(action_record['baseline'][seed_to_plot]).T

        # Plot action vectors component wise
        if mode == 'component-wise':
            arr_list = []
            yticks = []
            for i in range(data_fairdas.shape[0]):
                arr_list.append(data_baseline[i])
                arr_list.append(data_fairdas[i])
                yticks.append('Baseline_' + rf'$\theta_{i + 1}$')
                yticks.append('FAiRDAS_' + rf'$\theta_{i + 1}$')
            data = np.stack([arr_list], axis=0).reshape(data_fairdas.shape[0] * 2, -1)

            # Plot heatmap
            ax = sns.heatmap(data, vmin=0, vmax=1, yticklabels=yticks)
            ax.set(xlabel="Batches", ylabel="")


            # Save plot
            fname = f'Actions_Heatmap_seed_{seed_to_plot}'
            plt.savefig(os.path.join(imgpath_eps, fname + '.eps'), format='eps',
                        bbox_inches="tight", dpi=300)
            plt.savefig(os.path.join(imgpath_png, fname + '.png'), format='png',
                        bbox_inches="tight", dpi=300)
            plt.close('all')

        # Plot action vectors separately
        elif mode == 'split':
            # Baseline
            yticks = []
            for i in range(data_fairdas.shape[0]):
                yticks.append('Baseline_' + rf'$\theta_{i + 1}$')
            # Plot heatmap
            ax = sns.heatmap(data_baseline, vmin=0, vmax=1, yticklabels=yticks)
            ax.set(xlabel="Batches", ylabel="")


            # Save plot
            fname = f'Actions_Heatmap_Baseline_seed_{seed_to_plot}'
            plt.savefig(os.path.join(imgpath_eps, fname + '.eps'), format='eps',
                        bbox_inches="tight", dpi=300)
            plt.savefig(os.path.join(imgpath_png, fname + '.png'), format='png',
                        bbox_inches="tight", dpi=300)
            plt.close('all')

            # FAiRDAS
            yticks = []
            for i in range(data_fairdas.shape[0]):
                yticks.append('FAiRDAS_' + rf'$\theta_{i + 1}$')
            # Plot heatmap
            ax = sns.heatmap(data_fairdas, vmin=0, vmax=1, yticklabels=yticks)

            ax.set(xlabel="Batches", ylabel="")


            # Save plot
            fname = f'Actions_Heatmap_FAiRDAS_seed_{seed_to_plot}'
            plt.savefig(os.path.join(imgpath_eps, fname + '.eps'), format='eps',
                        bbox_inches="tight", dpi=300)
            plt.savefig(os.path.join(imgpath_png, fname + '.png'), format='png',
                        bbox_inches="tight", dpi=300)
            plt.close('all')

    elif actions == 'polynomial_fn':
        # Define data
        data_fairdas = np.stack(w_beta_record['fairdas'][seed_to_plot]).T.squeeze()
        data_baseline = np.stack(w_beta_record['baseline'][seed_to_plot]).T.squeeze()

        # Baseline

        fig, ax = plt.subplots(figsize=(10, 6))
        # Plot heatmap
        sns.heatmap(data_baseline[::-1,:], ax=ax, vmin=0, vmax=1)
        ax.set(xlabel="Batches", ylabel="ESCS")
        plt.yticks([0, 500, 1000], [1, .5, 0], )


        # Save plot
        fname = f'Actions_Heatmap_Baseline_seed_{seed_to_plot}'
        plt.savefig(os.path.join(imgpath_eps, fname + '.eps'), format='eps',
                    bbox_inches="tight", dpi=300)
        plt.savefig(os.path.join(imgpath_png, fname + '.png'), format='png',
                    bbox_inches="tight", dpi=300)
        plt.close('all')


        fig, ax = plt.subplots(figsize=(10, 6))
        # Plot heatmap
        sns.heatmap(data_fairdas[::-1,:], ax =ax, vmin=0, vmax=1)

        ax.set(xlabel="Batches", ylabel="ESCS")
        plt.yticks([0, 500, 1000], [1, .5, 0], )

        # Save plot
        fname = f'Actions_Heatmap_FAiRDAS_seed_{seed_to_plot}'
        plt.savefig(os.path.join(imgpath_eps, fname + '.eps'), format='eps',
                    bbox_inches="tight", dpi=300)
        plt.savefig(os.path.join(imgpath_png, fname + '.png'), format='png',
                    bbox_inches="tight", dpi=300)
        plt.close('all')