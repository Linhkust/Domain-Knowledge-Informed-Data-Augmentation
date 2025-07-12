import pandas as pd
import numpy as np
import torch
from matplotlib import pyplot as plt
from sdv.evaluation.single_table import evaluate_quality
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from filter import synthetic_ransac, synthetic_lof, synthetic_if
from sdv.single_table import CTGANSynthesizer
from sdv.single_table import CopulaGANSynthesizer
from sdv.single_table import TVAESynthesizer
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import Metadata
import itertools
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

np.random.seed(36)
torch.manual_seed(36)

'''
Tabular data augmentation methods
'''


def GaussianCopula(data,
                   ratio,
                   enforce_rounding=False,
                   enforce_min_max_values=True,
                   default_distribution='beta'):
    metadata = Metadata.detect_from_dataframe(
        data=data,
        table_name='concrete')

    synthesizer = GaussianCopulaSynthesizer(metadata=metadata,
                                            enforce_rounding=enforce_rounding,
                                            enforce_min_max_values=enforce_min_max_values,
                                            default_distribution=default_distribution)

    synthesizer.fit(data)
    synthetic_data = synthesizer.sample(int(len(data) * ratio))
    return synthetic_data


def CopulaGAN(data,
              ratio,
              epochs=300,
              enforce_rounding=False,
              enforce_min_max_values=True,
              default_distribution='beta'
              ):
    metadata = Metadata.detect_from_dataframe(
        data=data,
        table_name='concrete')
    synthesizer = CopulaGANSynthesizer(metadata=metadata,
                                       epochs=epochs,
                                       enforce_rounding=enforce_rounding,
                                       enforce_min_max_values=enforce_min_max_values,
                                       default_distribution=default_distribution)
    synthesizer.fit(data)
    synthetic_data = synthesizer.sample(int(len(data) * ratio))
    return synthetic_data


def CTGAN(data,
          ratio,
          epochs=300,
          enforce_rounding=False,
          enforce_min_max_values=True
          ):
    metadata = Metadata.detect_from_dataframe(
        data=data,
        table_name='concrete')
    synthesizer = CTGANSynthesizer(metadata=metadata,
                                   epochs=epochs,
                                   enforce_rounding=enforce_rounding,
                                   enforce_min_max_values=enforce_min_max_values)
    synthesizer.fit(data)
    synthetic_data = synthesizer.sample(int(len(data) * ratio))
    return synthetic_data


def TVAE(data,
         ratio,
         epochs=300,
         enforce_rounding=False,
         enforce_min_max_values=True):
    metadata = Metadata.detect_from_dataframe(data=data, table_name='concrete')
    synthesizer = TVAESynthesizer(metadata=metadata,
                                  epochs=epochs,
                                  enforce_rounding=enforce_rounding,
                                  enforce_min_max_values=enforce_min_max_values)
    synthesizer.fit(data)
    synthetic_data = synthesizer.sample(int(len(data) * ratio))
    return synthetic_data


# Evaluation metrics of synthetic data
def correlation_synthetic_evaluation(real_data, synthetic_data):
    metadata = Metadata.detect_from_dataframe(real_data)
    quality_report = evaluate_quality(
        real_data,
        synthetic_data,
        metadata,
        verbose=False)
    sdv_score = quality_report.get_score()

    real_corr = real_data.corr()['Strength'].values
    synthetic_corr = synthetic_data.corr()['Strength'].values

    # Normalization of cosine_similarity
    similarity_score = (cosine_similarity(real_corr.reshape(1, -1), synthetic_corr.reshape(1, -1))[0][0] + 1)/2

    return sdv_score, similarity_score, (sdv_score + similarity_score) / 2


'''
Hyperparameter tuning on synthesizer 
'''


# hyperparameter configurations to lists of parameter combinations
def generate_param_combinations(param_dict):
    keys = param_dict.keys()
    values = param_dict.values()
    combinations = itertools.product(*values)
    result = [dict(zip(keys, combo)) for combo in combinations]
    return result


def GaussianCopula_HP(real_data):
    configurations = {'ratio': [0.5, 1, 1.5],
                      'enforce_rounding': [True, False],
                      'enforce_min_max_values': [True, False],
                      'default_distribution': ['beta',
                                               'norm',
                                               'truncnorm',
                                               'uniform',
                                               'gamma',
                                               'gaussian_kde']
                      }

    hyperparam_combinations = generate_param_combinations(configurations)
    best_score = 0
    best_configuration = {}
    best_synthetic_data = pd.DataFrame()
    hp_log = []
    for hyperparam_combination in tqdm(hyperparam_combinations):
        synthetic_data = GaussianCopula(data=real_data,
                                        **hyperparam_combination)
        sdv, corr_s, fitness_score = correlation_synthetic_evaluation(real_data=real_data,
                                                                      synthetic_data=synthetic_data)
        hyperparam_combination['sdv_score'] = sdv
        hyperparam_combination['corr_similarity'] = corr_s
        hyperparam_combination['score'] = fitness_score

        hp_log.append(hyperparam_combination)
        if fitness_score > best_score:
            best_configuration = hyperparam_combination
            best_score = fitness_score
            best_synthetic_data = synthetic_data
        else:
            continue
    hp_log = pd.DataFrame(hp_log)
    return best_configuration, best_synthetic_data, best_score, hp_log


def CopulaGAN_HP(real_data):
    configurations = {'ratio': [0.5, 1, 1.5],
                      'enforce_rounding': [True, False],
                      'enforce_min_max_values': [True, False],
                      'epochs': [500, 1000]
                      }

    hyperparam_combinations = generate_param_combinations(configurations)
    best_score = 0
    best_configuration = {}
    best_synthetic_data = pd.DataFrame()
    hp_log = []
    for hyperparam_combination in tqdm(hyperparam_combinations):
        synthetic_data = CopulaGAN(data=real_data,
                                   **hyperparam_combination)
        sdv, corr_s, fitness_score = correlation_synthetic_evaluation(real_data=real_data,
                                                                      synthetic_data=synthetic_data)
        hyperparam_combination['sdv_score'] = sdv
        hyperparam_combination['corr_similarity'] = corr_s
        hyperparam_combination['score'] = fitness_score

        hp_log.append(hyperparam_combination)
        if fitness_score > best_score:
            best_configuration = hyperparam_combination
            best_score = fitness_score
            best_synthetic_data = synthetic_data
        else:
            continue
    hp_log = pd.DataFrame(hp_log)
    return best_configuration, best_synthetic_data, best_score, hp_log


def CTGAN_HP(real_data):
    configurations = {'ratio': [0.5, 1, 1.5],
                      'enforce_rounding': [True, False],
                      'enforce_min_max_values': [True, False],
                      'epochs': [500, 1000]
                      }

    hyperparam_combinations = generate_param_combinations(configurations)
    best_score = 0
    best_configuration = {}
    best_synthetic_data = pd.DataFrame()
    hp_log = []

    for hyperparam_combination in tqdm(hyperparam_combinations):
        synthetic_data = CTGAN(data=real_data,
                               **hyperparam_combination)
        sdv, corr_s, fitness_score = correlation_synthetic_evaluation(real_data=real_data,
                                                                      synthetic_data=synthetic_data)
        hyperparam_combination['sdv_score'] = sdv
        hyperparam_combination['corr_similarity'] = corr_s
        hyperparam_combination['score'] = fitness_score

        hp_log.append(hyperparam_combination)
        if fitness_score > best_score:
            best_configuration = hyperparam_combination
            best_score = fitness_score
            best_synthetic_data = synthetic_data
        else:
            continue
    hp_log = pd.DataFrame(hp_log)
    return best_configuration, best_synthetic_data, best_score, hp_log


def TVAE_HP(real_data):
    configurations = {'ratio': [0.5, 1, 1.5],
                      'enforce_rounding': [True, False],
                      'enforce_min_max_values': [True, False],
                      'epochs': [500, 1000]
                      }

    hyperparam_combinations = generate_param_combinations(configurations)
    best_score = 0
    best_configuration = {}
    best_synthetic_data = pd.DataFrame()
    hp_log = []
    for hyperparam_combination in tqdm(hyperparam_combinations):
        synthetic_data = CTGAN(data=real_data,
                               **hyperparam_combination)
        sdv, corr_s, fitness_score = correlation_synthetic_evaluation(real_data=real_data,
                                                                      synthetic_data=synthetic_data)
        hyperparam_combination['sdv_score'] = sdv
        hyperparam_combination['corr_similarity'] = corr_s
        hyperparam_combination['score'] = fitness_score

        hp_log.append(hyperparam_combination)
        if fitness_score > best_score:
            best_configuration = hyperparam_combination
            best_score = fitness_score
            best_synthetic_data = synthetic_data
        else:
            continue
    hp_log = pd.DataFrame(hp_log)
    return best_configuration, best_synthetic_data, best_score, hp_log


'''
Data augmentation without filters
'''


def synthesize_data_split(data,
                          target,
                          train_size=0.7,
                          random_state=0,
                          syn_method='ctGAN'):
    x_train, x_test, y_train, y_test = train_test_split(data.drop(target, axis=1),
                                                        data[target],
                                                        train_size=train_size,
                                                        random_state=random_state)
    real_train = pd.concat([x_train, y_train], axis=1).reset_index(drop=True)
    test = pd.concat([x_test, y_test], axis=1).reset_index(drop=True)

    print('Initiate data synthesizing without data filtering...')

    if syn_method == 'GaussianCopula':
        best_configuration, best_synthetic_data, best_score, hp_log = GaussianCopula_HP(real_train)
        return real_train, best_synthetic_data, test, hp_log

    elif syn_method == 'CTGAN':
        best_configuration, best_synthetic_data, best_score, hp_log = CTGAN_HP(real_train)
        return real_train, best_synthetic_data, test, hp_log

    elif syn_method == 'CopulaGAN':
        best_configuration, best_synthetic_data, best_score, hp_log = CopulaGAN_HP(real_train)
        return real_train, best_synthetic_data, test, hp_log

    elif syn_method == 'TVAE':
        best_configuration, best_synthetic_data, best_score, hp_log = TVAE_HP(real_train)
        return real_train, best_synthetic_data, test, hp_log


'''
Data augmentation with filters
'''


def filter_synthesize_data(real,
                           fake,
                           features,
                           filter_method='ransac'):
    print('Initiate data filtering...')

    synthetic_train_filter = pd.DataFrame()

    if filter_method == 'Ransac':
        synthetic_train_filter = synthetic_ransac(real=real, fake=fake, features=features)

    if filter_method == 'Lof':
        synthetic_train_filter = synthetic_lof(real=real, fake=fake, features=features)

    elif filter_method == 'If':
        synthetic_train_filter = synthetic_if(real=real, fake=fake, features=features)

    return synthetic_train_filter


# if __name__ == "__main__":
#     real_data = pd.read_csv('data.csv')
    # ds = real_data.describe().T
    # ds[['mean', 'std']] = ds[['mean', 'std']].round(2)
    # ds.to_csv('statistical_summary.csv')
    # print(real_data.kurt().round(2))
    # print(real_data.skew().round(2))
    # fitness_score = correlation_synthetic_evaluation(real_data=real_data,
    #                                                  synthetic_data=GaussianCopulaGAN(data=real_data,
    #                                                                                   ratio=0.5))
    # print(fitness_score)
    # score, configuration, syn_data, log = GaussianCopula_HP(real_data=real_data)
    # score, configuration, syn_data, log = CopulaGAN_HP(real_data=real_data)
    # score, configuration, syn_data, log  = CTGAN_HP(real_data=real_data)
    # score, configuration, syn_data, log  = TVAE_HP(real_data=real_data)
    # print(score)
    # print(configuration)
    # print(syn_data)
    # print(log)
