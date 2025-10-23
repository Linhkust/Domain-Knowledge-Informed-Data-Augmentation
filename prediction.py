# -*- coding: UTF-8 -*-
import os
import pandas as pd
import data_aug as da
from automls import _mjar_, _flaml_,  _pycaret_, _tabpfn_, _autogluon_
import warnings
import gc

warnings.filterwarnings('ignore')


# 生成：真实训练集和测试集，初始合成数据集
def real_syn_data_generate(data,
                           target,
                           train_size,
                           random_state,
                           syn_method):
    # Use synthesized data
    real_train, synthetic_train, test, hp_log = da.synthesize_data_split(data, target,
                                                                         train_size=train_size,
                                                                         random_state=random_state,
                                                                         syn_method=syn_method)

    return real_train, test, synthetic_train, hp_log


def data_saved(corr_thresh):
    real_data = pd.read_csv('data.csv', encoding='utf-8')
    random_states = [12, 45, 78, 3, 56, 89, 23, 67, 34, 91, 5, 28, 72, 19, 60, 84, 37, 50, 7, 95]

    for random_state in random_states:
        # ['GaussianCopula', 'CTGAN', 'CopulaGAN', 'TVAE']
        for syn_method in ['GaussianCopula', 'CTGAN', 'CopulaGAN', 'TVAE']:
            real_train, test, synthetic_train, hp_log = real_syn_data_generate(data=real_data,
                                                                       target='Strength',
                                                                       random_state=random_state,
                                                                       syn_method=syn_method,
                                                                       train_size=0.7)

            for filter_method in ['ransac', 'lof', 'if']:
                #
                filter_synthetic_train = da.filter_synthesize_data(real_train,
                                                                   synthetic_train,
                                                                   filter_method,
                                                                   corr_thresh)
                filter_synthetic_train.to_csv(
                    f'./data_hp_0924/filter_synthetic_train_{random_state}_{syn_method}_{filter_method}.csv', index=False)


def data_saved_sampling(corr_thresh):
    real_data = pd.read_csv('data.csv', encoding='utf-8')
    # random_states = [12, 45, 78, 3, 56, 89, 23, 67, 34, 91, 5, 28, 72, 19, 60, 84, 37, 50, 7, 95]
    random_states = [91, 5, 28, 72, 19, 60, 84, 37, 50, 7, 95]
    for frac in [0.7]:
        # create the directory
        os.makedirs(f'./data_hp_sampling_0924/{frac}', exist_ok=True)
        for random_state in random_states:
            for syn_method in ['GaussianCopula', 'CTGAN', 'CopulaGAN', 'TVAE']:
                real_train, test, synthetic_train, hp_log = real_syn_data_generate(data=real_data.sample(frac=frac, random_state=0),
                                                                           target='Strength',
                                                                           random_state=random_state,
                                                                           syn_method=syn_method,
                                                                           train_size=0.7)
                # data saved
                real_train.to_csv(f'./data_hp_sampling_0924/{frac}/real_train_{random_state}.csv', index=False)
                test.to_csv(f'./data_hp_sampling_0924/{frac}/test_{random_state}.csv', index=False)
                synthetic_train.to_csv(f'./data_hp_sampling_0924/{frac}/synthetic_train_{random_state}_{syn_method}.csv', index=False)
                # hp_log.to_csv(f'./hp_log_0924/hp_log_{random_state}_{syn_method}.csv', index=False)

                for filter_method in ['ransac', 'lof', 'if']:
                    filter_synthetic_train = da.filter_synthesize_data(real_train,
                                                                       synthetic_train,
                                                                       filter_method,
                                                                       corr_thresh)
                    filter_synthetic_train.to_csv(
                        f'./data_hp_sampling_0924/{frac}/filter_synthetic_train_{random_state}_{syn_method}_{filter_method}.csv',
                        index=False)

def fit_predict(train,
                test,
                target,
                train_method='TabPFN'):

    # if train_method == 'TPOT':
    #     return _tpot_(train, test, target)

    # elif train_method == 'H2O':
    #     return _h2o_(train, test, target)

    if train_method == 'FLAML':
        return _flaml_(train, test, target)

    elif train_method == 'mjar_supervised':
        return _mjar_(train, test, target)

    elif train_method == 'AutoGluon':
        return _autogluon_(train, test, target)

    elif train_method == 'PyCaret':
        return _pycaret_(train, test, target)

    elif train_method == 'TabPFN':
        return _tabpfn_(train, test, target)

    # elif train_method == 'AutoPyTorch':
    #     return _auto_pytorch_(train, test, target)

    # elif train_method == 'auto-sklearn':
    #     return _auto_sklearn_(train, test, target)

    # elif train_method == 'lr':
    #     return _lr_(train, test, target)



# train_method = ['TPOT', 'H20', 'FLAML', 'mjar_supervised', 'TabPFN'， ‘PyCaret', 'AutoGluon', 'auto-sklearn']
# auto-sklearn和AutoPytorch需要Ubuntu系统
class main_experiment(object):
    def __init__(self, random_states, import_path, output_path):
        self.random_states = random_states
        self.import_path=import_path
        self.output_path=output_path

    def scenario_1_benchmark(self, train_method, target):

        performances = []
        for random_state in self.random_states:
            real_train = pd.read_csv(f'./{self.import_path}/real_train_{random_state}.csv', encoding='unicode_escape')
            test = pd.read_csv(f'./{self.import_path}/test_{random_state}.csv', encoding='unicode_escape')

            '''Scenario 1: real_train + real test (RTRT)'''
            performance_s1 = fit_predict(train=real_train,
                                         test=test,
                                         target=target,
                                         train_method=train_method)

            performance_s1['scenario'] = 'Baseline'
            performance_s1['random state'] = random_state
            performance_s1['model'] = train_method
            performances.append(performance_s1)
        return performances

    def scenario_2_experiment(self, train_method, target):
        performances = []
        for random_state in self.random_states:
            real_train = pd.read_csv(f'./{self.import_path}/real_train_{random_state}.csv', encoding='unicode_escape')
            test = pd.read_csv(f'./{self.import_path}/test_{random_state}.csv', encoding='unicode_escape')
            # for syn_method in ['GaussianCopula']:
            for syn_method in ['GaussianCopula', 'CTGAN', 'CopulaGAN', 'TVAE']:
                synthetic_train = pd.read_csv(
                    f'./{self.import_path}/synthetic_train_{random_state}_{syn_method}.csv', encoding='unicode_escape')

                # real_train + synthetic_train
                syn_train = pd.concat([real_train, synthetic_train], axis=0)

                performance_syn = fit_predict(train=syn_train, test=test, target=target, train_method=train_method)

                del synthetic_train, syn_train
                gc.collect()

                # log
                performance_syn['scenario'] = 'RST'
                performance_syn['random state'] = random_state
                performance_syn['model'] = train_method
                performance_syn['syn_method'] = syn_method

                performances.append(performance_syn)
        return performances

    def scenario_3_experiment(self, train_method, target):
        performances = []
        for random_state in self.random_states:
            # real train and test
            real_train = pd.read_csv(f'./{self.import_path}/real_train_{random_state}.csv', encoding='unicode_escape')
            test = pd.read_csv(f'./{self.import_path}/test_{random_state}.csv', encoding='unicode_escape')
            for syn_method in ['GaussianCopula', 'CTGAN', 'CopulaGAN', 'TVAE']:
                for filter_method in ['ransac', 'lof', 'if']:
                    filter_synthetic_train = pd.read_csv(
                        f'./{self.import_path}/filter_synthetic_train_{random_state}_{syn_method}_{filter_method}.csv',
                        encoding='unicode_escape')

                    # real_train + filter_synthetic_train
                    filter_syn_train = pd.concat([real_train, filter_synthetic_train], axis=0)
                    if len(filter_synthetic_train) != 0:
                        performance_syn_filter = fit_predict(train=filter_syn_train,
                                                             test=test,
                                                             target=target,
                                                             train_method=train_method)
                    else:
                        performance_syn_filter={}

                    del filter_synthetic_train, filter_syn_train
                    gc.collect()

                    # log
                    performance_syn_filter['scenario'] = 'RCST'
                    performance_syn_filter['random state'] = random_state
                    performance_syn_filter['model'] = train_method
                    performance_syn_filter['syn_method'] = syn_method
                    performance_syn_filter['filter_method'] = filter_method
                    performances.append(performance_syn_filter)
        return performances


    def run(self, train_method, target):
        print('Training under scenario RT...')
        performances_1 = self.scenario_1_benchmark(train_method, target)

        print('Training under scenario RST...')
        performances_2 = self.scenario_2_experiment(train_method, target)

        print('Training under scenario RCST...')
        performance_3 = self.scenario_3_experiment(train_method, target)

        performances = performances_1 + performances_2 + performance_3
        summary = pd.DataFrame(performances)
        summary.to_csv(f'./{self.output_path}/{train_method}_summary.csv', index=False)

if __name__ == "__main__":
    # 更改train_method
    # train_method = ['TPOT', 'H20', 'FLAML', 'mjar_supervised', 'TabPFN'， ‘PyCaret', 'AutoGluon', 'auto-sklearn', 'AutoPytorch']
    # train_experiments(train_method='TabPFN', target='Strength')
    # data_saved(corr_thresh=0.3)
    # data_saved_sampling(corr_thresh=0.3)

    random_states = [12, 45, 78, 3, 56, 89, 23, 67, 34, 91, 5, 28, 72, 19, 60, 84, 37, 50, 7, 95]
    # method= ['FLAML', 'mjar_supervised','AutoGluon','PyCaret', 'TabPFN']
    for frac in [0.1, 0.3, 0.5, 0.7]:
        print(frac)
        os.makedirs(f'./results_sampling/{frac}', exist_ok=True)
        main_run = main_experiment(random_states=random_states,
                                   import_path=f'data_hp_sampling_0924/{frac}',
                                   output_path=f'./results_sampling/{frac}')
        main_run.run(train_method='FLAML', target='Strength')
