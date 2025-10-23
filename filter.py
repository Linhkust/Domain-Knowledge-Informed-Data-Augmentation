import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import RANSACRegressor
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestRegressor
'''
RANSAC
'''


# 用真实数据RANSAC拟合的线性模型去判断合成数据的inlier和outlier
def synthetic_ransac(real, fake, features):
    X_real = real.loc[:, features].values
    y_real = real.iloc[:, -1].values

    reg = RANSACRegressor(random_state=0)

    reg.fit(X_real, y_real)

    X_fake = fake.loc[:, features]
    y_fake = fake.iloc[:, -1].values

    fake_y = reg.predict(X=X_fake)
    real_y = reg.predict(X=real.loc[:, features])

    residual_threshold = mean_absolute_error(y_true=real.iloc[:, -1].values, y_pred=real_y)

    residuals_subset = abs(y_fake - fake_y)
    inlier_mask = residuals_subset <= residual_threshold
    outlier_mask = residuals_subset > residual_threshold
    return fake.iloc[inlier_mask, :]


def synthetic_ransac_plot(real, fake, features, idx):
    points = synthetic_ransac(real, fake, features)

    plt.figure(figsize=[12, 12])
    plt.scatter(real.loc[:, features[idx]], real.iloc[:, -1], c='blue', marker='.', label='Data Points')
    plt.scatter(fake.loc[:, features[idx]], fake.iloc[:, -1], c='red', marker='.', label='Synthetic Points')
    plt.scatter(points.loc[:, features[idx]], points.iloc[:, -1], c='green', marker='*',
                label='Synthetic Points after RANSAC-based filter')
    plt.xlabel(features[idx])
    plt.ylabel('Strength')
    plt.legend()
    plt.show()


'''
Local Outsider Factor: LOF
'''


def synthetic_lof(real, fake, features):
    target = real.columns.tolist()[-1]
    X = real.loc[:, features + [target]].values
    s = fake.loc[:, features + [target]].values

    clf = LocalOutlierFactor(novelty=True, contamination=0.25)
    model = clf.fit(X)

    mask = (model.predict(s) == 1)
    return fake.iloc[mask, :]


def synthetic_lof_plot(real, fake, features, idx):
    points = synthetic_lof(real, fake, features)

    plt.figure(figsize=[12, 12])
    plt.scatter(real.loc[:, features[idx]], real.iloc[:, -1], c='blue', marker='.', label='Data Points')
    plt.scatter(fake.loc[:, features[idx]], fake.iloc[:, -1], c='red', marker='.', label='Synthetic Points')
    plt.scatter(points.loc[:, features[idx]], points.iloc[:, -1], c='green', marker='*',
                label='Synthetic Points after LOF-based filter')
    plt.xlabel(features[idx])
    plt.ylabel('Strength')
    plt.legend()
    plt.show()


'''
Isolation Forest
'''


def synthetic_if(real, fake, features):
    target = real.columns.tolist()[-1]

    X = real.loc[:, features + [target]].values
    s = fake.loc[:, features + [target]].values

    # 对离群点检测之后的真实数据进行训练
    clf_fake = IsolationForest(contamination=0.25)
    model = clf_fake.fit(X)

    mask = (model.predict(s) == 1)
    return fake.iloc[mask, :]


def synthetic_if_plot(real, fake, features, idx):
    points = synthetic_if(real, fake, feature)
    plt.figure(figsize=[12, 12])
    plt.scatter(real.loc[:, features[idx]], real.iloc[:, -1], c='blue', marker='.', label='Data Points')
    plt.scatter(fake.loc[:, features[idx]], fake.iloc[:, -1], c='red', marker='.', label='Synthetic Points')
    plt.scatter(points.loc[:, features[idx]], points.iloc[:, -1], c='green', marker='*',
                label='Synthetic Points after IF-based filter')
    plt.xlabel(features[idx])
    plt.ylabel('Strength')
    plt.legend()
    plt.show()

'''
New abnormal samples detect and removal procedures
'''
def _ransac(real, fake, feature_set):
    X, y = real.loc[:, [feature_set[0]]].to_numpy(), real.loc[:, [feature_set[1]]].to_numpy()

    reg = RANSACRegressor(random_state=0).fit(X=X,
                                              y=y)
    fake_X = fake.loc[:, [feature_set[0]]].to_numpy()
    y_fake = fake.loc[:, [feature_set[1]]].to_numpy()

    fake_y = reg.predict(X=fake_X)
    real_y = reg.predict(X=X)

    residual_threshold = mean_absolute_error(y_true=y, y_pred=real_y)

    residuals_subset = abs(y_fake - fake_y)
    inlier_mask = residuals_subset <= residual_threshold
    # outlier_mask = residuals_subset > residual_threshold
    return fake.iloc[inlier_mask, :]

def _lof(real, fake, feature_set):

    r = real.loc[:, feature_set].values
    s = fake.loc[:, feature_set].values

    clf = LocalOutlierFactor(novelty=True)
    model = clf.fit(r)

    mask = (model.predict(s) == 1)

    return fake.iloc[mask, :]

def _if(real, fake, feature_set):
    r = real.loc[:, feature_set].values
    s = fake.loc[:, feature_set].values

    clf_fake = IsolationForest()
    model = clf_fake.fit(r)

    mask = (model.predict(s) == 1)
    return fake.iloc[mask, :]

def detect_removal_pipeline(real, fake, method, corr_thresh):
    # find the paired variables with correlation coefficients larger than correlation threshold
    corrlation = real.corr()
    result = corrlation.stack().loc['Strength']
    matches = result[(abs(result) > corr_thresh) & (abs(result) < 1)]
    # feature_sets = [[idx, col] for (idx, col), value in matches.items()]
    feature_sets = [[idx, 'Strength'] for idx, value  in matches.items()]

    df = fake
    for feature_set in feature_sets:
        if method == 'ransac':
            df = _ransac(real, df, feature_set)
        elif method == 'lof':
            df = _lof(real, df, feature_set)
        elif method == 'if':
            df = _if(real, df, feature_set)
    return df.reset_index(drop=True)


def main():
    real = pd.read_csv('data.csv')
    fake = pd.read_csv('synthetic.csv')
    filter_fake = detect_removal_pipeline(real, fake, method='ransac', corr_thresh=0.3)
    print(filter_fake)


if __name__ == "__main__":
    main()

