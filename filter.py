import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import RANSACRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest

'''
RANSAC
'''


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

    clf_fake = IsolationForest(contamination=0.25)
    model = clf_fake.fit(X)

    mask = (model.predict(s) == 1)
    return fake.iloc[mask, :]


def synthetic_if_plot(real, fake, features, idx):
    points = synthetic_if(real, fake, features)
    plt.figure(figsize=[12, 12])
    plt.scatter(real.loc[:, features[idx]], real.iloc[:, -1], c='blue', marker='.', label='Data Points')
    plt.scatter(fake.loc[:, features[idx]], fake.iloc[:, -1], c='red', marker='.', label='Synthetic Points')
    plt.scatter(points.loc[:, features[idx]], points.iloc[:, -1], c='green', marker='*',
                label='Synthetic Points after IF-based filter')
    plt.xlabel(features[idx])
    plt.ylabel('Strength')
    plt.legend()
    plt.show()


# if __name__ == "__main__":
#     real = pd.read_csv('data.csv')
#     fake = pd.read_csv('synthetic_data.csv')

    # print(corr_matrix.loc[:, 'Strength'])
    # Cement 0.497832
    # Superplasticizer 0.366079
    # Age 0.328873

    # features = ['Cement', 'Superplasticizer', 'Age']
    # print(synthetic_ransac(real, fake, features))
    # synthetic_ransac_plot(real, fake, features, idx=0)

    # print(synthetic_lof(real, fake, features))
    # synthetic_lof_plot(real, fake, features, idx=2)

    # print(synthetic_if(real, fake, 'Cement'))
    # synthetic_if_plot(real, fake, features, idx=1)
