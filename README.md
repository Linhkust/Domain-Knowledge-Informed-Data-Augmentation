# Domain Knowledge Informed Data Augmentation
>Data and codes for the Paper "Leveraging domain knowledge in data augmentation to boost concrete strength prediction accuracy with automated machine learning and deep learning"
1. The `data.csv` is collected from https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength
2. The `data_aug.py` provides the hyperparameter optimization process of four data augmentation methods: GaussianCopula, CTGAN, CopulaGAN, and TVAE. The generation of initial and finalized synthetic data is presented in the script.
3. The `filter.py` defines three anomaly detection methods: RANSAC, IF, and LOF.
4. The `automl.py` defines the AutoML and AutoDL frameworks.
5. The `prediction.py` defines the whole experimental process.
