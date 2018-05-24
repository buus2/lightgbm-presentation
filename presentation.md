
# LightGBM
24.05.2018

## Table of contents
- 'Ordinary' training vs. Gradient Boosting Machine
- XGBoost vs. LightGBM
  - Boosting techniques
  - Basic learning control parameters
  - Observation sampling techniques
  - Feature sampling techniques
  - Optimizations for categorical features
  - Dealing with missing data
  - Dealing with big data
  - Parallelizations
  - Dealing with GPU
- Summary

## 'Ordinary' training vs. Gradient Boosting Machine

Assume that we have $n$ observations $(x_i)_{i=1}^n$, each storing values of $K$ features.

### 'Ordinary training'
| formula | meaning, examples |
|---|---|
| $M^0=M(\cdot;w^0)$ - initial model | a neural network with fixed structure and initial weights $w^0$ |
| For given $M^t$ we compute: | model after $t$ iterations|
| $\hat{y}_i^t=M^t(x_i)$ for each $i$ | predictions |
| $l(\hat{y}_i^t,y_i)$ for each $i$ | prediction errors, e.g. $l(\hat{y}_i^t,y_i)=\frac{1}{2}(\hat{y}_i^t-y_i)^2$ |
| $L(\hat{y}^t,y)=\frac{1}{n}\sum_{i=1}^nl(\hat{y}_i^t,y_i)$ | loss function, e.g. MSE for the above $l$|
| $g_j:=\partial L(\hat{y}^t,y)/\partial w_j$ for each $j$ | gradients |
| $w^{t+1}_j:=w^t_j-\eta g_j$ for each $j$ | SGB step with learning rate $\eta$ |
| $M^{t+1}:=M(\cdot;w^{t+1})$ | model after $t+1$ iterations |

### Gradient Boosting Machine
| formula | meaning, examples |
|---|---|
| $f^0=const.$ - initial learner | constant model |
| $M^0=f^0$ - initial model | constant model |
| For given $M^t=\sum_{j=0}^tf^j$ we compute: | model after $t$ iterations|
| $\hat{y}_i^t=M^t(x_i)$ for each $i$ | predictions |
| $l(\hat{y}_i^t,y_i)$ for each $i$ | prediction errors, e.g. $l(\hat{y}_i^t,y_i)=\frac{1}{2}(\hat{y}_i^t-y_i)^2$ |
| $g_i:=\partial l(\hat{y}^t_i,y_i)/\partial \hat{y}^t_i$ for each $i$ | gradients, e.g. $\hat{y}_i^t-y_i$ for the above $l$ |
| $\tilde{f}^{t+1}\simeq -g$ | we approximate $-g$ with use of a learner, e.g. a decision tree |
| $M^{t+1}:=M^t+\eta \tilde{f}^{t+1} = \sum_{j=0}^t f^j + f^{t+1}$ | model after $t+1$ iterations |

Gradient Boosting Machine (GBM) using decision trees as learner is called Gradient Boosting Decision Tree (GBDT).

General notes regarding GBDT:
- Smaller learning rate - more iterations.
- Trees with $r$ leaves are able to model interactions between at most $r-1$ features. Therefore e.g. if you bet that 5 features interact with each other, don't use trees with maximal depth 2.

#### Split finding

| formula | meaning, examples |
|---|---|
| $k=1,\ldots,K$ | features |
| $g_i$, $h_i$ | gradients and hessians, e.g. $g_i=\hat{y}_i^t-y_i$, $h_i=1$ for the square loss function|
| $I$ | indices of observations in the current node sorted by feature $k$ |
| $I_L$, $I_R$ | indices of observation going to the left and right node, respectively |
| $$\frac{(\sum_{i\in I_L}g_i)^2}{\sum_{i\in I_L}h_i} + \frac{(\sum_{i\in I_R}g_i)^2}{\sum_{i\in I_R}h_i} - \frac{(\sum_{i\in I}g_i)^2}{\sum_{i\in I}h_i}$$ | gain from the split |

## XGBoots vs. LightGBM

According to the LightGBM paper, there are two main innovations in LightGBM compared to XGBoost: GOSS and EFB.

List of parameters for XGBoost and LightGBM are taken from [here](https://github.com/dmlc/xgboost/blob/master/doc/parameter.md) and [here](http://lightgbm.readthedocs.io/en/latest/Parameters.html), respectively.

LightGBM paper (messy): https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf
Complete documentation of LightGBM: https://media.readthedocs.org/pdf/lightgbm/latest/lightgbm.pdf
Magnificent parameters comparison: https://sites.google.com/view/lauraepp/parameters
Comparison with Python snippets: https://towardsdatascience.com/catboost-vs-light-gbm-vs-xgboost-5f93620723db

Below I explain some non-obvious parameters.

### Boosting techniques

| XGBoost | LightGBM | meaning |
|---|---|---|
| `booster=` | `boosting=` | boosting technique |
| `gbtree` (default) | `gbdt` (default) | GBDT |
| `gblinear` | --- | linear learners |
| `dart` | `dart` | DART |
| --- | `goss` | GOSS |

#### DART - Dropouts meet Multiple Additive Regression Trees (XGBoost and LightGBM)

Takes a random subset of learners in each iteration.

Additional parameters:

| XGBoost | LightGBM |
|---|---|
| `sample_type` | `uniform_drop` |
| `rate_drop` | `drop_rate` |
| `skip_drop` | `skip_drop` |
| `normalize_type` | --- |
| `one_drop` | --- |
| --- | `max_drop` |
| --- | `xgboost_dart_mode` |
| --- | `drop_seed` |

Given $M^t$:
- Sample a subset $D$ from $\{1,\ldots,t\}$.
- Perform GBM iteration, but for $\tilde{M}^t=f^0+\sum_{j\in D}f^j$.
- Obtained $\tilde{f}^{t+1}$.
- Let $k=t-|D|$ (number of dropped learners).
  - XGBoost and LightGBM with `xgboost_dart_mode=True`: put $$M^{t+1}=\sum_{j\in D}f^j+\frac{k}{k+1}\sum_{j\notin D}f^j+\frac{1}{k+1}\tilde{f}^{t+1}.$$
  - LightGBM with `xgboost_dart_mode=False` (default): put $$M^{t+1}=\sum_{j\in D}f^j+\frac{k}{k+\eta}\sum_{j\notin D}f^j+\frac{\eta}{k+\eta}\tilde{f}^{t+1}.$$

#### GOSS - Gradient-based One-Side Sampling (LightGBM-exclusive)

Use sampled observations with greatest gradients and also some other obserations to build the next tree.

Additional parameters:

| LightGBM | meaning |
|---|---|
| `top_rate` (default `0.2`) | fraction of large gradient data |
| `other_rate` (default `0.1`) | fraction of nonlarge gradient data |

Given $M^t$:
- Compute gradients $g_i$ and hessians $h_i$ for each $i$.
- Let:
  - $a=$`top_rate`$\cdot n$ (number of sampled observations with large gradients),
  - $b=$`other_rate`$\cdot n$ (number of sampled observations with nonlarge gradients).
- Compute $g_i\cdot h_i$ for all $i$. For example, for the square loss function we have $g_i=\hat{y}_i^t-y_i$, $h_i=1$.
- Sample $a$ observations with greatest $g_i\cdot h_i$.
- Also sample $b$ other observations obtaining $(x_i)_{i\in S}$. Multiply their gradients and hessians by $(1-a)/b$
- Build the next tree $f^{t+1}$ with use of sampled observations $(x_i)_{i\in S}$ only.

### Basic learning control parameters

| XGBoost | LightGBM | meaning |
|---|---|---|
| `eta` (default `0.3`) | `learning_rate` (default `0.1`) | $\eta$, learning/shrinkage rate |
| `alpha` (default `0`) | `lambda_l1` (default `0`) | L1 regularization |
| `lambda` (default `0`) | `lambda_l2` (default `0`) | L2 regularization |
| `max_depth` (default `6`) | `max_depth` (default `-1`) | maximal depth of a tree |
| `gamma` (default `0`) | `min_split_gain` (default `0`) | minimal gain to perform split |
| ... | ... | ... |

### Observation sampling techniques

| XGBoost | LightGBM | meaning |
|---|---|---|
| `subsample` (default `1`) | `bagging_fraction` (default `1`) | part of observations used for building the tree |
| --- | `bagging_freq` | performs bagging every `bagging_freq` iterations |
| --- | `bagging_seed` | random seed |
| `max_bin` (default `256`) | `max_bin` (default `255`) | maximum number of histogram bins |
| `tree_method=` | --- | split algorithm |
| `auto` (default) | --- | automatic |
| `exact` | --- | all the split points checked |
| `approx` | --- | histogram algorithm, once per iteration |
| `hist` | default | histogram algorithm, once |
| `gpu_exact` | --- | `exact` for GPU |
| `gpu_hist` | --- | `hist` for GPU |
| --- | `boosting=goss` | GOSS |

### Feature sampling techniques

| XGBoost | LightGBM | meaning |
|---|---|---|
| `colsample_bytree` (default `1`) | `feature_fraction` (default `1`) | column sampling, once per tree |
| `colsample_bylevel` (default `1`) | --- | column sampling, once per node |
| --- | `feature_fraction_seed` | random seed |
| `enable_feature_grouping` (default `0`) | --- | ? |
| --- | `enable_bundle` (default `True`) | EFB |

#### EFB - Exclusive Feature Bundling

Bundles exclusive features into a single feature.

Note: bundles one-hot-encoded features into a single one, so one-hot-encoding is useless.

### Optimizations for categorical features

LightGBM offers some, see: `categorical_feature` and `min_data_per_group`, `max_cat_threshold`, `cat_smooth`, `cat_l2 max_cat_to_onehot`. 

### Dealing with missing data

| XGBoost | LightGBM | meaning |
|---|---|---|
| default | `use_missing` (default `True`) | column sampling, once per tree |
| --- | `zero_as_missing` (default `False`) | Treats `0` as missing value. |
| --- | `is_sparse` (default `True`) | Optimizations for sparse data. |

In XGBoost, missing values goes are ignored during split finding and afterwards all of them goes to either left or right node (the better one is chosen). For the default `use_missing=True` LightGBM does the same.

### Dealing with big data

| XGBoost | LightGBM | meaning |
|---|---|---|
| --- | `two_round` (default `False`) | whether to not load features from the data in memory |
When data is bigger than memory size, set `two_round=True`. Otherwise leave it as `False` for better speed.

### Parallelizations

| XGBoost | LightGBM | meaning |
|---|---|---|
| `nthread` | `num_threads` | number of threads |
| `updater` | `tree_learner` | the choice of tree learner |

- While setting `num_threads`, follow the number of real CPU cores, not logical ones.
- Do not set `num_threads` too large if your dataset is small (i.e. do not use 64 threads for a dataset with 10,000 rows).
- Be aware a task manager or any similar CPU monitoring tool might report cores not being fully utilized. This is normal.

For LightGBM, the default value of `tree_learner` is `serial`. There are also three parallel scenarios prepared:
| | #data is small | #data is large |
| --- | --- | --- |
| **#feature is small** | `feature` | `data` |
| **#feature is large** | `feature` | `voting` |
For parallel learning, you should not use full CPU cores since this will cause poor performance for the network.
See also: http://lightgbm.readthedocs.io/en/latest/Parallel-Learning-Guide.html

Both XGBoost and [LightGBM](http://lightgbm.readthedocs.io/en/latest/Installation-Guide.html#build-mpi-version) support MPI.

### Dealing with GPU

| XGBoost | LightGBM | meaning |
|---|---|---|
| `predictor=` | `device=` | choice of device |
| `cpu_predictior` (default) | `cpu` (default) | CPU |
| `gpu_predictor` | `gpu` | GPU |
| --- | `gpu_use_dp` (default `False`) | whether to use 64-bit float point |

- In LightGBM with `device=gpu` it is recommended to use the smaller `max_bin` (e.g. `63`) to get the better speed up.
- For better speed, GPU use 32-bit float point to sum up by default, it may affect the accuracy for some tasks. You can set `gpu_use_dp=True` to enable 64-bit float point, but it will slow down the training.

## Summary

- Try LightGBM because why not.
- Try `boosting=goss`.
- Sparse features are bundled automatically - `enable_bundle=True` by default. One-hot-encoding is useless.
- Specify categorical features: `categorical_feature=`.
- Leave the default `use_missing=True`. Set `is_sparse=False` if your data are dense.
- Set `two_round=True` only if your data does not fit in memory.
- If you have some real CPU cores, set `num_threads` properly.
- If your data are big, try parallelization scenarios.
- You can build GPU version of LightGBM. Then use smaller `max_bin`. Set `gpu_use_dp=True` only if the accuracy is worse than on CPU.
