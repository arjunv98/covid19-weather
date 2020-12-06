##########
# REQUIRES XGBOOST- https://anaconda.org/conda-forge/xgboost
##########
train_models = False

# Forecasting Coronavirus:
# Developing a time-invariant model to predict outbreaks or subsidence of COVID-19 in the United States
# Arjun Viswanathan  
# arjunvis@usc.edu

import numpy as np
import pandas as pd
import time
import pickle

from modules import CovidDataset
from utils import get_Xy

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, truncnorm, uniform, loguniform

# --- NECESSARY TO FUNCTION (USES XGBREGRESSOR AND XGBFREGRESSOR MODELS)
import xgboost as xgb
# ---

# set start date at March 15, 2020 (first Sunday after announcement of global pandemic on March 11) and set end date to October 31, 2020 (exactly 33 weeks)
start_date = pd.Timestamp('2020-03-15')
end_date = pd.Timestamp('2020-10-31')


# %%
# load covid dataset and mapper class (mapper used to create choropleth maps)
d = CovidDataset(start_date, end_date)

num_cols = ['mean_temp', 'min_temp', 'max_temp', 'dewpoint', 'precipitation']
num_cols_pct = [s + "_pct" for s in num_cols]
bool_cols = ['fog', 'rain', 'snow', 'hail', 'thunder', 'tornado']
cat_cols = ['region']

# Clean data
d.clean_data()

# feature processing transformer
# encode regions to one hot and scale numerical imputs
feature_engineering = ColumnTransformer(
    transformers = [
        ('numerical', StandardScaler(), (num_cols + num_cols_pct)),
        ('categorical', OneHotEncoder(sparse=False), cat_cols)
    ], remainder = 'passthrough'
)

# do principal component analysis to reduce dimensionality after feature engineering adds dimensions
min_explained_variance = 0.95 # want at least 95% of the variance explained
X_pt, y_pt = get_Xy(d.get_rolling_data('pretraining', period=1))
pca = PCA()
p = Pipeline(
    steps = [
        ('feature_engineering', feature_engineering)
    ]
)
X_transform = p.fit_transform(X_pt)
X_pca = pca.fit_transform(X_transform)

total_explained_variance = pca.explained_variance_ratio_.cumsum()
opt_num_cols = len(total_explained_variance[total_explained_variance >= min_explained_variance])
opt_cols = X_transform.shape[1] - opt_num_cols + 1


# PIPELINES
# basic parameters
pca_opt = PCA(n_components=opt_num_cols)
verbose = 5
cv = 3
xgb_verbosity = 1
iters = 80
scoring = {'-MSE': 'neg_mean_squared_error', 'R2': 'r2'}
refit = 'R2'

# BASELINE: No PCA, Linear Regression
base_param_grid = {}
base_pipe = Pipeline([
    ('feature_engineering', feature_engineering),
    ('reduce_dim', 'passthrough'),
    ('model', LinearRegression())
])
base_model = RandomizedSearchCV(estimator=base_pipe, param_distributions=base_param_grid, scoring=scoring, refit=refit, n_jobs=-1, cv=cv, n_iter=1, verbose=verbose)


# PCA: Different num of PCA columns, Linear Regression
pca_param_grid = {
    'reduce_dim__n_components': randint(2, 15)
}
pca_pipe = Pipeline([
    ('feature_engineering', feature_engineering),
    ('reduce_dim', PCA()),
    ('model', LinearRegression())
])
pca_model = RandomizedSearchCV(estimator=pca_pipe, param_distributions=pca_param_grid, scoring=scoring, refit=refit, n_jobs=-1, cv=cv, n_iter=10, verbose=verbose)


# RIDGE: Ridge regression with and without PCA
ridge_param_grid = {
    'reduce_dim': ['passthrough', pca_opt],
    'model__alpha': loguniform(1e-2, 1e1)
}
ridge_pipe = Pipeline([
    ('feature_engineering', feature_engineering),
    ('reduce_dim', 'passthrough'),
    ('model', Ridge())
])
ridge_model = RandomizedSearchCV(estimator=ridge_pipe, param_distributions=ridge_param_grid, scoring=scoring, refit=refit, n_jobs=-1, cv=cv, n_iter=iters, verbose=verbose)


# ELASTIC NET: Elastic Net regression with and without PCA
en_param_grid = {
    'reduce_dim': ['passthrough', pca_opt],
    'model__alpha': loguniform(1e-2, 1e1),
    'model__l1_ratio': uniform(0, 1)
}
en_pipe = Pipeline([
    ('feature_engineering', feature_engineering),
    ('reduce_dim', 'passthrough'),
    ('model', ElasticNet())
])
en_model = RandomizedSearchCV(estimator=en_pipe, param_distributions=en_param_grid, scoring=scoring, refit=refit, n_jobs=-1, cv=cv, n_iter=iters, verbose=verbose)


# RANDOM FOREST: Random Forest regression using XGBoost algorithm, with and without PCA
rf_param_grid = {
    'reduce_dim': ['passthrough', pca_opt],
    'model__min_child_weight': randint(1, 20),
    'model__gamma': truncnorm(a=0, b=5, loc=0.5, scale=0.5),
    'model__subsample': uniform(0.3, 0.7),
    'model__colsample_bynode': uniform(0.3, 0.6),
    'model__max_depth': randint(3,6)
}
rf_pipe = Pipeline([
    ('feature_engineering', feature_engineering),
    ('reduce_dim', 'passthrough'),
    ('model', xgb.XGBRFRegressor(objective="reg:squarederror", nthread=-1, random_state=42, verbosity=xgb_verbosity))
])
rf_model = RandomizedSearchCV(estimator=rf_pipe, param_distributions=rf_param_grid, scoring=scoring, refit=refit, n_jobs=-1, cv=cv, n_iter=iters, verbose=verbose)


# GRADIENT BOOST: Gradient Boosted regression using XGBoost algorithm, with and without PCA
gb_param_grid = {
    'reduce_dim': ['passthrough', pca_opt],
    'model__learning_rate': loguniform(1e-2, 1e0),
    'model__min_child_weight': randint(1, 10),
    'model__gamma': truncnorm(a=0, b=5, loc=0.5, scale=0.5),
    'model__subsample': uniform(0.5, 0.5),
    'model__colsample_bytree': uniform(0.6, 0.4),
    'model__max_depth': randint(3, 6)
}
gb_pipe = Pipeline([
    ('feature_engineering', feature_engineering),
    ('reduce_dim', 'passthrough'),
    ('model', xgb.XGBRegressor(objective="reg:squarederror", nthread=-1, random_state=42, verbosity=xgb_verbosity))
])
gb_model = RandomizedSearchCV(estimator=gb_pipe, param_distributions=gb_param_grid, scoring=scoring, refit=refit, n_jobs=-1, cv=cv, n_iter=iters, verbose=verbose)

# all models in a dict
models = {
    'base': base_model,
    'pca': pca_model,
    'ridge': ridge_model,
    'en': en_model,
    'rf': rf_model,
    'gb': gb_model
}

# MAIN TRAINING LOOP (skip with train_models = False)
if train_models:
    results = {}
    for m in models:
        results[m] = {}
    rolling_period = np.arange(1, 8)

    start_time = time.time()
    for s in rolling_period:
        print("-------------------------\nLOADING TRAINING SET {}...\n-------------------------".format(s))
        # get training and validation sets
        X_tr, y_tr = get_Xy(d.get_rolling_data('training', period=s))
        X_val, y_val = get_Xy(d.get_rolling_data('validation', period=s))

        for m in models:
            print("RUNNING MODEL ~{}~ ON TRAINING SET {}...".format(m.upper(),s))
            model = models[m]
            # fit model and append to results
            model.fit(X_tr, y_tr)
            results[m][s] = {'cv_results_': model.cv_results_, 'best_params_': model.best_params_, 'best_score_': model.best_score_, 'best_estimator_': model.best_estimator_}
            
            # test on validation set and append to results
            y_true, y_pred = y_val, model.predict(X_val)
            results[m][s]['Eval'] = mean_squared_error(y_true, y_pred)

    total_time = time.gmtime(time.time() - start_time)
    print("-----------------------------------------------\n-----------------------------------------------\n-----------------------------------------------")
    print("TOTAL ELAPSED TIME: ", time.strftime("%H:%M:%S", total_time))

    # get best results for each model (1 estimator for each model out of 7 possible each)
    for m in results:
        _m = results[m]
        minError = max(v['best_score_'] for k,v in _m.items())
        best_estimator_key = [k for k,v in _m.items() if v['best_score_'] == minError][0]
        results[m]['best'] = results[m][best_estimator_key]
        results[m]['best']['period'] = best_estimator_key

    # save to pickle
    pickle.dump(results, open("results.p", "wb"))

# load results and models
results = pickle.load(open("results.p", "rb"))

# get validation errors for 6 best models
print("\nValidation errors on each model:")
val_errors = {}
for m in results:
    val_errors[m] = results[m]['best']['Eval']
    print("Estimator: {}\t :: Eval = {}".format(m, val_errors[m]))

# get single best model from validation errors
best_estimator = min(val_errors, key=val_errors.get)

# get test errors
test_errors = {}
for m in ['base', best_estimator]:
    best_period = results[m]['best']['period']
    best_model = results[m]['best']['best_estimator_']
    X_test, y_test = get_Xy(d.get_rolling_data('testing', period=best_period))
    y_true, y_pred = y_test, best_model.predict(X_test)
    test_errors[m] = mean_squared_error(y_true, y_pred)

# print test error against baseline
print("\nTest error for best model (compared against baseline):")
for t in test_errors:
    print("Estimator: {}\t :: Etest = {}".format(t, test_errors[t]))
