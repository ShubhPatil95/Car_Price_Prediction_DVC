import joblib
import os
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from get_data import read_file
from sklearn import metrics
import pandas as pd
import json
from sklearn.model_selection import RandomizedSearchCV



def train_test_evaluates(paths_path,params_path):
    config_params=read_file(params_path)
    config_paths=read_file(paths_path)
    processed_data=config_paths["data"]["processed_data"]

    df=pd.read_csv(processed_data,index_col=0)
    ## Data Selection(dependent and independent)
    X=df.iloc[:,1:]
    y=df.iloc[:,0]
    print(X.head())
    #TRAIN TEST SPLIT
    random_state=config_params['base']["random_state"]
    split_ratio=config_params["base"]["split_ratio"]
    print(split_ratio)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=split_ratio,random_state=random_state)

    #model_creation
    rf=RandomForestRegressor()

    #Randomized Search CV
    start_e=config_params["random_cv"]["n_estimators"]["start"]
    stop_e=config_params["random_cv"]["n_estimators"]["stop"]
    num=config_params["random_cv"]["n_estimators"]["num"]
   
    max_features=config_params["random_cv"]["max_features"]

    start_d=config_params["random_cv"]["max_depth"]["start"]
    stop_d=config_params["random_cv"]["max_depth"]["stop"]
    num_d=config_params["random_cv"]["max_depth"]["num"]
    
    min_samples_split=config_params["random_cv"]["min_samples_split"]

    min_samples_leaf = config_params["random_cv"]["min_samples_leaf"]

    n_estimators = [int(x) for x in np.linspace(start = start_e, stop = stop_e, num = num)]
    max_depth = [int(x) for x in np.linspace(start=start_d,stop=stop_d, num = num_d)]

    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf}

    n_iter=config_params["random_cv"]["n_iter"]
    cv=config_params["random_cv"]["cv"]

    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error',
                               n_iter = n_iter, cv = cv, verbose=2, random_state=random_state, n_jobs = 1)

    rf_random.fit(X_train,y_train)

    predictions=rf_random.predict(X_test)

    MAE = metrics.mean_absolute_error(y_test, predictions)
    MSE = metrics.mean_squared_error(y_test, predictions)
    RMSE = np.sqrt(metrics.mean_squared_error(y_test, predictions))

    print('MAE', MAE)
    print('MSE', MSE)
    print('RMSE', RMSE)

    scores_file = config_paths["reports"]["scores"]
    params_file = config_paths["reports"]["params"]

    with open(scores_file, "w") as f:
        scores = {
            "MAE": MAE,
            "MSE": MSE,
            "RMSE": RMSE
        }
        json.dump(scores, f, indent=4)

    with open(params_file, "w") as f:
        params = rf_random.best_params_
        json.dump(params, f, indent=4)

    model_dir=config_paths["model_dir"]
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.joblib")

    joblib.dump(rf_random, model_path)

if __name__=="__main__":
    args=argparse.ArgumentParser()
    args.add_argument("--config_paths",default="paths.yaml")
    args.add_argument("--config_params",default="params.yaml")
    parsed_args=args.parse_args()
    train_test_evaluates(parsed_args.config_paths,parsed_args.config_params)
