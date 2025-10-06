import numpy as np
import libpysal.weights as weights
import pysal.explore as esda
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def separate_features(df):
    # Drop R columns
    df = df.drop(columns = [col for col in df.columns if "fold_id_r" in col])
    
    # Dependent variables
    labels = df.pop('very_good_health')

    # Outer CV folds
    outer_fold_ids = df["outer_loop_fold_id_python"]
    outer_splits = np.sort(outer_fold_ids.unique().astype(int))

    # Inner CV folds
    inner_fold_ids = df[[col for col in df.columns if "inner_loop" in col]]
    inner_splits = np.sort(inner_fold_ids.stack().unique().astype(int))

    # Independent variables
    features = df.drop(columns = [col for col in df.columns if "fold_id" in col])
    features["x_coord"] = features["geometry"].centroid.x
    features["y_coord"] = features["geometry"].centroid.y
    features = features.drop(columns = ["geometry"])

    return labels, outer_fold_ids, outer_splits, inner_fold_ids, inner_splits, features


def split_data(current_split, fold_ids, features, labels, is_outer = False, inner_fold_ids = None):
    is_in_validation_set = fold_ids == current_split
    is_in_training_set = ~is_in_validation_set
    train_features = features.loc[is_in_training_set]
    train_labels = labels.loc[is_in_training_set]
    val_features = features.loc[is_in_validation_set]
    val_labels = labels.loc[is_in_validation_set]
    current_inner_fold_ids = None
    if is_outer:
        current_inner_fold_ids = inner_fold_ids.loc[is_in_training_set]
    return train_features, train_labels, val_features, val_labels, current_inner_fold_ids

def get_ann_random_hyperparameters(features):
    no_of_layers = np.random.randint(1, 5)
    no_of_nodes = []
    for i in range(0, no_of_layers):
        no_of_nodes.append(np.random.randint((len(features.columns) / 2), (len(features.columns) * 2)))
    learning_rate = np.random.uniform(0.0001, 0.1)
    return no_of_layers, no_of_nodes, learning_rate

def get_gwr_random_hyperparameters():
    kernel = np.random.choice(["bisquare", "Gaussian", "exponential"])
    criterion = np.random.choice(["AICc", "AIC", "BIC", "CV"])
    return kernel, criterion    

def get_evaluation_metrics(val_features, val_labels, predictions):
    mae = mean_absolute_error(val_labels, predictions)
    mse = mean_squared_error(val_labels, predictions)
    r2 = r2_score(val_labels, predictions)
    return mae, mse, r2

def get_avg_scores(cv_results):
    mae_scores = []
    mse_scores = []
    r2_scores = []

    for result in cv_results:
        mae_scores.append(result["mae"])
        mse_scores.append(result["mse"])
        r2_scores.append(result["r2"])

    avg_mae = np.mean(mae_scores)
    avg_mse = np.mean(mse_scores)
    avg_r2 = np.mean(r2_scores)

    return avg_mae, avg_mse, avg_r2

def get_optimal_hyperparameters(hp_combinations, cv_results):
    hp_combination_scores = []
    for i in range(len(hp_combinations)):
        current_hp_combination_results = [result for result in cv_results if result["hp_combination"] == i]
        mae, mse, r2 = get_avg_scores(current_hp_combination_results)
        hp_combination_scores.append(mse)
    optimal_combination = np.argmin(hp_combination_scores)
    optimal_hps = hp_combinations[optimal_combination]
    return optimal_hps