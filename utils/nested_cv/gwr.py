import numpy as np
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW
from utils.model_utils import (
    separate_features,
    split_data,
    get_evaluation_metrics,
    get_optimal_hyperparameters,
)


def get_gwr_inputs(
    features, predictor_cols, labels, bandwidth=False, kernel=None, criterion=None
):
    coords = np.array(list(zip(features["x_coord"], features["y_coord"])))
    target = labels.values.reshape((-1, 1))
    predictors = np.hstack(
        [features[col].values.reshape((-1, 1)) for col in predictor_cols]
    )
    opt_bandwidth = None
    if bandwidth:
        opt_bandwidth = Sel_BW(coords, target, predictors, kernel=kernel).search(
            criterion=criterion
        )
    return coords, predictors, target, opt_bandwidth


def get_random_hyperparameters():
    kernel = np.random.choice(["bisquare", "Gaussian", "exponential"])
    criterion = np.random.choice(["AICc", "AIC", "BIC", "CV"])
    return kernel, criterion


def evaluate_gwr(df):
    outer_cv_results = []
    labels, outer_fold_ids, outer_splits, inner_fold_ids, inner_splits, features = (
        separate_features(df)
    )
    predictor_cols = features.columns

    # Outer cross-validation loop to evaluate model
    for current_outer_split in outer_splits:

        hp_combinations = []
        cv_results = []

        # Get outer cross-validation splits
        (
            outer_train_features,
            outer_train_labels,
            outer_val_features,
            outer_val_labels,
            current_inner_fold_ids,
        ) = split_data(
            current_outer_split,
            outer_fold_ids,
            features,
            labels,
            is_outer=True,
            inner_fold_ids=inner_fold_ids,
        )

        # Loop to test 8 hyperparameter combinations
        for i in range(8):

            # Set random hyperparameters
            kernel, criterion = get_random_hyperparameters()
            current_hps = {"kernel": kernel, "criterion": criterion}
            hp_combinations.append(current_hps)

            # Inner cross-validation for to select model
            for current_inner_split in inner_splits:
                print(
                    f"\n --- Outer split {current_outer_split} - Training model {i} on inner split {current_inner_split} ---"
                )

                # # Separate df into features and labels
                (
                    inner_train_features,
                    inner_train_labels,
                    inner_val_features,
                    inner_val_labels,
                    _,
                ) = split_data(
                    current_inner_split,
                    current_inner_fold_ids[
                        f"inner_loop_{current_inner_split + 1}_fold_id_python"
                    ],
                    outer_train_features,
                    outer_train_labels,
                )

                print("Getting inputs...")
                (
                    inner_train_coords,
                    inner_train_predictors,
                    inner_train_target,
                    inner_bandwidth,
                ) = get_gwr_inputs(
                    inner_train_features,
                    predictor_cols,
                    inner_train_labels,
                    bandwidth=True,
                    kernel=kernel,
                    criterion=criterion,
                )
                inner_val_coords, inner_val_predictors, inner_val_target, _ = (
                    get_gwr_inputs(inner_val_features, predictor_cols, inner_val_labels)
                )

                print("Building model...")
                model = GWR(
                    inner_train_coords,
                    inner_train_target,
                    inner_train_predictors,
                    bw=inner_bandwidth,
                    kernel=kernel,
                )

                model.fit()

                print("Getting predictions...")
                results = model.predict(inner_val_coords, inner_val_predictors)
                predictions = results.predy

                print("Evaluating predictions...")
                mae, mse, r2 = get_evaluation_metrics(inner_val_labels, predictions)

                # Add scores for current split to results
                cv_results.append(
                    {
                        "hp_combination": i,
                        "inner_split": current_inner_split,
                        "hps": current_hps,
                        "mae": mae,
                        "mse": mse,
                        "r2": r2,
                    }
                )

        print(
            f"\n --- Outer split {current_outer_split} - Training optimised model ---"
        )

        # Get optimal hyperparameters for current training set
        opt_hps = get_optimal_hyperparameters(hp_combinations, cv_results)

        print("Getting inputs...")
        (
            outer_train_coords,
            outer_train_predictors,
            outer_train_target,
            outer_bandwidth,
        ) = get_gwr_inputs(
            outer_train_features,
            predictor_cols,
            outer_train_labels,
            bandwidth=True,
            kernel=opt_hps["kernel"],
            criterion=opt_hps["criterion"],
        )
        outer_val_coords, outer_val_predictors, outer_val_target, _ = get_gwr_inputs(
            outer_val_features, predictor_cols, outer_val_labels
        )

        print("Building model...")
        model = GWR(
            outer_train_coords,
            outer_train_target,
            outer_train_predictors,
            bw=outer_bandwidth,
            kernel=opt_hps["kernel"],
        )

        model.fit()

        print("Getting predictions...")
        results = model.predict(outer_val_coords, outer_val_predictors)
        predictions = results.predy

        print("Evaluating predictions...")
        mae, mse, r2 = get_evaluation_metrics(outer_val_labels, predictions)

        outer_cv_results.append(
            {
                "outer_split": current_outer_split,
                "hps": opt_hps,
                "mae": mae,
                "mse": mse,
                "r2": r2,
                "inner_cv_results": cv_results,
            }
        )

    return outer_cv_results
