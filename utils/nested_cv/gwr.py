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


def build_and_evaluate_model(
    train_features,
    train_labels,
    val_features,
    val_labels,
    predictor_cols,
    kernel,
    criterion,
):
    print("Getting inputs...")
    (
        train_coords,
        train_predictors,
        train_target,
        bandwidth,
    ) = get_gwr_inputs(
        train_features,
        predictor_cols,
        train_labels,
        bandwidth=True,
        kernel=kernel,
        criterion=criterion,
    )
    val_coords, val_predictors, _, _ = get_gwr_inputs(
        val_features, predictor_cols, val_labels
    )

    print("Building model...")
    model = GWR(
        train_coords,
        train_target,
        train_predictors,
        bw=bandwidth,
        kernel=kernel,
    )

    model.fit()

    print("Getting predictions...")
    results = model.predict(val_coords, val_predictors)
    predictions = results.predy

    print("Evaluating predictions...")
    mae, mse, r2 = get_evaluation_metrics(val_labels, predictions)

    return mae, mse, r2


def evaluate_gwr(df, search_iterations):
    outer_cv_results = []
    labels, outer_fold_ids, outer_splits, inner_fold_ids, inner_splits, features = (
        separate_features(df)
    )
    predictor_cols = features.columns

    # Outer cross-validation loop to evaluate model
    for current_outer_split in outer_splits:

        hp_combinations = []
        inner_cv_results = []

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

        # Loop to test hyperparameter combinations
        for i in range(search_iterations):

            # Set random hyperparameters
            kernel, criterion = get_random_hyperparameters()
            current_hps = {"kernel": kernel, "criterion": criterion}
            hp_combinations.append(current_hps)

            # Inner cross-validation for to select model
            for current_inner_split in inner_splits:
                print(
                    f"\n --- Outer split {current_outer_split} - Training model {i} on inner split {current_inner_split} ---"
                )

                # Separate df into features and labels
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

                # Build and evaluate model on current inner split
                mae, mse, r2 = build_and_evaluate_model(
                    inner_train_features,
                    inner_train_labels,
                    inner_val_features,
                    inner_val_labels,
                    predictor_cols,
                    kernel,
                    criterion,
                )

                # Add scores for current inner split to results
                inner_cv_results.append(
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

        # Get optimal hyperparameters for current outer split
        opt_hps = get_optimal_hyperparameters(hp_combinations, inner_cv_results)

        # Build and evaluate model on current outer split
        mae, mse, r2 = build_and_evaluate_model(
            outer_train_features,
            outer_train_labels,
            outer_val_features,
            outer_val_labels,
            predictor_cols,
            opt_hps["kernel"],
            opt_hps["criterion"],
        )

        # Add scores for current outer split to results
        outer_cv_results.append(
            {
                "outer_split": current_outer_split,
                "hps": opt_hps,
                "mae": mae,
                "mse": mse,
                "r2": r2,
                "inner_cv_results": inner_cv_results,
            }
        )

    return outer_cv_results
