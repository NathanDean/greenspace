import numpy as np
import tensorflow.keras as keras
from utils.model_utils import (
    separate_features,
    split_data,
    get_evaluation_metrics,
    get_optimal_hyperparameters,
)


def build_model(
    train_features, no_of_layers, no_of_nodes, learning_rate, loss_function
):

    layers = []

    normaliser = keras.layers.Normalization(axis=-1)
    normaliser.adapt(np.array(train_features))
    layers.append(normaliser)

    for layer_no in range(no_of_layers):
        layers.append(keras.layers.Dense(no_of_nodes[layer_no], activation="relu"))

    layers.append(keras.layers.Dense(1))  # Single output for regression value

    model = keras.Sequential(layers)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss=loss_function
    )

    return model


def build_early_stopper():
    early_stopper = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=20, restore_best_weights=True
    )
    return early_stopper


def get_random_hyperparameters(features):
    no_of_layers = np.random.randint(1, 5)
    no_of_nodes = []
    for i in range(0, no_of_layers):
        no_of_nodes.append(
            np.random.randint((len(features.columns) / 2), (len(features.columns) * 2))
        )
    batch_size = np.random.randint(10, 64)
    learning_rate = np.random.uniform(0.0001, 0.1)
    loss_function = np.random.choice(["mae", "mse"])
    return no_of_layers, no_of_nodes, batch_size, learning_rate, loss_function


def evaluate_ann(df):

    outer_cv_results = []
    (
        labels,
        outer_fold_ids,
        outer_cv_splits,
        inner_fold_ids,
        inner_cv_splits,
        features,
    ) = separate_features(df)

    for current_outer_split in outer_cv_splits:

        hp_combinations = []
        cv_results = []

        # Get training and validation sets for current outer split
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

            # Get hyperparameters
            no_of_layers, no_of_nodes, batch_size, learning_rate, loss_function = (
                get_random_hyperparameters(features)
            )
            current_hps = {
                "outer_loop_split": current_outer_split,
                "no_of_layers": no_of_layers,
                "no_of_nodes": no_of_nodes,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "loss_function": loss_function,
            }
            hp_combinations.append(current_hps)

            # Inner cross-validation for model selection
            for current_inner_split in inner_cv_splits:
                print(
                    f"\n --- Outer split {current_outer_split}: Training model {i} on inner split {current_inner_split} ---"
                )

                # Get training and validation sets for current inner split
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

                # Build model
                model = build_model(
                    inner_train_features,
                    no_of_layers,
                    no_of_nodes,
                    learning_rate,
                    loss_function,
                )
                early_stopper = build_early_stopper()

                # Fit model
                model.fit(
                    inner_train_features,
                    inner_train_labels,
                    batch_size=batch_size,
                    epochs=200,
                    validation_split=0.2,
                    callbacks=[early_stopper],
                    verbose=0,
                )

                # Get predictions using fitted model
                predictions = model.predict(inner_val_features).flatten()

                # Get accuracy scores
                mae, mse, r2 = get_evaluation_metrics(inner_val_labels, predictions)

                # Add scores for current fold to results
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

        print(f"--- Outer split {current_outer_split}: Training on optimised model ---")

        # Get optimal hyperparameters for current outer split training set
        opt_hps = get_optimal_hyperparameters(hp_combinations, cv_results)
        opt_no_of_layers = opt_hps["no_of_layers"]
        opt_no_of_nodes = opt_hps["no_of_nodes"]
        opt_batch_size = opt_hps["batch_size"]
        opt_learning_rate = opt_hps["learning_rate"]
        opt_loss_function = opt_hps["loss_function"]

        # Build model
        model = build_model(
            outer_train_features,
            opt_no_of_layers,
            opt_no_of_nodes,
            opt_learning_rate,
            opt_loss_function,
        )
        early_stopper = build_early_stopper()

        # Fit model
        model.fit(
            outer_train_features,
            outer_train_labels,
            batch_size=opt_batch_size,
            epochs=200,
            validation_split=0.2,
            callbacks=[early_stopper],
            verbose=0,
        )

        # Get predictions using fitted model
        predictions = model.predict(outer_val_features).flatten()

        # Get accuracy scores
        mae, mse, r2 = get_evaluation_metrics(outer_val_labels, predictions)

        # Add scores for current fold to results
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
