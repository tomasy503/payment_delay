# Import libraries
import argparse
import os

import lightgbm as lgb
import pandas as pd
from azureml.core import Run


# Evaluate the model
def evaluate_preds(y_true, y_preds, y_probs):
    """
    Performs evaluation comparison on y_true labels vs. y_preds labels on a classification.
    """
    accuracy = accuracy_score(y_true, y_preds)
    precision = precision_score(y_true, y_preds, average="macro")
    recall = recall_score(y_true, y_preds, average="macro")
    f1 = f1_score(y_true, y_preds, average="macro")
    metric_dict = {
        "accuracy": round(accuracy, 2),
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "f1": round(f1, 2),
    }

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1 score: {f1 * 100:.2f}%")

    # Confusion matrix
    confusion_matrix = pd.crosstab(
        y_true, y_preds, rownames=["Actual"], colnames=["Predicted"]
    )
    print("Confusion matrix:")
    print(confusion_matrix)

    # Evaluate per class metrics
    class_metrics = classification_report(y_true, y_preds, output_dict=True)
    print("Per class metrics:")
    for class_label, metrics in class_metrics.items():
        if isinstance(metrics, dict):  # Skip entries that are not dictionaries
            print(f"For class {class_label}:")
            print(f"\tPrecision: {metrics['precision'] * 100:.2f}%")
            print(f"\tRecall: {metrics['recall'] * 100:.2f}%")
            print(f"\tF1 score: {metrics['f1-score'] * 100:.2f}%")

    return metric_dict


if __name__ == "__main__":
    # Get parameters
    run = Run.get_context()

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_folder", type=str, dest="train_folder")
    parser.add_argument("--prediction_folder", type=str, dest="prediction_folder")

    args = parser.parse_args()

    os.makedirs(args.prediction_folder, exist_ok=True)

    df = pd.read_csv("{}/df_for_training.csv".format(args.train_folder))

    # df.set_index("index", inplace=True, drop=True)

    # Convert date columns to datetime
    date_columns = [
        "invoice_date",
        "create_date",
        "due_date",
        "due_discount_date",
        "payment_date",
    ]

    for col in date_columns:
        df[col] = pd.to_datetime(df[col])
        df = df[
            (df[col].dt.year >= 2010) & (df[col].dt.year <= dt.datetime.now().year)
            | df[col].isnull()
        ]

    # Since we are dealing with time series data, we will split the data based on the invoice date.
    # We will use data from 2017 to 2021 for training and data from 2022 onwards for testing.

    # sort the dataframe by 'invoice_date' and reset the index
    df = df.sort_values(by=["customer_id", "company_id", "invoice_date"]).reset_index(
        drop=True
    )

    # Define mask for training and test sets in real world scenario for production
    mask_train = (df["invoice_date"].dt.year >= 2019) & (
        df["invoice_date"].dt.year < 2023
    )

    mask_test = df["invoice_date"].dt.year >= 2023

    # Create training and test sets
    train_data = df[mask_train]
    test_data = df[mask_test]

    # Drop date columns as they are no longer needed
    columns_to_remove_before_training = [
        "due_date",
        "invoice_date",
        "create_date",
        "payment_date",
        "payment_delay",
    ]

    train_data.drop(columns_to_remove_before_training, axis=1, inplace=True)
    test_data.drop(columns_to_remove_before_training, axis=1, inplace=True)

    train_data["payment_bucket"] = train_data["payment_bucket"].astype(int)
    test_data["payment_bucket"] = test_data["payment_bucket"].astype(int)

    # define target variables
    train_target = train_data["payment_bucket"]
    test_target = test_data["payment_bucket"]

    # Drop the target variable from the training and test sets
    train_data.drop("payment_bucket", axis=1, inplace=True)
    test_data.drop("payment_bucket", axis=1, inplace=True)

    # Define the class weights
    class_weights = train_target.value_counts().to_dict()
    for key in class_weights.keys():
        class_weights[key] = len(train_target) / class_weights[key]

    # # Define mask for training and test sets in real world scenario for production
    # mask_train = (df["invoice_date"].dt.year >= 2020) & (df["payment_date"].notna())

    # mask_test = (df["payment_date"].isna()) & (df["payment_complete"] == 0)

    # search_spaces = {
    #     "learning_rate": Real(0.01, 0.05),  # around 0.03
    #     "max_depth": Integer(20, 30),  # around 25
    #     "n_estimators": Integer(1300, 1700),  # around 1500
    #     "num_leaves": Integer(14000, 18000),  # around 16000
    #     "subsample": Real(0.6, 0.8),  # around 0.7
    #     "subsample_freq": Integer(6, 10),  # around 8
    #     "reg_alpha": Real(0.5, 1.5),  # around 1
    #     "reg_lambda": Real(1.5, 2.5),  # around 2
    #     "min_child_samples": Integer(80, 100),  # around 90
    # }

    # sm = SMOTEENN(random_state=42)
    # train_data_sm, train_target_sm = sm.fit_resample(
    #     train_data, train_target)

    # search_spaces = {
    #     # Maximum tree leaves for base learners
    #     'num_leaves': Integer(5, 16000),
    #     # Maximum tree depth for base learners, <=0 means no limit
    #     'max_depth': Integer(1, 30),
    #     'n_estimators': Integer(10, 400),
    #     'learning_rate': Real(0.001, 1.0),
    #     'feature_fraction': Real(0.1, 0.9),
    #     'bagging_fraction': Real(0.8, 1.0),
    #     'min_split_gain': Real(0.001, 0.01),
    #     'min_child_weight': Integer(5, 200),
    #     # L2 regularization
    #     'reg_lambda': Real(0.000000000001, 3.0, 'log-uniform'),
    #     # L1 regularization
    #     'reg_alpha': Real(0.0000000000001, 5.0, 'log-uniform'),
    # }

    # model = lgb.LGBMClassifier(class_weight=class_weights)

    # bayes_search = BayesSearchCV(
    #     model, search_spaces, n_iter=32, n_jobs=32, cv=5)

    # bayes_search.fit(train_data, train_target)

    # best_params = bayes_search.best_params_

    # print("Best parameters:", best_params)

    # bayes_search = BayesSearchCV(
    #     estimator=model,
    #     search_spaces=search_spaces,
    #     cv=5,
    #     n_iter=60,  # max number of trials
    #     n_points=3,  # number of hyperparameter sets evaluated at the same time
    #     n_jobs=-1,  # number of jobs
    #     iid=False,  # if not iid it optimizes on the cv score
    #     return_train_score=False,
    #     refit=False,
    #     optimizer_kwargs={
    #         "base_estimator": "GP"
    #     },  # optmizer parameters: we use Gaussian Process (GP)
    #     random_state=0,
    # )

    # best_params = bayes_search.fit(
    #     train_data_sm, train_target_sm).best_params_

    # print(best_params)

    model = lgb.LGBMClassifier(
        objective="multiclass",
        metric="multiclass",
        n_jobs=-1,
        verbose=-1,
        random_state=0,
        class_weight=class_weights,
        learning_rate=0.022550720790521145,
        max_depth=20,
        min_child_samples=88,
        n_estimators=1580,
        num_leaves=16917,
        reg_alpha=0.513841643356537,
        reg_lambda=2.1667566844499695,
        subsample=0.7859753700282479,
        subsample_freq=8,
    )

    model.fit(train_data, train_target)

    y_preds = model.predict(test_data)

    y_probs = model.predict_proba(test_data)

    metrics = evaluate_preds(y_true=test_target, y_preds=y_preds, y_probs=y_probs)

    print(metrics)

    # test_data["real_bucket"] = test_target
    test_data["predicted_bucket"] = y_preds

    test_data.to_csv(
        os.path.join(args.prediction_folder, "predictions.csv"), index=False
    )
