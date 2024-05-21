# Import libraries
import argparse
import datetime as dt
import os

import numpy as np
import pandas as pd
from azureml.core import Run
from sklearn.linear_model import LinearRegression

# We will create special features to determine payment behavior of the customers


def calculate_trend(x):
    """Fit a linear regression and return the slope."""
    if len(x) > 1:
        model = LinearRegression()
        model.fit(np.arange(len(x)).reshape(-1, 1), x)
        return model.coef_[0]
    else:
        return 0


# Calculate feature for payment behaviour
def calculate_features(df):
    # Special situation where a discount or alternative due date is given
    df["discount_offered"] = df["due_discount_date"].apply(
        lambda x: 0 if pd.isna(x) else 1
    )

    # Remove the due_discount_date column
    df.drop("due_discount_date", axis=1, inplace=True)

    # Calculate 'payment_delay'
    df["payment_delay"] = (df["payment_date"] - df["due_date"]).dt.days

    # Group by customer and company
    grouped = df.groupby(["customer_id", "company_id"])

    # Previous payment delay - initialize to 0 for first transaction
    df["previous_payment_delay"] = grouped["payment_delay"].shift()

    df["previous_payment_delay"].fillna(0, inplace=True)

    # Average payment delay until current invoice - initialize to 0 for first transaction
    df["average_payment_delay"] = grouped["payment_delay"].transform(
        lambda x: x.expanding().mean()
    )

    df["average_payment_delay"].fillna(0, inplace=True)

    # Count of previous invoices until current invoice
    df["number_of_previous_invoices"] = grouped.cumcount()

    # Total amount of previous invoices
    df["total_amount_of_previous_invoices"] = grouped["net_amount"].transform(
        lambda x: x.expanding().sum()
    )

    # Average amount of previous invoices
    df["average_invoice_amount"] = df.apply(
        lambda row: (
            0
            if row["number_of_previous_invoices"] == 0
            else row["total_amount_of_previous_invoices"]
            / row["number_of_previous_invoices"]
        ),
        axis=1,
    )

    # Standard deviation in the payment delay - initialize to 0 for first transaction
    df["payment_deviation"] = grouped["payment_delay"].transform(
        lambda x: x.expanding().std()
    )
    df["payment_deviation"].fillna(0, inplace=True)

    # Standard deviation in the invoice amounts - initialize to 0 for first transaction
    df["invoice_deviation"] = grouped["net_amount"].transform(
        lambda x: x.expanding().std()
    )
    df["invoice_deviation"].fillna(0, inplace=True)

    # Has there been any previous delay in payment in the last 3 invoices
    df["has_delayed_payment_last_3"] = (
        grouped["payment_delay"].transform(lambda x: x.rolling(window=3).max()) > 0
    ).astype(int)

    # Has there been any previous delay in payment in the last 3 invoices
    df["has_early_payment_last_3"] = (
        grouped["payment_delay"].transform(lambda x: x.rolling(window=3).max()) < 0
    ).astype(int)

    # Calculate trend of payment delay over time for each customer and company
    df["payment_delay_trend"] = (
        grouped["payment_delay"]
        .transform(lambda x: x.rolling(window=3).mean().diff())
        .fillna(0)
    )

    # Calculate trend of invoice amount over time for each customer and company
    df["invoice_amount_trend"] = (
        grouped["net_amount"]
        .transform(lambda x: x.rolling(window=3).mean().diff())
        .fillna(0)
    )

    # Calculate maximum payment delay until current invoice
    df["maximum_payment_delay"] = grouped["payment_delay"].transform(
        lambda x: x.expanding().max()
    )

    # Overall payment delay trend for each customer and company
    df["overall_payment_delay_trend"] = grouped["payment_delay"].transform(
        calculate_trend
    )

    return df


# We will calculate some time related features to understand seasonality effects on the payment behavior of the customers


def calculate_trend_features(df):
    # Convert to datetime if it's not already
    df["due_date"] = pd.to_datetime(df["due_date"])

    # Features for the time of the year
    df["due_date_month"] = df["due_date"].dt.month
    df["due_date_quarter"] = df["due_date"].dt.quarter
    df["due_date_dayofweek"] = df["due_date"].dt.dayofweek

    df["due_date_is_year_end"] = (df["due_date"].dt.month == 12).astype(int)
    df["due_date_is_year_start"] = (df["due_date"].dt.month == 1).astype(int)

    # Due date is in summer time (June, July, August)
    df["due_date_is_summer"] = df["due_date"].dt.month.isin([6, 7, 8]).astype(int)

    # Group by customer and company
    grouped = df.groupby(["customer_id", "company_id"])

    # Rate of change of the delay (second derivative)
    df["payment_delay_trend_acceleration"] = (
        grouped["payment_delay"].transform(lambda x: x.diff().diff()).fillna(0)
    )

    # Variance in payment behavior over time
    df["payment_delay_variance"] = (
        grouped["payment_delay"].transform(lambda x: x.expanding().std()).fillna(0)
    )

    return df


# We will calculate some business related features to understand the business behavior of the customers and summarize them in a global score


def calculate_risk_tolerance_score(df):
    # Get the necessary data
    average_payment_delay = df["average_payment_delay"]
    has_delayed_payment_last_3 = df["has_delayed_payment_last_3"]
    maximum_payment_delay = df["maximum_payment_delay"]
    overall_payment_delay_trend = df["overall_payment_delay_trend"]
    number_of_previous_invoices = df["number_of_previous_invoices"]
    invoice_deviation = df["invoice_deviation"]
    net_amount = df["net_amount"]
    average_invoice_amount = df["average_invoice_amount"]
    # discount_offered = df["discount_offered"]

    # Normalize these values for scoring
    # In this case, I'll assume higher value indicates higher risk for delay related features
    # and higher value indicates lower risk for value related features
    # For binary variables like `has_delayed_payment_last_3`, no normalization is needed.
    # Inverse because higher delay should lower the score
    average_payment_delay_score = 1 - (
        (average_payment_delay - df["average_payment_delay"].min())
        / (df["average_payment_delay"].max() - df["average_payment_delay"].min())
    )
    maximum_payment_delay_score = 1 - (
        (maximum_payment_delay - df["maximum_payment_delay"].min())
        / (df["maximum_payment_delay"].max() - df["maximum_payment_delay"].min())
    )
    overall_payment_delay_trend_score = 1 - (
        (overall_payment_delay_trend - df["overall_payment_delay_trend"].min())
        / (
            df["overall_payment_delay_trend"].max()
            - df["overall_payment_delay_trend"].min()
        )
    )
    invoice_deviation_score = (invoice_deviation - df["invoice_deviation"].min()) / (
        df["invoice_deviation"].max() - df["invoice_deviation"].min()
    )
    number_of_previous_invoices_score = (
        number_of_previous_invoices - df["number_of_previous_invoices"].min()
    ) / (
        df["number_of_previous_invoices"].max()
        - df["number_of_previous_invoices"].min()
    )
    net_amount_score = (net_amount - df["net_amount"].min()) / (
        df["net_amount"].max() - df["net_amount"].min()
    )
    average_invoice_amount_score = (
        average_invoice_amount - df["average_invoice_amount"].min()
    ) / (df["average_invoice_amount"].max() - df["average_invoice_amount"].min())
    # discount_offered_score = (discount_offered - df["discount_offered"].min()) / (
    #     df["discount_offered"].max() - df["discount_offered"].min()
    # )

    # Combine these factors into a single risk tolerance score
    # The weights can be adjusted based on business requirements.
    risk_tolerance_score = (
        0.2 * average_payment_delay_score
        + 0.2 * maximum_payment_delay_score
        + 0.1 * overall_payment_delay_trend_score
        + 0.2 * has_delayed_payment_last_3
        + 0.1 * invoice_deviation_score
        + 0.1 * number_of_previous_invoices_score
        + 0.1 * net_amount_score
        + 0.1 * average_invoice_amount_score
        # + 0.01 * discount_offered_score
    ) * 10

    risk_tolerance_score = risk_tolerance_score.where(
        ~np.isnan(risk_tolerance_score), -1
    )

    return risk_tolerance_score


def calculate_customer_value_score(df):
    # Get the necessary data
    net_amount = df["net_amount"]
    average_invoice_amount = df["average_invoice_amount"]
    number_of_previous_invoices = df["number_of_previous_invoices"]

    # Define the frequency of transactions
    df["frequency_of_transactions"] = df.groupby("customer_id")["invoice_id"].transform(
        "count"
    )

    # Define recency: the more recent, the higher the score
    latest_transaction = (df["invoice_date"].max() - df["invoice_date"]).dt.days

    # Normalize these values for scoring
    net_amount_score = (net_amount - df["net_amount"].min()) / (
        df["net_amount"].max() - df["net_amount"].min()
    )
    average_invoice_amount_score = (
        average_invoice_amount - df["average_invoice_amount"].min()
    ) / (df["average_invoice_amount"].max() - df["average_invoice_amount"].min())
    number_of_previous_invoices_score = (
        number_of_previous_invoices - df["number_of_previous_invoices"].min()
    ) / (
        df["number_of_previous_invoices"].max()
        - df["number_of_previous_invoices"].min()
    )

    frequency_of_transactions_score = (
        df["frequency_of_transactions"] - df["frequency_of_transactions"].min()
    ) / (df["frequency_of_transactions"].max() - df["frequency_of_transactions"].min())

    latest_transaction_score = (latest_transaction - latest_transaction.min()) / (
        latest_transaction.max() - latest_transaction.min()
    )
    latest_transaction_score = (
        1 - latest_transaction_score
    )  # Reverse the score: the more recent, the higher the score

    # Combine these factors into a single customer value score
    # The weights can be adjusted based on business requirements.
    customer_value_score = (
        0.3 * net_amount_score
        + 0.2 * average_invoice_amount_score
        + 0.1 * number_of_previous_invoices_score
        + 0.2 * frequency_of_transactions_score
        + 0.2 * latest_transaction_score
    ) * 10

    customer_value_score = customer_value_score.where(
        ~np.isnan(customer_value_score), -1
    )

    return customer_value_score


def calculate_cash_flow_score(df):
    # Calculate average payment delay
    average_payment_delay = df["average_payment_delay"]

    # Customers who consistently pay on time should have a higher score
    # We can calculate this by using the standard deviation of the payment delays
    consistency = df["payment_delay_variance"]

    # Total income from this customer
    customer_income = df["net_amount"]

    # Normalize these values for scoring
    average_payment_delay_score = 1 - (
        (average_payment_delay - df["average_payment_delay"].min())
        / (df["average_payment_delay"].max() - df["average_payment_delay"].min())
    )  # inverse because higher delay should lower the score
    consistency_score = 1 - (
        (consistency - df["payment_delay_variance"].min())
        / (df["payment_delay_variance"].max() - df["payment_delay_variance"].min())
    )  # inverse because higher variance should lower the score
    customer_income_score = (customer_income - df["net_amount"].min()) / (
        df["net_amount"].max() - df["net_amount"].min()
    )

    # Combine these factors into a single cash flow score
    # The weights can be adjusted based on business requirements
    cash_flow_score = (
        0.3 * average_payment_delay_score
        + 0.3 * consistency_score
        + 0.4 * customer_income_score
    ) * 10

    cash_flow_score = cash_flow_score.where(~np.isnan(cash_flow_score), -1)

    return cash_flow_score


def calculate_historical_score(df):
    alpha = 0.1  # decay parameter, adjust as needed

    # Calculate the weighted trend score with exponential decay
    df["weighted_payment_delay_trend"] = df["payment_delay_trend"].apply(
        lambda x: alpha * (1 - alpha) ** x * x
    )
    payment_delay_trend_score = 1 - (
        (df["weighted_payment_delay_trend"] - df["weighted_payment_delay_trend"].min())
        / (
            df["weighted_payment_delay_trend"].max()
            - df["weighted_payment_delay_trend"].min()
        )
    )  # inverse because higher delay trend should lower the score

    # We consider the maximum payment delay a customer has ever had
    max_payment_delay = df["maximum_payment_delay"]

    # Calculate the weighted consistency score with exponential decay
    df["weighted_payment_delay_variance"] = df["payment_delay_variance"].apply(
        lambda x: alpha * (1 - alpha) ** x * x
    )
    consistency_score = 1 - (
        (
            df["weighted_payment_delay_variance"]
            - df["weighted_payment_delay_variance"].min()
        )
        / (
            df["weighted_payment_delay_variance"].max()
            - df["weighted_payment_delay_variance"].min()
        )
    )  # inverse because higher variance should lower the score

    # Normalize max delay for scoring
    max_payment_delay_score = 1 - (
        (max_payment_delay - df["maximum_payment_delay"].min())
        / (df["maximum_payment_delay"].max() - df["maximum_payment_delay"].min())
    )  # inverse because higher max delay should lower the score

    # Combine these factors into a single historical score
    # The weights can be adjusted based on business requirements. For instance, we give a higher weight to the trend score.
    historical_score = (
        0.5 * payment_delay_trend_score
        + 0.25 * max_payment_delay_score
        + 0.25 * consistency_score
    ) * 10

    historical_score = historical_score.where(~np.isnan(historical_score), -1)

    return historical_score


def calculate_global_score(df):
    # Calculate individual scores
    df["risk_tolerance_score"] = calculate_risk_tolerance_score(df)
    df["customer_value_score"] = calculate_customer_value_score(df)
    df["cash_flow_score"] = calculate_cash_flow_score(df)
    df["historical_score"] = calculate_historical_score(df)

    # If any individual score is -1, then set composite score as -1
    df.loc[
        (df["risk_tolerance_score"] == -1)
        | (df["customer_value_score"] == -1)
        | (df["cash_flow_score"] == -1)
        | (df["historical_score"] == -1),
        [
            "risk_tolerance_score",
            "customer_value_score",
            "cash_flow_score",
            "historical_score",
            "global_score",
        ],
    ] = -1

    # Define weights based on business context (these need to be refined)
    weights = {
        "risk_tolerance_score": 0.10,  # 25% weight
        "customer_value_score": 0.35,  # 25% weight
        "cash_flow_score": 0.05,  # 25% weight
        "historical_score": 0.50,  # 25% weight
    }

    # If composite score is not already -1, calculate composite score
    mask = df["global_score"] != -1
    df.loc[mask, "global_score"] = (
        df.loc[mask, "risk_tolerance_score"] * weights["risk_tolerance_score"]
        + df.loc[mask, "customer_value_score"] * weights["customer_value_score"]
        + df.loc[mask, "cash_flow_score"] * weights["cash_flow_score"]
        + df.loc[mask, "historical_score"] * weights["historical_score"]
    )

    # Normalize the composite score to have a range of 0-10
    df.loc[mask, "global_score"] = (
        df.loc[mask, "global_score"] / df.loc[mask, "global_score"].max()
    ) * 10

    return df


"""Risk Tolerance Score (0-10): This score quantifies the level of payment risk associated with a particular customer. A lower score (closer to 0) suggests that the customer is high risk, i.e., there is a significant chance of delayed payment or default based on the customer's past behaviour. This is reflected in things like a high average payment delay, frequent delayed payments, a trend of increasing payment delay, etc. A higher score (closer to 10) means the customer is low risk and generally pays their invoices on time.
Customer Value Score (0-10): This score represents the financial value of the customer to the business. Customers who have purchased frequently and recently, and who typically make large purchases, will have higher scores (closer to 10). Customers who purchase infrequently, make small purchases, or have not purchased recently will have lower scores (closer to 0).
Cash Flow Score (0-10): This score represents the predictability and reliability of the cash flow from a customer. A high score (closer to 10) suggests that the customer consistently pays their invoices on time, leading to a reliable and predictable cash flow. A low score (closer to 0) indicates that the customer's payment behaviour is inconsistent, which makes the cash flow from this customer unreliable.
Historical Score (0-10): This score captures the historical payment behaviour of a customer, with recent behaviour weighted more heavily. A high score (closer to 10) means that the customer has been improving in their payment behaviour over time, i.e., they are paying their invoices sooner. A low score (closer to 0) suggests the customer's payment behaviour has been deteriorating over time, i.e., they are taking longer to pay their invoices.
In each case, a higher score is better from the business's perspective, as it suggests a customer who provides more value and less risk to the business. These scores help the business to make informed decisions about credit terms, marketing efforts, etc., for each customer. Note that the specific meaning of these scores can be adjusted by changing the factors and weights used in their calculation, according to the business's specific needs and context."""


if __name__ == "__main__":

    # Get parameters
    run = Run.get_context()

    parser = argparse.ArgumentParser()
    # parser.add_argument("--load_folder", type=str, dest="load_folder")
    parser.add_argument("--train_folder", type=str, dest="train_folder")

    args = parser.parse_args()

    os.makedirs(args.train_folder, exist_ok=True)

    print(run.input_datasets["payment_data"])

    # Load the dataset
    df = run.input_datasets["payment_data"].to_pandas_dataframe()

    # conver all columns to lower case
    df.columns = df.columns.str.lower()

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

    # We remove invoices with missing payment date and as well with payment amount greater than 0 (this indicates the open amount of the invoices). For the purpose of this analysis we will only consider invoices that have been paid.
    df = df[
        df["payment_date"].notnull()
        & ~(df["payment_amount"] > 0)
        & (df["payment_complete"] == "Y")
    ]

    df.isnull().sum()

    # Remove unpaid invoices older than 1 year
    df = df[
        ~(
            (df["payment_date"].isnull())
            & (
                (df["create_date"] < pd.Timestamp.today() - pd.DateOffset(years=1))
                | (df["invoice_date"] < pd.Timestamp.today() - pd.DateOffset(years=1))
            )
        )
    ]

    # company_id == -1 means the company number is missing. A negative net_amount seems to be a special case, that without any further information we can not determine the real meaning. (We are probably handling debit notes from the company?)
    # We will remove these rows from the dataset. We will as well remove the columnt for payment_amount and payment_complete as they are no longer relevant for the analysis.

    df = df[(df["company_id"] != -1) & ~(df["net_amount"] < 0)]

    df.drop(columns=["payment_amount", "payment_complete"], inplace=True)

    df = df[(df["invoice_date"] < df["payment_date"])]

    df = df[(df["invoice_date"] < df["due_date"])]

    # Sort the dataframe by 'customer_id', 'company_id', and 'invoice_date' and reset the index
    df = df.sort_values(by=["customer_id", "company_id", "invoice_date"]).reset_index(
        drop=True
    )

    # Calculate main features
    calculate_features(df)

    # Calculate trend features
    calculate_trend_features(df)

    # Calcualte scores
    calculate_global_score(df)

    # Remove all invoices with a score of -1, this means that there is not enough data to calculate the score therefore unable to predict
    df = df[df["global_score"] != -1]

    # transform region_level1, region_level2, region_level3 to categorical and then to numerical
    categorical_columns = [
        "region_level1",
        "region_level2",
        "region_level3",
    ]

    df[categorical_columns] = df[categorical_columns].astype("category")
    df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)

    # Create the target variable. We will create buckets of 15 days for the payment delay
    bins = [-np.inf, 15, 30, 45, 60, np.inf]
    labels = [0, 1, 2, 3, 4]

    # "payment_bucket" will be defined as follow
    # 0: payment delay <= 15 days
    # 1: 15 < payment delay <= 30 days
    # 2: 30 < payment delay <= 45 days
    # 3: 45 < payment delay <= 60 days
    # 4: payment delay > 60 days

    # Only create 'payment_bucket' where 'payment_delay' is not null
    df.loc[df["payment_date"].notna(), "payment_bucket"] = pd.cut(
        df.loc[df["payment_date"].notna(), "payment_delay"], bins=bins, labels=labels
    ).astype(int)

    df.to_csv(os.path.join(args.train_folder, "df_for_training.csv"), index=False)
