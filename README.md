# <center>Payment Delay</center>

Predictive models help businesses prioritize their collection efforts and customize credit terms to reduce the risk of future payment delays.

Accurately predicting payment delays allows companies to take proactive measures to manage their cash flow and customer relationships, rather than reacting to late payments

The ability to accurately predict payment delays is crucial for businesses to proactively manage their AR processes, mitigate financial risks, and maintain healthy customer relationships. Predictive models empower companies to be strategic rather than reactive in their approach to late payments.

In this project, we will analyze B2B invoices to analyze the payment behavior of customers and be able to predict their delay on payment.

We will define the risk of payment delay on 4 different payment buckets. Each of these buckets will have an interval of 15 days:

* 0-15 days delay
* 15-30 days delay
+ 30-45 days delay
+ 45-60 days delay
* more than 60 days delay

## Project Development

We will create a classification Model to predict the payment delay in buckets of 15 days on open invoices. All Analysis and first training will be done in a Jupyter Notebook to be found here:

**Train Notebook:** [financial_prediction_model_v2.ipynb](notebooks/payment_delay.ipynb)

Afterwards we will create our scripts to be able to deploy this model in Azure.

Out main Pipeline is found under:

**Azure Pipeline** [run_train_pipeline.py](pipelines/run_train_pipeline.py)

Each step of the pipeline is defined under our code directory:

**Preprocess Step** [preprocesss.py](code/preprocessing.py)

**Training Step** [training.py](code/training.py)

## Final Results

**Accuracy:** 94.70%
**Precision:** 74.04%
**Recall:** 76.20%
**F1 score:** 74.81%


### Confusion matrix:
|        Actual \ Predicted         |    0-15 days    |    15-30 days    |    30-45 days    |   45-60 days   |   more than 60 days    |
|-----------------------------------|-----------------|------------------|------------------|----------------|------------------------|
|            0-15 days              |     140619      |       2794       |       128        |       45       |           78           |
|            15-30 days             |      2394       |      10141       |       750        |       79       |           75           |
|            30-45 days             |       210       |       868        |      2863        |      319       |          111           |
|            45-60 days             |       58        |       79         |       276        |      898       |          273           |
|         more than 60 days         |       38        |       14         |       25         |      100       |          1040          |

