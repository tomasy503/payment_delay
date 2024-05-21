import os

from azureml.core import Datastore, Environment, Experiment, Workspace
from azureml.core.compute import ComputeTarget
from azureml.core.dataset import Dataset
from azureml.core.runconfig import RunConfiguration
from azureml.data import OutputFileDatasetConfig
from azureml.data.data_reference import DataReference
from azureml.pipeline.core import Pipeline, PipelineData, PipelineParameter
from azureml.pipeline.steps import PythonScriptStep

from packages.azureml_functions import get_ws


def get_pipeline_steps(ws):
    steps = []

    # Step 1
    df = ws.datasets.get("payment_data")

    location_preprocessing = "code/preprocessing/"
    location_environment = "environments/"
    pipeline_preprocessing_run_config = RunConfiguration()

    env_prep = Environment.from_conda_specification(
        name="pipeline_environment_preprocess",
        file_path=location_environment + "preprocessing.yml",
    )
    pipeline_preprocessing_run_config.environment = env_prep
    pipeline_cluster_preprocessing = ComputeTarget(workspace=ws, name="high-memory512")

    train_folder = PipelineData("train_folder")

    preprocessing_step = PythonScriptStep(
        name="Preprocess Data",
        source_directory=location_preprocessing,
        script_name="preprocessing.py",
        inputs=[df.as_named_input("payment_data")],
        outputs=[train_folder],
        arguments=["--train_folder", train_folder],
        compute_target=pipeline_cluster_preprocessing,
        runconfig=pipeline_preprocessing_run_config,
        allow_reuse=True,
    )

    steps.append(preprocessing_step)

    # Step 2
    location_training = "code/training/"
    location_environment = "environments/"
    pipeline_training_run_config = RunConfiguration()

    train_env = Environment.from_conda_specification(
        name="pipeline_environment_preprocess",
        file_path=location_environment + "training.yml",
    )
    pipeline_training_run_config.environment = train_env
    pipeline_cluster_training = ComputeTarget(workspace=ws, name="high-memory512")

    prediction_folder = PipelineData("prediction_folder")

    train_step = PythonScriptStep(
        name="Training",
        source_directory=location_training,
        script_name="training.py",
        inputs=[train_folder],
        outputs=[prediction_folder],
        arguments=[
            "--train_folder",
            train_folder,
            "--prediction_folder",
            prediction_folder,
        ],
        compute_target=pipeline_cluster_training,
        runconfig=pipeline_training_run_config,
        allow_reuse=True,
    )

    steps.append(train_step)

    return steps


if __name__ == "__main__":
    PIPELINE_NAME = "payment delay predictions"

    ws = get_ws("train")
    pipeline_steps = get_pipeline_steps(ws)

    pipeline = Pipeline(workspace=ws, steps=pipeline_steps)
    print("Pipeline is built.")

    # Create an experiment and run the pipeline
    # repo_shorthand = Repository('.').head.shorthand
    repo_shorthand = "train"
    name = repo_shorthand + "_" + PIPELINE_NAME  # branch name
    name = name.replace("/", "_")

    experiment = Experiment(workspace=ws, name=name)
    pipeline_run = experiment.submit(pipeline)
    print("Pipeline submitted for execution.")
    pipeline_result = pipeline_run.wait_for_completion(show_output=True)


# if __name__ == "__main__":
#     PIPELINE_NAME = "payment_delay"

#     ws = get_ws("train")
#     # Get the default datastore
#     datastore = ws.get_default_datastore()

#     pipeline_runs = []

#     for filename in file_names:
#         # date_string = filename[18:26]

#         pipeline_steps = get_pipeline_steps(ws)

#         pipeline = Pipeline(workspace=ws, steps=pipeline_steps)
#         print("Pipeline is built.")

#         name = f"train_{PIPELINE_NAME}"

#         experiment = Experiment(workspace=ws, name=name)
#         pipeline_run = experiment.submit(pipeline)
#         print("Pipeline submitted for execution.")
#         pipeline_runs.append(pipeline_run)
#         # pipeline_result = pipeline_run.wait_for_completion(show_output=True)
#         # Wait for all pipeline runs to complete
#     for run in pipeline_runs:
#         run.wait_for_completion(show_output=True)
