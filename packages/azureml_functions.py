import os
import json
import joblib
import logging
from azureml.core import Workspace, Run, Model
from azureml.core.authentication import ServicePrincipalAuthentication, InteractiveLoginAuthentication
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential

def get_sp_auth():
    # in case your working on a local machine with the service principle located in the workfolder root
    f_path = '.cloud/.azure/AZURE_SERVICE_PRINCIPAL.json'
    if os.path.exists(f_path):
        with open(f_path) as f:
            cred = f.read()
            os.environ['AZURE_SERVICE_PRINCIPAL'] = cred
    service_principle_str = os.environ.get('AZURE_SERVICE_PRINCIPAL')
    
    if service_principle_str is not None:
        service_principle_cred = json.loads(service_principle_str)
        tenant_id = service_principle_cred["tenant"]
        sp_id = service_principle_cred["appId"]
        sp_pwd = service_principle_cred["password"]
    else:
        tenant_id = open("tenant.txt").read()
        sp_id = open("appid.txt").read()
        sp_pwd = open("password.txt").read()

    sp_auth = ServicePrincipalAuthentication(tenant_id=tenant_id, service_principal_id=sp_id, service_principal_password=sp_pwd)
    return sp_auth


def get_ws(stage="ml"):
    stages = {"dev", "uat", "prod", "train"}
    if stage not in stages:
        raise ValueError("Invalid stage for workspace: got %s, should be from %s" %(stage, stages))

    if stage in {"dev", "train"}:
        print("Logging in as user")
        credential = InteractiveLoginAuthentication()
    else:
        print("Logging in as Service Principal")
        credential = get_sp_auth()

    config_path = ".cloud/.azure/config_{stage}.json".format(stage=stage.upper())
    ws = Workspace.from_config(config_path, auth=credential)

    return ws
    


# def register_model(model, run, file_path, model_name):
#     # create output folder
#     os.makedirs(file_path, exist_ok=True)
#     model_file = os.path.join(file_path, 'model.pkl')
#     joblib.dump(value=model, filename=model_file)

#     print("register model")
#     print(run.get_metrics())
#     Model.register(workspace=run.experiment.workspace,
#                     model_path=model_file,
#                     model_name=model_name,
#                     tags=model.get_params(),
#                     properties=run.get_metrics()
#     )