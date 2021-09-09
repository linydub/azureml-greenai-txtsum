# old outputs hf-metrics (storage)
# GET https://txtsumstorage.blob.core.windows.net/azureml/ExperimentRun/dcid.$RUN_ID/outputs/all_results.json

# POST https://management.azure.com/subscriptions/$SUB_ID/resourceGroups/$RES_GROUP/providers/Microsoft.Storage/storageAccounts/$STORAGE_ACC/listKeys?api-version=2021-04-01

# POST https://management.azure.com/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.MachineLearningServices/workspaces/{workspaceName}/listKeys?api-version=2021-03-01-preview

# /api/2.0/mlflow

SUB_ID=$(az account show --query id -o tsv)
RES_GROUP="UW-Embeddings"
WORKSPACE="TxtsumDev"
API_VERSION="2021-04-01"
TOKEN=$(az account get-access-token --query accessToken -o tsv)

curl https://westus2.api.azureml.ms/mlflow/v1.0/subscriptions/$SUB_ID/resourceGroups/$RES_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/api/2.0/preview/mlflow/experiments/list?api-version=$API_VERSION \
-H "Authorization:Bearer $TOKEN"

# az rest -m get --header "Accept=application/json" -u 'https://westus2.api.azureml.ms/mlflow/v1.0/$SUB_ID/resourceGroups/$RES_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/api/2.0/preview/mlflow/experiments/list' --resource 'https://management.azure.com/'
