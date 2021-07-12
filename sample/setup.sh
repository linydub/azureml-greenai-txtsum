az upgrade -y
az extension remove -n ml
az extension remove -n azure-cli-ml
az extension add --source https://azuremlsdktestpypi.blob.core.windows.net/wheels/sdk-cli-v2/ml-0.0.79-py3-none-any.whl --pip-extra-index-urls https://azuremlsdktestpypi.azureedge.net/sdk-cli-v2 -y

export AZURE_ML_CLI_PRIVATE_FEATURES_ENABLED="true"

az login
az account set -s 6560575d-fa06-4e7d-95fb-f962e74efd7a
az configure --defaults group="UW-Embeddings" workspace="TxtSum" location="westus2"

az ml environment create --file ./setup/environment.yml
az ml data create --file ./setup/data.yml
az ml compute create --file ./setup/compute.yml