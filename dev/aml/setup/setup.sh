az upgrade -y
az extension remove -n ml
az extension remove -n azure-cli-ml
az extension add --source https://azuremlsdktestpypi.blob.core.windows.net/wheels/sdk-cli-v2/ml-0.0.79-py3-none-any.whl --pip-extra-index-urls https://azuremlsdktestpypi.azureedge.net/sdk-cli-v2 -y

export AZURE_ML_CLI_PRIVATE_FEATURES_ENABLED="true"

az login
az account set -s {string}
az configure --defaults group="UW-Embeddings" workspace="TxtSum" location="westus2"

az ml environment create --file ./env/hf-env.yml
az ml data create --file ./data/data-local.yml
az ml compute create --file ./compute/gpu-v100-lp.yml