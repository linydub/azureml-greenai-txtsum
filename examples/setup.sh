az extension add -n ml -y

az group create --name "AML-GreenAI" --location "westus2"

az ml workspace create --file ./assets/workspace.yml -g "AML-GreenAI"

az configure --defaults group="AML-GreenAI" workspace="TxtsumDemo"
