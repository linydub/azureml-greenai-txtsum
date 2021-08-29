# Standard_DS2_v2, Standard_DS1_v2, Standard_F2s_v2
# gh self-hosted runner setup script for aml compute instance
az upgrade -y

az extension remove -n azure-cli-ml
az extension remove -n ml

az extension add -n ml -y

# login using system-assigned identity
az login --identity

# actions-runner download
cd /./tmp && mkdir actions-runner && cd actions-runner

curl -o actions-runner-linux-x64-2.280.3.tar.gz -L https://github.com/actions/runner/releases/download/v2.280.3/actions-runner-linux-x64-2.280.3.tar.gz

tar xzf ./actions-runner-linux-x64-2.280.3.tar.gz