# download run outputs from workspace storage
STORAGE_ACCOUNT="linydub"
RUN_ID="a956f04c-3d63-4072-8d76-de64ec1a7029"

az storage blob download-batch --source azureml --destination . --pattern ExperimentRun/dcid.$RUN_ID/outputs/* --account-name $STORAGE_ACCOUNT