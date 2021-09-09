SUB_ID=$(az account show --query id -o tsv)
RES_GROUP="UW-Embeddings"
WORKSPACE="TxtSum"
RESOURCE_ID="/subscriptions/$SUB_ID/resourceGroups/$RES_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE"

RUN_ID="b8ad2ad7-ecf8-481f-948e-e94761eeec09"
START_TIME=""
END_TIME=""

# cpu compute
az monitor metrics list --resource $RESOURCE_ID --metric CpuUtilizationPercentage --dimension RunID --aggregation Average --start-time 2021-07-07T00:00:00Z --end-time 2021-07-08T00:00:00Z --interval P1D --top 5

# gpu mem
az monitor metrics list --resource $RESOURCE_ID --metric GpuMemoryUtilizationMegabytes --dimension RunID --aggregation Average Maximum --start-time 2021-07-07T00:00:00Z --end-time 2021-07-08T00:00:00Z --interval P1D --top 5

# gpu compute
az monitor metrics list --resource $RESOURCE_ID --metric GpuUtilizationPercentage --dimension RunID --aggregation Average --start-time 2021-07-07T00:00:00Z --end-time 2021-07-08T00:00:00Z --interval P1D --top 5

# gpu energy
az monitor metrics list --resource $RESOURCE_ID --metric GpuEnergyJoules --dimension RunID --aggregation Total --start-time 2021-07-07T00:00:00Z --end-time 2021-07-08T00:00:00Z --interval P1D --top 5

# gpu mem per device
az monitor metrics list --resource $RESOURCE_ID --metric GpuMemoryUtilizationMegabytes --filter "RunID eq '$RUN_ID' and DeviceID eq '*'" --aggregation Average Maximum --start-time 2021-07-07T00:00:00Z --end-time 2021-07-08T00:00:00Z --interval P1D

# table output
az monitor metrics list --resource $RESOURCE_ID --metric GpuMemoryUtilizationMegabytes --filter "RunID eq '$RUN_ID' and DeviceID eq '*'" --aggregation Average Maximum --start-time 2021-07-07T00:00:00Z --end-time 2021-07-08T00:00:00Z --interval P1D --output table > metrics-memory.txt

az monitor metrics list --resource $RESOURCE_ID --metric GpuEnergyJoules --dimension RunID --aggregation Total --start-time 2021-07-07T00:00:00Z --end-time 2021-07-08T00:00:00Z --interval P1D --top 5 --output table > metrics-energy.txt

# json output
az monitor metrics list --resource $RESOURCE_ID --metric GpuEnergyJoules --filter "RunID eq '$RUN_ID'" --aggregation Total --start-time 2021-07-07T00:00:00Z --end-time 2021-07-08T00:00:00Z --interval P1D --output json > metrics-energy.json
