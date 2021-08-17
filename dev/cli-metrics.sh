# cpu compute
az monitor metrics list --resource /subscriptions/6560575d-fa06-4e7d-95fb-f962e74efd7a/resourceGroups/UW-Embeddings/providers/Microsoft.MachineLearningServices/workspaces/TxtSum --metric CpuUtilizationPercentage --dimension RunID --aggregation Average --start-time 2021-07-07T00:00:00Z --end-time 2021-07-08T00:00:00Z --interval PT24H --top 5

# gpu mem
az monitor metrics list --resource /subscriptions/6560575d-fa06-4e7d-95fb-f962e74efd7a/resourceGroups/UW-Embeddings/providers/Microsoft.MachineLearningServices/workspaces/TxtSum --metric GpuMemoryUtilizationMegabytes --dimension RunID --aggregation Average Maximum --start-time 2021-07-07T00:00:00Z --end-time 2021-07-08T00:00:00Z --interval PT24H --top 5

# gpu compute
az monitor metrics list --resource /subscriptions/6560575d-fa06-4e7d-95fb-f962e74efd7a/resourceGroups/UW-Embeddings/providers/Microsoft.MachineLearningServices/workspaces/TxtSum --metric GpuUtilizationPercentage --dimension RunID --aggregation Average --start-time 2021-07-07T00:00:00Z --end-time 2021-07-08T00:00:00Z --interval PT24H --top 5

# gpu energy
az monitor metrics list --resource /subscriptions/6560575d-fa06-4e7d-95fb-f962e74efd7a/resourceGroups/UW-Embeddings/providers/Microsoft.MachineLearningServices/workspaces/TxtSum --metric GpuEnergyJoules --dimension RunID --aggregation Total --start-time 2021-07-07T00:00:00Z --end-time 2021-07-08T00:00:00Z --interval PT24H --top 5

# gpu mem per device
az monitor metrics list --resource /subscriptions/6560575d-fa06-4e7d-95fb-f962e74efd7a/resourceGroups/UW-Embeddings/providers/Microsoft.MachineLearningServices/workspaces/TxtSum --metric GpuMemoryUtilizationMegabytes --filter "RunID eq 'b8ad2ad7-ecf8-481f-948e-e94761eeec09' and DeviceID eq '*'" --aggregation Average Maximum --start-time 2021-07-07T00:00:00Z --end-time 2021-07-08T00:00:00Z --interval PT24H

# table output
az monitor metrics list --resource /subscriptions/6560575d-fa06-4e7d-95fb-f962e74efd7a/resourceGroups/UW-Embeddings/providers/Microsoft.MachineLearningServices/workspaces/TxtSum --metric GpuMemoryUtilizationMegabytes --filter "RunID eq 'b8ad2ad7-ecf8-481f-948e-e94761eeec09' and DeviceID eq '*'" --aggregation Average Maximum --start-time 2021-07-07T00:00:00Z --end-time 2021-07-08T00:00:00Z --interval PT24H --output table > metrics-memory.txt

az monitor metrics list --resource /subscriptions/6560575d-fa06-4e7d-95fb-f962e74efd7a/resourceGroups/UW-Embeddings/providers/Microsoft.MachineLearningServices/workspaces/TxtSum --metric GpuEnergyJoules --dimension RunID --aggregation Total --start-time 2021-07-07T00:00:00Z --end-time 2021-07-08T00:00:00Z --interval PT24H --top 5 --output table > metrics-energy.txt

# json output
az monitor metrics list --resource /subscriptions/6560575d-fa06-4e7d-95fb-f962e74efd7a/resourceGroups/UW-Embeddings/providers/Microsoft.MachineLearningServices/workspaces/TxtSum --metric GpuEnergyJoules --filter "RunID eq 'b8ad2ad7-ecf8-481f-948e-e94761eeec09'" --aggregation Total --start-time 2021-07-07T00:00:00Z --end-time 2021-07-08T00:00:00Z --interval PT24H --output json > metrics-energy.json
