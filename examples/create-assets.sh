az ml environment create --file ./assets/environment/env-gpu.yml

az ml data create --file ./assets/data/samsum-local.yml

az ml compute create --file ./assets/compute/gpucluster-lp.yml