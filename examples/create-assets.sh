az ml environment create --file ./assets/environment/hf-zero-gpu.yml

az ml data create --file ./assets/data/samsum-local.yml

az ml compute create --file ./assets/compute/gpu-cluster-large.yml
