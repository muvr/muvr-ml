help:
	@cat Makefile

DATA?=$(abspath ../muvr-exercise-data)

ifdef GPU
	DOCKER_FILE=DockerfileGPU
	DOCKER=GPU=$(GPU) nvidia-docker
else
	DOCKER_FILE=Dockerfile
	DOCKER=docker
endif

build:
	docker build -t muvr_ml -f $(DOCKER_FILE) .

dev: build
	$(DOCKER) run -it -v `pwd`:/src -v $(DATA):/data muvr_ml bash

notebook: build
	$(DOCKER) run -it -v `pwd`:/src -v $(DATA):/data -p 8888:8888 muvr_ml

test: build
	$(DOCKER) run -it -v `pwd`:/src -v $(DATA):/data muvr_ml nosetests -v sensorcnn

deploy-model:
	#Example: make deploy-model MODEL=output/mlp_1/1464206380
	cp $(MODEL)/labels.txt ../muvr-ios/Muvr/Models.bundle/setup_model.labels.txt
	cp $(MODEL)/layers.txt ../muvr-ios/Muvr/Models.bundle/setup_model.layers.txt
	cp $(MODEL)/weights.raw ../muvr-ios/Muvr/Models.bundle/setup_model.weights.raw
