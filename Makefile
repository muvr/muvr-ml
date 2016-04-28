help:
	@cat Makefile

DATA?="${HOME}/Google Drive/Exercise Data"

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
	$(DOCKER) run -it -v `pwd`:/src -v $(DATA):/data muvr_ml nosetests -v */*_test.py

