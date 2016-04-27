help:
	@cat Makefile

DATA?="${HOME}/Google Drive/Exercise Data"

build:
	docker build -t muvr_ml .

dev: build
	docker run -it -v `pwd`:/src -v $(DATA):/data muvr_ml bash

notebook: build
	docker run -it -v `pwd`:/src -v $(DATA):/data -p 8888:8888 muvr_ml

test: build
	docker run -it -v `pwd`:/src -v $(DATA):/data muvr_ml nosetests -v */*_test.py
