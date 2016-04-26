build: Dockerfile
	docker build -t muvr_ml .

dev: build
	docker run -it -v `pwd`:/src -v "${HOME}/Google Drive/Exercise Data":/data muvr_ml bash

notebook: build
	docker run -it -v `pwd`:/src -v "${HOME}/Google Drive/Exercise Data":/data -p 8889:8888 muvr_ml

test: build
	docker run -it -v `pwd`:/src -v "${HOME}/Google Drive/Exercise Data":/data muvr_ml nosetests -v */*_test.py
