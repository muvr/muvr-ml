# Muvr

[muvr](http://www.muvr.io/) is a demonstration of an application that uses _wearable devices_ (Pebble)—in combination with a mobile app—to submit physical (i.e. accelerometer, compass) and biological (i.e. heart rate) information to a CQRS/ES cluster to be analysed.

#### muvr-ml
`muvr-analytics` contains machine learning pipelines external to the main application, including
* pipelines to suggest future exercise sessions
* pipelines to classify users to groups by attribute similarity
* pipelines to improve classification and other models
* pipelines to train models to recognize repetitions of exercises

This part of the project can be viewed as a data science playground. The models used in the application are trained using python and exports the parameters of the trained model (configuration and weights).

#### Other components of the system
- [muvr-server](https://github.com/muvr/muvr-server) CQRS/ES cluster
- [muvr-ios](https://github.com/muvr/muvr-ios) iOS application showcasing mobile machine learning and data collection
- [muvr-pebble](https://github.com/muvr/muvr-pebble) Pebble application, example implementation of a wearable device
- [muvr-preclassification](https://github.com/muvr/muvr-preclassification) mobile data processing and classification

## Getting started
Basic information to get started is below. Please also have a look at the other components of the system to get a better understanding how everything fits together.

### Clone
```
git clone git@github.com:muvr/muvr-ml.git
```
There are two ways to get started and work with the code, either by using docker (recommended) or by manually creating conda environment and install the dependenceis.

### Using Docker [recommended]
All of the following commands assumes you have the data on the default Google drive directory on your home directory: `~/Google Drive/Exercise Data`.
But you can override that data path by passing `DATA` arg to the command e.g.: `make dev DATA=~/my-data/my-exercises`
#### Build the container and start a shell
```bash
$ make dev
# data will be mounted in /data
# code will be mounted in /src
```
#### Start Jupyter notebook
```bash
$ make notebook
```
#### Run tests
```bash
$ make test
```
### Manual setup on Mac OS
#### Install miniconda for Mac:
http://conda.pydata.org/docs/download.html
#### Build and Source the Environment
```bash
# build the environment
$ ./venv.sh

# source the environment
$ source activate muvr_ml

# link the data directory. Make sure /data doesn't exist before the following step
$ sudo ln -s "${HOME}/Google Drive/Exercise Data" /data
$ sudo chown `whoami` /data
```
#### Start jupyter notebook
```bash
$ jupyter notebook
```
#### Run tests
```bash
nosetests -v */*_test.py
```
### Issues

For any bugs or feature requests please:

1. Search the open and closed
   [issues list](https://github.com/muvr/muvr-analytics/issues) to see if we're
   already working on what you have uncovered.
2. Make sure the issue / feature gets filed in the relevant components (e.g. server, analytics, ios)
3. File a new [issue](https://github.com/muvr/muvr-analytics/issues) or contribute a
  [pull request](https://github.com/muvr/muvr-analytics/pulls)

## License
Please have a look at the [LICENSE](https://github.com/muvr/muvr-analytics/blob/develop/LICENSE) file.
