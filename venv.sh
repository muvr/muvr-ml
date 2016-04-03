#!/bin/bash
set -xe

# Install pip
if ! [ `command -v pip` ]
then
  sudo easy_install pip
fi

# Create virtual env
if ! [ `command -v virtualenv` ]
then
  sudo pip2.7 install virtualenv
fi

VENV=.venv
rm -rf $VENV
virtualenv $VENV -p /usr/bin/python2.7
source $VENV/bin/activate

# Install dependencies
pip2.7 install -r muvr-ml.pip

# Install pem
cd pem
python setup.py install
cd -

source $VENV/bin/activate
