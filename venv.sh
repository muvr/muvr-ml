#!/bin/bash
set -e

# Install conda
if [[ ! `conda --version` ]]
then
    # TODO install conda?
    echo "You need to install conda first goto: http://conda.pydata.org/docs/download.html"
    exit 1
fi

# Create env muvr_ml with python 3.5
if [[ `conda env list|grep muvr_ml` ]]
then
    conda env update -f environment_mac.yml
else
    conda env create -f environment_mac.yml
fi
