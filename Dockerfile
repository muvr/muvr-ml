FROM ubuntu:14.04

ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH

RUN mkdir -p $CONDA_DIR && \
    echo export PATH=$CONDA_DIR/bin:'$PATH' > /etc/profile.d/conda.sh && \
    apt-get install -y wget && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda3-3.9.1-Linux-x86_64.sh && \
    apt-get -y purge wget && \
    echo "6c6b44acdd0bc4229377ee10d52c8ac6160c336d9cdd669db7371aa9344e1ac3 *Miniconda3-3.9.1-Linux-x86_64.sh" | sha256sum -c - && \
    /bin/bash /Miniconda3-3.9.1-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
    rm Miniconda3-3.9.1-Linux-x86_64.sh

ENV NB_USER muvr
ENV NB_UID 1000

RUN useradd -m -s /bin/bash -N -u $NB_UID $NB_USER && \
    mkdir -p $CONDA_DIR && \
    chown muvr $CONDA_DIR -R && \
    mkdir -p /src && \
    chown muvr /src

USER muvr

RUN conda install -y python=3.5 theano=0.7* pandas=0.18* scikit-learn=0.17* notebook=4* nose && \
    pip install keras ipdb && \
    conda clean -yt

WORKDIR /src

CMD jupyter notebook --port=8888 --ip=0.0.0.0
