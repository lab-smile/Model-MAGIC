FROM nvidia/cuda:11.7.1-devel-ubuntu20.04
RUN apt-get update && apt-get install -y apt-utils wget unzip
RUN apt-get update && apt-get install -y apt-utils wget unzip
RUN wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh -O /Anaconda3-2021.11-Linux-x86_64.sh
RUN mv /Anaconda3-2022.05-Linux-x86_64.sh ~/conda.sh \
    && /bin/bash ~/conda.sh -b -p /usr/local/conda \
    && rm -f ~/conda.sh
ENV PATH /usr/local/conda/bin:$PATH

ADD MAGIC-main.zip / 
RUN unzip /MAGIC-main.zip
RUN cd /MAGIC-main/src/gpu \
    && conda env create -f magic_env.yml \
    && /bin/bash -c "source activate magic_env" \
    && cd /MAGIC-main/src/gpu \
    && python magic_train.py 