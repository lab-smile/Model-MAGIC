FROM nvidia/cuda:11.6.1-devel-ubuntu20.04
RUN apt-get update && apt-get install -y apt-utils wget unzip
RUN wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh -O /Anaconda3-2021.11-Linux-x86_64.sh
# ADD Anaconda3-2021.11-Linux-x86_64.sh /
RUN mv /Anaconda3-2021.11-Linux-x86_64.sh ~/conda.sh \
    && /bin/bash ~/conda.sh -b -p /usr/local/conda \
    && rm -f ~/conda.sh
ENV PATH /usr/local/conda/bin:$PATH
ADD magic_test.yml /magic_test.yml
RUN conda env create -f /magic_test.yml
RUN echo "conda activate magic_test" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

ENTRYPOINT ["/bin/bash", "/run.sh"]