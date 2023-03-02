FROM nvidia/cuda:11.4.0-cudnn8-devel-ubuntu18.04


# Install some basic utilities
RUN apt-get update || true && apt-get install -y \
    wget curl git git-lfs vim zip unzip tmux htop \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    python3 python3-pip python3-venv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Create a working directory
RUN mkdir /app
WORKDIR /app

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app
#RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user

USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

# Install Miniconda and Python 3.8
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/home/user/miniconda/bin:$PATH
RUN curl -sLo ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-py38_4.8.2-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh \
 && conda install -y python==3.8.1 \
 && conda clean -ya

# Set the default command to python3

RUN pip install notebook
RUN pip install jupyterlab
# RUN git clone https://github.com/nokiroki/NLP-Transactions.git
CMD jupyter notebook --allow-root --ip='0.0.0.0' --port=8890 --NotebookApp.token='' --NotebookApp.password=''