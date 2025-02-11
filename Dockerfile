FROM nvcr.io/nvidia/tensorrt:24.10-py3

# Set non-interactive mode to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive

# Update package list and install packages without prompts
RUN apt-get update && apt-get install -y \
    zip \
    unzip \
    git \
    git-lfs \
    cmake \
    build-essential \
    gdb \
    vim \
    llvm \
    lldb \
    net-tools \
    libgl1-mesa-dev \
    byobu

# install pytorch
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# install opencv headless
RUN pip3 install --no-cache-dir opencv-python-headless

# install ultralytics
RUN pip3 install --no-cache-dir ultralytics

# install scikit-learn
RUN pip3 install --no-cache-dir scikit-learn

# install pandas
RUN pip3 install --no-cache-dir pandas

# install sqlite-vec
RUN pip3 install --no-cache-dir sqlite-vec

# install tqdm
RUN pip3 install --no-cache-dir tqdm

# install matplotlib
RUN pip3 install --no-cache-dir matplotlib

# install timm
RUN pip3 install --no-cache-dir timm

# install pyrqlite
RUN pip3 install git+https://github.com/rqlite/pyrqlite.git

WORKDIR /app
