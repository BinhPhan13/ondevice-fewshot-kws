FROM nvcr.io/nvidia/pytorch:22.08-py3
WORKDIR /workspace

RUN sed -ie '/^#include/a #include <ATen/core/Tensor.h>' /opt/conda/lib/python3.8/site-packages/torch/include/ATen/cuda/CUDAUtils.h \
&& git clone https://github.com/pytorch/audio.git && cd audio && git checkout v0.13.0 && BUILD_RNNT=0 python setup.py install && cd .. && rm -rf audio \
&& pip install --no-cache-dir -U git+https://github.com/shwoo93/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas" --install-option="--force_cuda"

ADD apt.installs.txt .
ARG DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y $(cat apt.installs.txt) && rm -rf /var/lib/apt/lists/*

ADD requirements.lock requirements-extra.lock .
RUN PYTHONDONTWRITEBYTECODE=1 pip install --no-cache-dir --no-deps -r requirements.lock -r requirements-extra.lock

ADD . ./ondevice-fewshot-kws

