# Official CUDA/PyTorch base with Blackwell support. Override if host driver requires another release.
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.06-py3
FROM ${BASE_IMAGE}
WORKDIR /workspace
# Isolate project packages while retaining the image's CUDA-enabled torch.
RUN python -m venv --system-site-packages /opt/ntl-venv
ENV PATH=/opt/ntl-venv/bin:$PATH
ENV TOKENIZERS_PARALLELISM=false
COPY requirements-cpu.txt requirements-gpu.txt requirements-fetch.txt ./
# Project libraries live in the venv; torch from the NVIDIA image is retained.
RUN env -u PIP_CONSTRAINT python -m pip install --no-cache-dir -r requirements-gpu.txt -r requirements-fetch.txt
COPY . .
CMD ["python", "server_start.py", "--phase", "all", "--devices", "0"]
