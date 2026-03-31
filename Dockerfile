FROM pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

USER root
WORKDIR /workspaces

RUN apt-get update -y && apt-get install -y cmake build-essential git htop curl sudo wget nano

COPY environment.yaml .

RUN conda env update -n base -f environment.yaml -vvv && conda clean -afy

RUN mkdir -p /workspaces/data /workspaces/predictions

SHELL ["/bin/bash", "-c"]

CMD ["bash"]