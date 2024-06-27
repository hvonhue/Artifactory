FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04

COPY ./conda.yml ./

RUN conda env update -n base -f ./conda.yml