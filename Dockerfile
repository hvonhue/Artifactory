FROM mcr.microsoft.com/azureml/openmpi3.1.2-cuda10.2-cudnn8-ubuntu18.04

COPY ./conda.yml ./

RUN conda env update -n base -f ./conda.yml