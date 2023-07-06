# Base image for ARM architecture (Apple M1)
FROM python:latest

# Set the working directory in the container
WORKDIR /app

# Copy the project files to the working directory
COPY . /app

# Install Mamba for ARM architecture
RUN apt-get update && apt-get install -y wget bzip2
RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
RUN bash Miniforge3-Linux-aarch64.sh -b -p /opt/conda
RUN rm Miniforge3-Linux-aarch64.sh

# Add Conda to the PATH
ENV PATH="/opt/conda/bin:$PATH"

# Initialize shell to use conda activate
RUN conda init bash

# Create and activate Conda environment with Mamba
RUN conda install -y mamba -n base -c conda-forge
COPY env.yml .
RUN mamba env create -n mlops-wandb -f env.yml

# Set the entrypoint command
CMD ["conda", "run", "-n", "mlops-wandb", "/bin/bash"]

# Set the entrypoint command and include source activate
ENTRYPOINT ["/bin/bash", "-c", "source activate mlops-wandb && /bin/bash"]