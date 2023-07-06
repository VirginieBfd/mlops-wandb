# Base image
FROM python:3.11

# Set the working directory in the container
WORKDIR /app

# Copy the project files to the working directory
COPY . /app

# Install Conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh

# Add Conda to the PATH
ENV PATH="/opt/conda/bin:$PATH"

# Create and activate Conda environment
COPY env.yml .
RUN conda env create -f env.yml
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]
RUN echo "source activate myenv" >> ~/.bashrc

# Set the entrypoint command
CMD ["python", "train.py"]
