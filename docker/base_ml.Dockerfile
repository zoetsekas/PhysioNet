# ---- Builder Stage ----
# This stage installs build tools, creates the virtual environment, and installs packages.
FROM nvidia/cuda:13.0.1-cudnn-runtime-ubuntu24.04 AS builder

# Set non-interactive frontend for apt-get to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install build-time dependencies required for compiling Python packages.
# We explicitly install python3.12 and set it as the default for consistency.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.12 \
    build-essential \
    curl \
    libpq-dev \
    git && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --set python3 /usr/bin/python3.12 && \
    rm -rf /var/lib/apt/lists/*

# Install uv, our Python package manager and venv creator.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Create a virtual environment. The --seed flag is important to ensure pip is available.
ENV VENV_PATH=/opt/venv
RUN uv venv $VENV_PATH --seed

# Activate the virtual environment for the package installation step.
ENV PATH="$VENV_PATH/bin:$PATH"

# Copy only the requirements file to leverage Docker's layer caching.
WORKDIR /app
COPY ./docker/requirements_core.txt .

# Increase the HTTP timeout to allow more time for downloading large packages
# like PyTorch or the NVIDIA RAPIDS libraries.
ENV UV_HTTP_TIMEOUT=300
RUN uv pip install --no-cache -r requirements_core.txt

COPY ./docker/requirements.txt .
# Install dependencies into the virtual environment.
RUN uv pip install --no-cache -r requirements.txt


# ---- Final Stage ----
# This stage starts from a clean base image and copies only the virtual environment.
FROM nvidia/cuda:13.0.1-cudnn-runtime-ubuntu24.04

# Set non-interactive frontend for apt-get to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive

# Ensure python3.12 is the default python in the final image as well, and install sudo.
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3.12 sudo && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --set python3 /usr/bin/python3.12 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user and give it sudo permissions.
RUN useradd -m -s /bin/bash trading && echo "trading ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Copy the virtual environment from the builder stage.
COPY --from=builder /opt/venv /opt/venv

# Copy uv from the builder stage.
COPY --from=builder /root/.local/bin/uv /usr/local/bin/uv

# Set the working directory for the final application.
WORKDIR /app

# Grant ownership of venv and workdir to the new user.
RUN chown -R trading:trading /opt/venv /app

# Switch to the non-root user.
USER trading

# Add the virtual environment to the PATH.
ENV PATH="/opt/venv/bin:$PATH"

# Keep the container running to allow PyCharm to connect.
CMD ["tail", "-f", "/dev/null"]
