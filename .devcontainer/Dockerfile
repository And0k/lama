FROM python:3.12-slim
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    bubblewrap \
    socat \
    git \
    curl \
    build-essential \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspaces/lama

# copy only minimal files here so postCreateCommand can run setup.sh
COPY . /workspaces/lama
