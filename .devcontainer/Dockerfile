FROM mcr.microsoft.com/devcontainers/universal:2-linux
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    bubblewrap \
    socat \
    git-lfs \
 && rm -rf /var/lib/apt/lists/*
RUN git lfs install

# убран бессмысленный COPY . /workspaces/lama, workspace и так bind-mounted Codespaces;
