# Replace the base tag with a different one if needed
FROM us-docker.pkg.dev/tensorflow-training-176320/terraform-multi-region-images/torch-experiments-flyte:latest

USER root
RUN apt-get -y update && apt-get install -y sudo locales zsh vim

ENV GOOGLE_CLOUD_PROJECT=tensorflow-training-176320

# Store pre-commit environment in the workspace to persist across devcontainer rebuilds
ENV PRE_COMMIT_HOME=.pre-commit

ENV PYTHONPATH="/workspaces/torch-experiments:/workspaces/torch-experiments/libs/racon-torch:/workspaces/torch-experiments/libs/recogni-training-utils:/workspaces/torch-experiments/libs/torch-alu:/workspaces/torch-experiments/applications"

# Use a non-root user
# devcontainer will then update uid & gid to the host system user
# to not break the file permissions mounted from the host.
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Wrap pip and pip-sync with sudo to avoid permission issues
RUN echo "alias pip='sudo \$(which pip)'" >> /home/$USERNAME/.bashrc && \
    echo "alias pip-sync='sudo \$(which pip-sync)'" >> /home/$USERNAME/.bashrc && \
    echo "alias pip='sudo \$(which pip)'" >> /home/$USERNAME/.zshconfig && \
    echo "alias pip-sync='sudo \$(which pip-sync)'" >> /home/$USERNAME/.zshconfig

WORKDIR /home/$USERNAME
