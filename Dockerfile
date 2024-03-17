FROM ubuntu:22.04
FROM python:3.11.8-slim-bookworm
# Setting up description
LABEL author="Andrey Grishin"
LABEL group="M05-318a"
LABEL title="Neural Machine Translation (MLOps project)"
LABEL description="Construct the env. suitable for solving NMT problem."
# Setting up env. variables
ARG YOUR_ENV
ENV YOUR_ENV=${YOUR_ENV} \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    # Poetry's configuration:
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_CACHE_DIR='/var/cache/pypoetry' \
    POETRY_HOME='/usr/local' \
    POETRY_VERSION=1.7.1

# Updating distribution
RUN apt-get update -y
RUN apt-get install python3 -y
RUN apt-get install curl -y

# Installing poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Copy project files
COPY config /home/config/
COPY mlops /home/mlops/
COPY commands.py /home/
COPY poetry.lock /home/
COPY pyproject.toml /home/
COPY README.md /home/

# Changing working directory
WORKDIR /home/
RUN poetry install

# Installing spacy tokenizers
RUN python -m spacy download en_core_web_sm
RUN python -m spacy download ru_core_news_sm

# Execute bash command
ENTRYPOINT ["/bin/bash"]