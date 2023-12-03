FROM python:3.10.7-buster

WORKDIR /app/

ENV DEBIAN_FRONTEND=noninteractive

# Install ffmpeg
RUN apt update && apt install -y ffmpeg

# vadのインストール
RUN apt install -y git

# リポジトリをクローンし、特定のコミットにチェックアウト
RUN mkdir /silero-vad && \ 
    git clone https://github.com/snakers4/silero-vad /silero-vad && \
    cd /silero-vad && \
    git checkout 563106ef8cfac329c8be5f9c5051cd365195aff9

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=/opt/poetry python && \
    cd /usr/local/bin && \
    ln -s /opt/poetry/bin/poetry && \
    poetry config virtualenvs.create false

# Copy poetry.lock* in case it doesn't exist in the repo
COPY ./pyproject.toml ./poetry.lock* /app/
RUN poetry config installer.max-workers 10

# Allow installing dev dependencies to run tests
ARG INSTALL_DEV=false
RUN bash -c "if [ $INSTALL_DEV == 'true' ] ; then poetry install --no-root ; else poetry install --no-root --no-dev ; fi"

COPY . .

CMD ["sh", "./run.sh"]
