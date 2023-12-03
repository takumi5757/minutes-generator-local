# minutes generator

## Run Web API
### Local

### setup
```
poetry init
poetry install
direnv allow
```

```sh
$ sh run.sh
```

```sh
$ poetry run uvicorn app.main:app --reload --host 0.0.0.0 --port 9000
```

### Docker Compose

```sh
$ docker-compose up --build
```

## Request Commands

```sh
curl --request GET --url http://0.0.0.0:9000/healthz
```

```sh
curl --request POST --url http://0.0.0.0:9000/api/v1/summarize -H "Content-Type: multipart/form-data" -F "upload_file=@/home/sato/works/play_ground/takumi_saito/minutes_generator/sample/sampleTokyo.wav" | jq
```


## Development
### Run Tests and Linter

```
$ poetry run tox
```


### How to use
```
curl --request POST --url http://0.0.0.0:9000/api/v1/summarize -H "Content-Type: multipart/form-data" -F "upload_file=@/home/sato/works/play_ground/takumi_saito/minutes_generator/sample/sampleTokyo.wav" | jq
```

メモ
torch install
!poetry source add torch_cpu --priority=explicit https://download.pytorch.org/whl/cpu
!poetry add torch torchvision torchaudio --source torch_cpu