# minutes-generator-local

## What's this
音声ファイルからその文字起こしと要約をするAPIの実装

[chatGPTとwhisperで音声要約APIを作ってみた](https://zenn.dev/hoge/hoge)

## Setup

OPENAIのAPIキーと、SlackのWebhook URLを.envに記載する

```
cp .env.example .env
```

.envを編集して、


```sh
$ docker-compose up --build
```

## How to use

```sh
curl --request GET --url http://0.0.0.0:9000/heartbeat
```

```
curl --request POST --url http://0.0.0.0:9000/api/v1/summarize -H "Content-Type: multipart/form-data" -F "upload_file=@/hoge/fuga.wav" | jq
```

## Input limits
- 対応するファイルの最大長は4時間
- 対応しているファイル形式： [.mp4, .mp3, .wav, .m4a]

## Reference

- [機械学習の推論WebAPIの実装をテンプレート化して使い回せるようした](https://zenn.dev/yag_ays/articles/eef1a8c8e1ee39)