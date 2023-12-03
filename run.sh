# poetry run uvicorn app.main:app --reload --host 0.0.0.0 --port 9000

# http2に対応するため、hypercornにする
# CloudRun のサービスクォータにより、リクエストファイルサイズは HTTP/1 で 32MB に制限されています。
# しかし、HTTP/2 にはそのような制限はありません。
poetry run hypercorn app.main:app --reload --bind 0.0.0.0:9000