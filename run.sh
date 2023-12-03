# http2に対応するため、hypercornにする
# CloudRun のリソース制限により、リクエストファイルサイズは HTTP/1 で 32MB に制限されています。
# しかし、HTTP/2 にはそのような制限はありません。
# https://cloud.google.com/run/quotas?hl=ja
poetry run hypercorn app.main:app --reload --bind 0.0.0.0:9000