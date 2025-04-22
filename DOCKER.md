# Docker での実行方法

## 必要条件

- Docker
- Docker Compose (オプション)

## 単独での Docker ビルドと実行

### ビルド

```bash
docker build -t llm-eval-backend .
```

### 実行

```bash
docker run -p 8000:8000 -e LLMEVAL_ENV=development llm-eval-backend
```

データセットと結果を永続化するためのボリュームマウント:

```bash
docker run -p 8000:8000 \
  -v $(pwd)/datasets:/app/datasets \
  -v $(pwd)/results:/app/results \
  -e LLMEVAL_DATASET_DIR=/app/datasets/test \
  -e LLMEVAL_TRAIN_DIR=/app/datasets/train \
  -e LLMEVAL_RESULTS_DIR=/app/results \
  -e LLMEVAL_LITELLM_BASE_URL=http://host.docker.internal:11434/api/generate \
  llm-eval-backend
```

## Docker Compose での実行

環境変数を設定するには、`.env` ファイルを作成します（`.env.sample` をコピーして使用できます）:

```bash
cp .env.sample .env
# 必要に応じて .env ファイルを編集
```

Docker Compose での起動:

```bash
docker-compose up -d
```

## ログの確認

```bash
docker-compose logs -f api
```

## API エンドポイントへのアクセス

- API ドキュメント: http://localhost:8000/docs
- 評価エンドポイント: http://localhost:8000/api/evaluation/run

## Ollama との連携

Ollama を使用する場合は、`docker-compose.yml` 内のコメントを解除して構成を有効にします。
または、ホスト側で実行している Ollama に接続するには、環境変数 `LLMEVAL_LITELLM_BASE_URL` を適切に設定します。

例（ホスト側で実行している Ollama に接続）:
```
LLMEVAL_LITELLM_BASE_URL=http://host.docker.internal:11434/api/generate
```
