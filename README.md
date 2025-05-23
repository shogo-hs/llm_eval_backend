# LLM Evaluation Backend

LLM Evaluation Backendは、大規模言語モデル（LLM）の評価を効率的に実行するためのバックエンドサービスです。日本語および英語の複数データセットを使用して、モデルの性能を様々なメトリクスで評価できます。

## 主な特徴

- **複数データセット評価**: JASTER、AIO、JANLIなど複数の標準データセットで評価
- **Few-shot学習のサポート**: 0-shot、Few-shotでの評価をサポート
- **多様なメトリクス**: 文字ベースF1、完全一致、その他のメトリクスで評価
- **REST API**: FastAPIを使用した高速なAPI提供
- **LiteLLM統合**: 様々なLLMプロバイダへの統一インターフェース
- **MLflow連携**: 評価結果の追跡と可視化

## 使い方

### インストール

```bash
# リポジトリのクローン
git clone https://github.com/shogohasegawa/llm_eval_backend.git
cd llm_eval_backend

# 依存関係のインストール
uv init
uv install
```

### 設定

`src/app/config/config.py`で評価に必要な設定を行います：

```python
# データセットパス、モデル設定などを設定
```

または、環境変数を使用して設定することもできます（`.env.sample`を参照）。

### サーバー起動

```bash
# 開発環境での起動
uv run src/app/main.py

# 本番環境での起動
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Dockerでの実行

```bash
# Dockerイメージのビルド
docker build -t llm-eval-backend .

# Dockerコンテナの実行
docker run -p 8000:8000 llm-eval-backend
```

詳細なDocker設定については、[DOCKER.md](DOCKER.md)を参照してください。

## APIエンドポイント

### `/api/evaluation/run` (POST)

複数データセットに対するLLM評価を実行します。

#### リクエスト例

```json
{
  "datasets": ["aio", "janli"],
  "num_samples": 10,
  "n_shots": [0, 2],
  "model": {
    "provider": "ollama",
    "model_name": "llama3:latest",
    "max_tokens": 1024,
    "temperature": 0.0,
    "top_p": 1.0,
    "additional_params": {}
  }
}
```

#### レスポンス例

```json
{
  "model_info": {
    "provider": "ollama",
    "model_name": "llama3:latest",
    "max_tokens": 1024,
    "temperature": 0.0,
    "top_p": 1.0,
    "additional_params": {}
  },
  "metrics": {
    "aio_0shot_char_f1": 0.78,
    "aio_0shot_exact_match": 0.35,
    "aio_2shot_char_f1": 0.82,
    "aio_2shot_exact_match": 0.41,
    "janli_0shot_char_f1": 0.65,
    "janli_0shot_exact_match": 0.30,
    "janli_2shot_char_f1": 0.70,
    "janli_2shot_exact_match": 0.33
  }
}
```

## 評価メトリクス

- **char_f1**: 文字ベースのF1スコア（fuzzywuzzyを使用）
- **exact_match**: 完全一致率
- **exact_match_figure**: 図や数値を含む完全一致率
- **contains_answer**: 正解が出力に含まれているかどうか
- **set_f1**: 集合ベースのF1スコア
- **bleu**: BLEUスコア（機械翻訳評価で使用）

## データセットの追加

新しいデータセットを追加するには、以下の形式のJSONファイルを作成し、データセットディレクトリに配置します：

```json
{
  "instruction": "タスクの指示",
  "metrics": ["char_f1", "exact_match"],
  "output_length": 1024,
  "samples": [
    {
      "input": "入力テキスト",
      "output": "正解出力"
    },
    ...
  ]
}
```

## ライセンス

MIT License
