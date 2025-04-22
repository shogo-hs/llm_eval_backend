# ベースイメージとしてPython 3.12を使用
FROM python:3.12-slim

# 作業ディレクトリを設定
WORKDIR /app

# 環境変数の設定
ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 必要なパッケージをインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# uv（Python パッケージマネージャ）のインストール
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"
RUN pip install uv

# Pythonプロジェクトファイルをコピー
COPY pyproject.toml README.md ./

# 依存関係をインストール（追加の依存関係も含む）
RUN uv init && \
    uv pip install --system "sacrebleu>=2.3.1" "pytest>=7.4.0" "python-levenshtein>=0.21.1" && \
    uv pip install --system -e .

# アプリケーションのソースコードをコピー
COPY src ./src

# ポートの公開
EXPOSE 8000

# アプリケーションの起動
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
