"""
設定管理モジュール
"""
from pathlib import Path
from functools import lru_cache
from pydantic import BaseSettings
from typing import Dict, Any, Optional, List


class Settings(BaseSettings):
    """
    アプリケーション設定

    アプリケーション全体で使用する設定パラメータを管理するクラス
    """
    # データセットディレクトリパス
    DATASET_DIR: Path = Path("/work/hasegawa/llm_eval/datasets/jaster/1.3.0/evaluation/test/")
    TRAIN_DIR: Path = Path("/work/hasegawa/llm_eval/datasets/jaster/1.3.0/evaluation/train/")
    RESULTS_DIR: Path = Path("../../../results/")

    # APIとモデル関連
    LITELLM_BASE_URL: str = "http://192.168.101.204:11434/api/generate"
    DEFAULT_MAX_TOKENS: int = 1024
    DEFAULT_TEMPERATURE: float = 0.0
    DEFAULT_TOP_P: float = 1.0
    # タイムアウト設定（秒）
    MODEL_TIMEOUT: float = 60.0
    # 再試行設定
    MAX_RETRIES: int = 3
    RETRY_BACKOFF_MIN: float = 2.0
    RETRY_BACKOFF_MAX: float = 30.0
    RETRY_BACKOFF_MULTIPLIER: float = 1.5

    # モデル管理設定
    AUTO_DOWNLOAD_MODELS: bool = True  # モデルの自動ダウンロード設定
    MODEL_CHECK_BEFORE_CALL: bool = True  # 呼び出し前にモデルの存在をチェックするかどうか

    # プロバイダ設定
    ENABLED_PROVIDERS: List[str] = ["ollama", "openai", "anthropic"]
    DEFAULT_PROVIDER: str = "ollama"  # デフォルトプロバイダー
    FALLBACK_PROVIDERS: List[str] = []  # フォールバックプロバイダー（順番に試行）
    
    # キャッシュ設定
    ENABLE_LITELLM_CACHE: bool = True
    CACHE_EXPIRATION: int = 3600  # 秒

    # 処理設定
    BATCH_SIZE: int = 5
    DEFAULT_NUM_SAMPLES: int = 10
    DEFAULT_N_SHOTS: list = [0, 2]
    
    # ロギング設定
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # 環境設定
    PROJECT_ROOT: Optional[Path] = None
    ENV: str = "development"  # development, staging, production
    
    class Config:
        """Pydantic設定クラス"""
        env_prefix = "LLMEVAL_"  # 環境変数のプレフィックス
        env_file = ".env"  # 環境変数ファイル
        env_file_encoding = "utf-8"

    def initialize_dirs(self) -> None:
        """
        必要なディレクトリを初期化する
        """
        self.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        
        # プロジェクトルートが設定されている場合、結果ディレクトリを調整
        if self.PROJECT_ROOT:
            self.RESULTS_DIR = self.PROJECT_ROOT / "results"
            self.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def get_provider_settings(self, provider_name: str) -> Dict[str, Any]:
        """
        特定プロバイダの設定を取得する

        Args:
            provider_name: プロバイダ名

        Returns:
            設定辞書
        """
        # プロバイダ固有の設定を返す
        if provider_name == "openai":
            return {
                "base_url": self.OPENAI_API_BASE,
                "api_key": self.OPENAI_API_KEY
            }
        elif provider_name == "anthropic":
            return {
                "api_key": self.ANTHROPIC_API_KEY
            }
        elif provider_name == "ollama":
            return {
                "base_url": self.LITELLM_BASE_URL
            }
        return {}


@lru_cache()
def get_settings() -> Settings:
    """
    設定シングルトンインスタンスを取得

    Returns:
        Settings: 設定インスタンス
    """
    settings = Settings()
    settings.initialize_dirs()
    return settings
