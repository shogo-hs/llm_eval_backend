"""
設定管理モジュール
"""
from pathlib import Path
from functools import lru_cache
from pydantic import BaseSettings
from typing import Dict, Any, Optional


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
