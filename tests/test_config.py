"""
設定モジュールのテスト
"""
import pytest
from pathlib import Path
from app.config import get_settings, Settings


def test_settings_singleton():
    """設定のシングルトンパターンが機能することを確認する"""
    settings1 = get_settings()
    settings2 = get_settings()
    
    assert settings1 is settings2, "get_settings()は同じインスタンスを返すべき"


def test_settings_default_values():
    """設定のデフォルト値が正しく設定されていることを確認する"""
    settings = get_settings()
    
    # デフォルト値の確認
    assert isinstance(settings.DATASET_DIR, Path)
    assert isinstance(settings.TRAIN_DIR, Path)
    assert isinstance(settings.RESULTS_DIR, Path)
    
    assert settings.BATCH_SIZE == 5
    assert settings.DEFAULT_NUM_SAMPLES == 10
    assert settings.DEFAULT_N_SHOTS == [0, 2]
    
    assert settings.LITELLM_BASE_URL == "http://192.168.101.204:11434/api/generate"
    assert settings.DEFAULT_MAX_TOKENS == 1024
    assert settings.DEFAULT_TEMPERATURE == 0.0
    assert settings.DEFAULT_TOP_P == 1.0
    
    assert settings.LOG_LEVEL == "INFO"
    assert settings.ENV == "development"


def test_initialize_dirs():
    """ディレクトリ初期化関数が正しく動作することを確認する"""
    settings = Settings()
    
    # 結果ディレクトリが存在しない場合を模擬
    test_results_dir = Path("/tmp/test_results")
    settings.RESULTS_DIR = test_results_dir
    
    if test_results_dir.exists():
        import shutil
        shutil.rmtree(test_results_dir)
    
    assert not test_results_dir.exists()
    
    # 初期化実行
    settings.initialize_dirs()
    
    # ディレクトリが作成されたことを確認
    assert test_results_dir.exists()
    assert test_results_dir.is_dir()
    
    # テスト後の後片付け
    import shutil
    shutil.rmtree(test_results_dir)
