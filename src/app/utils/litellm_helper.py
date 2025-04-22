"""
LiteLLM関連のユーティリティ関数とヘルパークラス
"""
import logging
import json
import os
import hashlib
from typing import Dict, Any, List, Optional, Union
import litellm
from app.config import get_settings

# 設定の取得
settings = get_settings()
logger = logging.getLogger(__name__)

def init_litellm_cache():
    """
    LiteLLMのキャッシュを初期化する関数
    
    キャッシュが有効な場合はLiteLLMのキャッシュを有効化します
    """
    if settings.ENABLE_LITELLM_CACHE:
        logger.info("Initializing LiteLLM cache")
        litellm.cache = litellm.Cache(
            type="redis",
            host="localhost",
            port=6379,
            ttl=settings.CACHE_EXPIRATION
        )
        logger.info("LiteLLM cache initialized")
    else:
        logger.info("LiteLLM cache is disabled")

def generate_cache_key(messages: List[Dict[str, str]], model: str) -> str:
    """
    キャッシュキーを生成する関数
    
    Args:
        messages: メッセージのリスト
        model: モデル名
        
    Returns:
        キャッシュキーの文字列
    """
    # シンプルなキャッシュキー生成
    key_data = {
        "model": model,
        "messages": messages
    }
    key_str = json.dumps(key_data, sort_keys=True)
    return hashlib.md5(key_str.encode()).hexdigest()

def check_cache(messages: List[Dict[str, str]], model: str) -> Optional[str]:
    """
    キャッシュをチェックする関数
    
    Args:
        messages: メッセージのリスト
        model: モデル名
        
    Returns:
        キャッシュが存在する場合はキャッシュされた出力、そうでない場合はNone
    """
    if not settings.ENABLE_LITELLM_CACHE:
        return None
    
    key = generate_cache_key(messages, model)
    
    try:
        cache_path = f"cache/{key}.json"
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                logger.info(f"Cache hit for key {key}")
                return data.get("content")
    except Exception as e:
        logger.warning(f"Error checking cache: {e}")
    
    return None

def update_cache(messages: List[Dict[str, str]], model: str, content: str) -> None:
    """
    キャッシュを更新する関数
    
    Args:
        messages: メッセージのリスト
        model: モデル名
        content: キャッシュする内容
    """
    if not settings.ENABLE_LITELLM_CACHE:
        return
    
    key = generate_cache_key(messages, model)
    
    try:
        os.makedirs("cache", exist_ok=True)
        cache_path = f"cache/{key}.json"
        data = {
            "model": model,
            "content": content
        }
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        logger.info(f"Cache updated for key {key}")
    except Exception as e:
        logger.warning(f"Error updating cache: {e}")

def get_provider_options(provider_name: str) -> Dict[str, Any]:
    """
    特定のプロバイダのオプションを取得する関数
    
    Args:
        provider_name: プロバイダ名
        
    Returns:
        プロバイダオプションの辞書
    """
    # プロバイダごとの設定を取得
    provider_settings = settings.get_provider_settings(provider_name)
    
    # プロバイダごとのデフォルトヘッダーやオプションを設定
    default_options = {
        "openai": {
            "headers": {
                "User-Agent": "LLM-Evaluation-Tool/1.0"
            }
        },
        "anthropic": {
            "headers": {
                "User-Agent": "LLM-Evaluation-Tool/1.0",
                "anthropic-version": "2023-06-01"
            }
        },
        "ollama": {
            "headers": {
                "User-Agent": "LLM-Evaluation-Tool/1.0"
            }
        }
    }
    
    # プロバイダがサポートされているか確認
    if provider_name not in default_options:
        logger.warning(f"Provider {provider_name} is not supported. Using default options.")
        return {}
    
    # ベースとなるオプションを取得
    options = default_options.get(provider_name, {}).copy()
    
    # 設定から追加のパラメータを適用
    if provider_settings:
        # ヘッダーがある場合は更新
        if "headers" in options and "headers" in provider_settings:
            options["headers"].update(provider_settings["headers"])
        # その他の設定を更新
        for key, value in provider_settings.items():
            if key != "headers":
                options[key] = value
    
    return options

def format_litellm_model_name(provider_name: str, model_name: str) -> str:
    """
    LiteLLM形式のモデル名を生成する関数
    
    Args:
        provider_name: プロバイダ名
        model_name: モデル名
        
    Returns:
        LiteLLM形式のモデル名
    """
    return f"{provider_name}/{model_name}"
