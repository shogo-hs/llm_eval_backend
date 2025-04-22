"""
モデル管理コントローラーモジュール

このモジュールは、異なるLLMプロバイダーのモデル管理を担当します。
特にOllamaの場合、モデルのダウンロード状態の確認や必要に応じたダウンロードを処理します。
"""
import os
import json
import asyncio
import logging
import aiohttp
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urljoin

from app.config import get_settings

# 設定の取得
settings = get_settings()
logger = logging.getLogger(__name__)

class ModelController:
    """モデル管理コントローラークラス"""
    
    def __init__(self):
        """コントローラーの初期化"""
        self.settings = get_settings()
        self.providers = {}
        self.available_models = {}
        self.initialize_providers()
        
    def initialize_providers(self):
        """プロバイダー設定の初期化"""
        for provider in self.settings.ENABLED_PROVIDERS:
            provider_settings = self.settings.get_provider_settings(provider)
            self.providers[provider] = provider_settings
            self.available_models[provider] = []
            
    async def get_ollama_models(self) -> List[Dict[str, Any]]:
        """
        Ollamaから利用可能なモデルリストを取得
        
        Returns:
            利用可能なモデルのリスト
        """
        if "ollama" not in self.providers:
            logger.warning("Ollama provider is not enabled")
            return []
            
        base_url = self.providers["ollama"].get("base_url")
        if not base_url:
            logger.warning("Ollama base URL not configured")
            return []
            
        models_url = base_url.replace("/api/generate", "/api/tags")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(models_url) as response:
                    if response.status == 200:
                        data = await response.json()
                        # 整形されたモデルリストを返す
                        models = data.get("models", [])
                        # キャッシュに保存
                        self.available_models["ollama"] = models
                        return models
                    else:
                        logger.error(f"Failed to get Ollama models. Status: {response.status}")
                        return []
        except Exception as e:
            logger.error(f"Error getting Ollama models: {e}")
            return []
    
    async def check_ollama_model_exists(self, model_name: str) -> bool:
        """
        指定されたOllamaモデルが存在するかチェック
        
        Args:
            model_name: チェックするモデル名
            
        Returns:
            モデルが存在する場合はTrue、そうでない場合はFalse
        """
        # キャッシュに存在しない場合は最新のモデルリストを取得
        if not self.available_models.get("ollama"):
            await self.get_ollama_models()
            
        # モデルリストからチェック
        models = self.available_models.get("ollama", [])
        for model in models:
            if model.get("name") == model_name:
                return True
                
        # タグバージョンを考慮（"phi4:latest" のようなフォーマット）
        if ":" in model_name:
            base_model = model_name.split(":")[0]
            for model in models:
                if model.get("name").startswith(f"{base_model}:"):
                    return True
        
        return False
    
    async def pull_ollama_model(self, model_name: str) -> Tuple[bool, str]:
        """
        Ollamaモデルをダウンロード
        
        Args:
            model_name: ダウンロードするモデル名
            
        Returns:
            (成功したかどうか, メッセージ)のタプル
        """
        if "ollama" not in self.providers:
            return False, "Ollama provider is not enabled"
            
        base_url = self.providers["ollama"].get("base_url")
        if not base_url:
            return False, "Ollama base URL not configured"
            
        pull_url = base_url.replace("/api/generate", "/api/pull")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(pull_url, json={"name": model_name}) as response:
                    if response.status == 200:
                        logger.info(f"Successfully pulled Ollama model: {model_name}")
                        # モデルリストを更新
                        await self.get_ollama_models()
                        return True, f"Successfully pulled model: {model_name}"
                    else:
                        error_msg = await response.text()
                        logger.error(f"Failed to pull Ollama model {model_name}. Status: {response.status}, Error: {error_msg}")
                        return False, f"Failed to pull model: {error_msg}"
        except Exception as e:
            logger.error(f"Error pulling Ollama model {model_name}: {e}")
            return False, f"Error pulling model: {str(e)}"
    
    async def ensure_model_availability(self, provider_name: str, model_name: str) -> Tuple[bool, str]:
        """
        モデルが利用可能であることを確認し、必要に応じてダウンロード
        
        Args:
            provider_name: プロバイダー名
            model_name: モデル名
            
        Returns:
            (利用可能かどうか, メッセージ)のタプル
        """
        if provider_name not in self.settings.ENABLED_PROVIDERS:
            return False, f"Provider {provider_name} is not enabled"
            
        # プロバイダーごとの処理
        if provider_name == "ollama":
            # モデルの存在チェック
            model_exists = await self.check_ollama_model_exists(model_name)
            if model_exists:
                return True, f"Model {model_name} is available"
                
            # 自動ダウンロードが有効な場合
            if self.settings.AUTO_DOWNLOAD_MODELS:
                logger.info(f"Model {model_name} not found. Attempting to download...")
                success, message = await self.pull_ollama_model(model_name)
                return success, message
            else:
                return False, f"Model {model_name} not found and auto-download is disabled"
                
        elif provider_name == "openai" or provider_name == "anthropic":
            # OpenAIとAnthropicはAPIキーが設定されていればモデルが利用可能と仮定
            provider_settings = self.settings.get_provider_settings(provider_name)
            if provider_name == "openai" and provider_settings.get("api_key"):
                return True, f"OpenAI API key is configured"
            elif provider_name == "anthropic" and provider_settings.get("api_key"):
                return True, f"Anthropic API key is configured"
            else:
                return False, f"{provider_name.capitalize()} API key is not configured"
        
        return False, f"Unknown provider: {provider_name}"
    
    async def get_provider_models(self, provider_name: str) -> List[Dict[str, Any]]:
        """
        特定のプロバイダーで利用可能なモデルリストを取得
        
        Args:
            provider_name: プロバイダー名
            
        Returns:
            利用可能なモデルのリスト
        """
        if provider_name not in self.settings.ENABLED_PROVIDERS:
            logger.warning(f"Provider {provider_name} is not enabled")
            return []
            
        if provider_name == "ollama":
            return await self.get_ollama_models()
        elif provider_name == "openai":
            # 実際の実装では、OpenAI APIを呼び出してモデルリストを取得する
            # この例では、一般的なモデルリストを返す
            return [
                {"id": "gpt-4o", "name": "GPT-4o"},
                {"id": "gpt-4-turbo", "name": "GPT-4 Turbo"},
                {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo"},
            ]
        elif provider_name == "anthropic":
            # 実際の実装では、Anthropic APIを呼び出してモデルリストを取得する
            # この例では、一般的なモデルリストを返す
            return [
                {"id": "claude-3-opus-20240229", "name": "Claude 3 Opus"},
                {"id": "claude-3-sonnet-20240229", "name": "Claude 3 Sonnet"},
                {"id": "claude-3-haiku-20240307", "name": "Claude 3 Haiku"},
            ]
        
        return []
    
    async def get_all_available_models(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        全プロバイダーの利用可能なモデルリストを取得
        
        Returns:
            プロバイダー名をキーとするモデルリストの辞書
        """
        result = {}
        for provider in self.settings.ENABLED_PROVIDERS:
            models = await self.get_provider_models(provider)
            result[provider] = models
        
        return result
    
    async def get_model_info(self, provider_name: str, model_name: str) -> Optional[Dict[str, Any]]:
        """
        特定のモデルの詳細情報を取得
        
        Args:
            provider_name: プロバイダー名
            model_name: モデル名
            
        Returns:
            モデル情報の辞書またはNone
        """
        if provider_name not in self.settings.ENABLED_PROVIDERS:
            return None
            
        if provider_name == "ollama":
            models = await self.get_ollama_models()
            for model in models:
                if model.get("name") == model_name:
                    return model
        
        # 他のプロバイダーの実装
        
        return None

# シングルトンパターンでコントローラーを提供
_controller_instance = None

def get_model_controller() -> ModelController:
    """
    モデルコントローラーのシングルトンインスタンスを取得
    
    Returns:
        ModelControllerインスタンス
    """
    global _controller_instance
    if _controller_instance is None:
        _controller_instance = ModelController()
    return _controller_instance
