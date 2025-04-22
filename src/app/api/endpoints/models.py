"""
モデル管理のインターフェースモジュール
"""
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends

from app.utils.model_management.controller import get_model_controller, ModelController

router = APIRouter()

@router.get("/providers")
async def list_providers():
    """
    有効なプロバイダーのリストを取得するエンドポイント
    """
    controller = get_model_controller()
    return {
        "providers": controller.settings.ENABLED_PROVIDERS
    }

@router.get("/models/{provider_name}")
async def list_models(provider_name: str):
    """
    特定のプロバイダーの利用可能なモデルリストを取得するエンドポイント
    
    Args:
        provider_name: プロバイダー名
    """
    controller = get_model_controller()
    
    if provider_name not in controller.settings.ENABLED_PROVIDERS:
        raise HTTPException(status_code=404, detail=f"Provider {provider_name} not found or not enabled")
    
    models = await controller.get_provider_models(provider_name)
    return {
        "provider": provider_name,
        "models": models
    }

@router.get("/models")
async def list_all_models():
    """
    全プロバイダーの利用可能なモデルリストを取得するエンドポイント
    """
    controller = get_model_controller()
    models = await controller.get_all_available_models()
    return {
        "models": models
    }

@router.get("/check/{provider_name}/{model_name}")
async def check_model_availability(provider_name: str, model_name: str):
    """
    モデルの利用可能性をチェックするエンドポイント
    
    Args:
        provider_name: プロバイダー名
        model_name: モデル名
    """
    controller = get_model_controller()
    
    if provider_name not in controller.settings.ENABLED_PROVIDERS:
        raise HTTPException(status_code=404, detail=f"Provider {provider_name} not found or not enabled")
    
    available, message = await controller.ensure_model_availability(provider_name, model_name)
    
    return {
        "provider": provider_name,
        "model": model_name,
        "available": available,
        "message": message
    }

@router.post("/download/ollama/{model_name}")
async def download_ollama_model(model_name: str):
    """
    Ollamaモデルをダウンロードするエンドポイント
    
    Args:
        model_name: ダウンロードするモデル名
    """
    controller = get_model_controller()
    
    if "ollama" not in controller.settings.ENABLED_PROVIDERS:
        raise HTTPException(status_code=404, detail="Ollama provider not found or not enabled")
    
    # モデルが既に存在するかチェック
    model_exists = await controller.check_ollama_model_exists(model_name)
    if model_exists:
        return {
            "success": True,
            "message": f"Model {model_name} is already downloaded"
        }
    
    # モデルをダウンロード
    success, message = await controller.pull_ollama_model(model_name)
    
    if not success:
        raise HTTPException(status_code=500, detail=message)
    
    return {
        "success": success,
        "message": message
    }

@router.get("/info/{provider_name}/{model_name}")
async def get_model_info(provider_name: str, model_name: str):
    """
    モデル情報を取得するエンドポイント
    
    Args:
        provider_name: プロバイダー名
        model_name: モデル名
    """
    controller = get_model_controller()
    
    if provider_name not in controller.settings.ENABLED_PROVIDERS:
        raise HTTPException(status_code=404, detail=f"Provider {provider_name} not found or not enabled")
    
    model_info = await controller.get_model_info(provider_name, model_name)
    
    if not model_info:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    return {
        "provider": provider_name,
        "model": model_name,
        "info": model_info
    }
