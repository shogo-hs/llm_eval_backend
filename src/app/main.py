from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import evaluation, models
from app.utils.model_management import get_model_controller

app = FastAPI(title="Model Evaluation API")

# CORS設定（React Appとの連携用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番環境では適切に設定する
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# モデル管理コントローラーの初期化
model_controller = get_model_controller()

# ルーターの登録
app.include_router(evaluation.router, prefix="/api/evaluation", tags=["evaluation"])
app.include_router(models.router, prefix="/api/models", tags=["models"])

# アプリ起動イベント
@app.on_event("startup")
async def startup_event():
    # アプリケーション起動時に実行する処理
    # モデルの利用可能性確認やキャッシュの初期化など
    from app.utils.litellm_helper import init_litellm_cache
    init_litellm_cache()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
