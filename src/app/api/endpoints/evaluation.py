from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any

from app.api.models import EvaluationRequest, EvaluationResponse
from app.core.evaluation import run_multiple_evaluations
from app.utils.logging import log_evaluation_results

router = APIRouter()


@router.post("/run", response_model=EvaluationResponse)
async def evaluate(
    request: EvaluationRequest,
    background_tasks: BackgroundTasks
) -> EvaluationResponse:
    """
    複数データセットに対するLLM評価を実行し、
    使用したモデル情報とフラットメトリクスのみを返却するシンプルエンドポイント。

    Args:
        request (EvaluationRequest): 評価対象データセット、モデル設定、サンプル数、few-shot数
        background_tasks (BackgroundTasks): バックグラウンドタスク登録用

    Returns:
        EvaluationResponse: 使用モデル情報とメトリクスの辞書
    """
    try:
        # 1) 評価エンジン呼び出し
        results_full: Dict[str, Any] = await run_multiple_evaluations(
            datasets=request.datasets,
            provider_name=request.model.provider,
            model_name=request.model.model_name,
            num_samples=request.num_samples,
            n_shots=request.n_shots
        )

        # 2) フラットなメトリクス辞書を作成
        flat_metrics: Dict[str, float] = {}
        for ds, ds_res in results_full.get("results", {}).items():
            details = ds_res.get("details", {})
            for key, value in details.items():
                if key.endswith("_details"):
                    continue
                flat_metrics[key] = value  # 例: "aio_0shot_char_f1": 0.11

        # 3) バックグラウンドでMLflowへログ
        background_tasks.add_task(
            log_evaluation_results,
            model_name=request.model.model_name,
            metrics=flat_metrics
        )

        # 4) シンプルレスポンス返却
        return EvaluationResponse(
            model_info=request.model,
            metrics=flat_metrics
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))