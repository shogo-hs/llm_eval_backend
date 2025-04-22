from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any

from app.api.models import EvaluationRequest, EvaluationResponse
from app.core.evaluation import run_multiple_evaluations
from app.utils.logging import log_evaluation_results
from app.utils.litellm_helper import get_provider_options

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
        # リクエストからモデル情報と追加パラメータを取得
        provider_name = request.model.provider
        model_name = request.model.model_name
        
        # 追加パラメータを準備（プロバイダごとのデフォルト設定を適用）
        additional_params = get_provider_options(provider_name)
        
        # ユーザー指定の追加パラメータを適用（優先）
        if request.model.additional_params:
            # ヘッダーの場合は更新
            if "headers" in additional_params and "headers" in request.model.additional_params:
                additional_params["headers"].update(request.model.additional_params["headers"])
                # ヘッダーをリクエストの追加パラメータから削除（重複適用を避けるため）
                user_params = request.model.additional_params.copy()
                user_params.pop("headers", None)
                additional_params.update(user_params)
            else:
                # ヘッダーがない場合はそのまま更新
                additional_params.update(request.model.additional_params)
        
        # 1) 評価エンジン呼び出し
        results_full: Dict[str, Any] = await run_multiple_evaluations(
            datasets=request.datasets,
            provider_name=provider_name,
            model_name=model_name,
            num_samples=request.num_samples,
            n_shots=request.n_shots,
            additional_params=additional_params
        )

        # 2) フラットなメトリクス辞書を作成
        flat_metrics: Dict[str, float] = {}
        for ds, ds_res in results_full.get("results", {}).items():
            details = ds_res.get("details", {})
            for key, value in details.items():
                if key.endswith("_details") or key.endswith("_error_rate"):
                    continue
                flat_metrics[key] = value  # 例: "aio_0shot_char_f1": 0.11
                
                # エラー率も追加
                error_rate_key = f"{key}_error_rate"
                if error_rate_key in details:
                    flat_metrics[error_rate_key] = details[error_rate_key]

        # 3) バックグラウンドでMLflowへログ
        background_tasks.add_task(
            log_evaluation_results,
            model_name=f"{provider_name}/{model_name}",
            metrics=flat_metrics
        )

        # 4) シンプルレスポンス返却
        return EvaluationResponse(
            model_info=request.model,
            metrics=flat_metrics
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
