# app/utils/logging.py

import mlflow
from mlflow.entities import ViewType
from typing import Dict


def log_evaluation_results(
    model_name: str,
    metrics: Dict[str, float]
) -> None:
    """
    LLM評価結果をMLflowにロギングする関数。

    同じモデル名のRunを再利用し、新しいメトリクスは列として自動追加されます。

    Args:
        model_name (str): ログ対象のモデル名。Run Nameとして利用します。
        metrics (Dict[str, float]): ロギングするメトリクスの辞書。
            キーは "{dataset}_{shot}shot_{metric_name}" 形式、
            値は対応するスコア値を指定します。

    Returns:
        None
    """
    # トラッキングサーバー設定
    mlflow.set_tracking_uri("http://localhost:5000")
    experiment_name = "model_evaluation"
    mlflow.set_experiment(experiment_name)

    # 既存Runを検索（param.model_nameでフィルタ）
    experiment = mlflow.get_experiment_by_name(experiment_name)
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"params.model_name = '{model_name}'",
        run_view_type=ViewType.ACTIVE_ONLY
    )

    # Runの開始または再利用
    if runs.empty:
        run = mlflow.start_run(run_name=model_name)
        mlflow.log_param("model_name", model_name)
    else:
        run_id = runs.iloc[0].run_id
        run = mlflow.start_run(run_id=run_id)

    # メトリクスをログ
    for key, value in metrics.items():
        mlflow.log_metric(key, value)

    mlflow.end_run()
