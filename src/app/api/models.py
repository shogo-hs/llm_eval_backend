from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class ModelConfig(BaseModel):
    """モデル設定"""
    provider: str
    model_name: str
    max_tokens: int
    temperature: Optional[float] = 0.0
    top_p: Optional[float] = 1.0
    additional_params: Optional[Dict[str, Any]] = Field(default_factory=dict)

class EvaluationRequest(BaseModel):
    """評価リクエストモデル"""
    datasets: List[str]           # 評価対象のデータセット名一覧
    num_samples: int              # 評価サンプル数
    n_shots: List[int]            # few-shot数リスト
    model: ModelConfig            # フィールド名を model に変更

class EvaluationResponse(BaseModel):
    """評価レスポンスモデル"""
    model_info: ModelConfig       # 使用したモデル情報
    metrics: Dict[str, float]     # フラットメトリクス辞書
