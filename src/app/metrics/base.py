"""
評価指標の抽象基底クラスと登録機能を定義するモジュール
"""
from abc import ABC, abstractmethod
from typing import Dict, Type, Callable, Any


# メトリクスクラスの登録用ディクショナリ
METRIC_REGISTRY: Dict[str, Type["BaseMetric"]] = {}


class BaseMetric(ABC):
    """
    評価指標の抽象基底クラス

    全ての評価指標はこのクラスを継承して実装する必要がある
    """

    def __init__(self, name: str):
        """
        初期化メソッド

        Args:
            name: 評価指標の名前
        """
        self.name = name

    @abstractmethod
    def calculate(self, hypothesis: str, reference: str) -> float:
        """
        評価スコアを計算する

        Args:
            hypothesis: モデルの予測出力
            reference: 正解出力

        Returns:
            float: 評価スコア
        """
        pass

    def __str__(self) -> str:
        return f"{self.name}"


def register_metric(cls: Type[BaseMetric]) -> Type[BaseMetric]:
    """
    評価指標クラスを登録するデコレータ

    Args:
        cls: 登録する評価指標クラス

    Returns:
        評価指標クラス（デコレータパターン）
    """
    # インスタンスを作成してnameプロパティを取得
    instance = cls()
    name = instance.name
    
    if name in METRIC_REGISTRY:
        raise ValueError(f"Metric '{name}' is already registered")
    
    METRIC_REGISTRY[name] = cls
    return cls


def get_metrics_functions() -> Dict[str, Callable[[str, str], float]]:
    """
    すべての登録済みメトリクス計算関数のディクショナリを取得

    Returns:
        Dict[str, Callable]: メトリクス名をキー、計算関数を値とするディクショナリ
    """
    metrics_funcs = {}
    for name, metric_cls in METRIC_REGISTRY.items():
        metrics_funcs[name] = metric_cls().calculate
    return metrics_funcs
