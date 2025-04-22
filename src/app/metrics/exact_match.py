"""
完全一致評価指標モジュール
"""
from .base import BaseMetric


class ExactMatch(BaseMetric):
    """
    完全一致評価指標
    """

    def __init__(self):
        """
        初期化メソッド
        """
        super().__init__(name="exact_match")

    def calculate(self, hypothesis: str, reference: str) -> float:
        """
        完全一致で評価する

        Args:
            hypothesis: モデルの予測出力
            reference: 正解出力

        Returns:
            float: 評価スコア（一致: 1.0, 不一致: 0.0）
        """
        return float(hypothesis.strip() == reference.strip())