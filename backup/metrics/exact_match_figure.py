"""
数値の完全一致評価指標モジュール
"""
from .base import BaseMetric


class ExactMatchFigure(BaseMetric):
    """
    数値の完全一致評価指標
    """

    def __init__(self):
        """
        初期化メソッド
        """
        super().__init__(name="exact_match_figure")

    def calculate(self, hypothesis: str, reference: str) -> float:
        """
        数値として解釈した場合の完全一致で評価する

        Args:
            hypothesis: モデルの予測出力
            reference: 正解出力

        Returns:
            float: 評価スコア（一致: 1.0, 不一致: 0.0）
        """
        try:
            pred_value = float(hypothesis.strip())
            ref_value = float(reference.strip())
            return float(pred_value == ref_value)
        except ValueError:
            return 0.0