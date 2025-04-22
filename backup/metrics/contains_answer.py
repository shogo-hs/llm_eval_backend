"""
回答を含むかどうかの評価指標モジュール
"""
from .base import BaseMetric


class ContainsAnswer(BaseMetric):
    """
    回答を含むかどうかの評価指標
    """

    def __init__(self):
        """
        初期化メソッド
        """
        super().__init__(name="contains_answer")

    def calculate(self, hypothesis: str, reference: str) -> float:
        """
        予測文字列が正解を含んでいるかどうかで評価する

        Args:
            hypothesis: モデルの予測出力
            reference: 正解出力

        Returns:
            float: 評価スコア（含む: 1.0, 含まない: 0.0）
        """
        return float(reference.strip() in hypothesis.strip())