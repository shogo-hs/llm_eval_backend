"""
文字ベースのF1スコア評価指標モジュール
"""
from .base import BaseMetric
from fuzzywuzzy import fuzz


class CharF1(BaseMetric):
    """
    文字ベースのF1スコア評価指標
    """

    def __init__(self):
        """
        初期化メソッド
        """
        super().__init__(name="char_f1")

    def calculate(self, hypothesis: str, reference: str) -> float:
        """
        文字ベースのF1スコアで評価する

        Args:
            hypothesis: モデルの予測出力
            reference: 正解出力

        Returns:
            float: 評価スコア（0.0-1.0）
        """
        return fuzz.token_sort_ratio(hypothesis.strip(), reference.strip()) / 100.0