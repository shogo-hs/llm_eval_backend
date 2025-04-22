"""
BLEUスコア評価指標モジュール
"""
from .base import BaseMetric


class BLEUScore(BaseMetric):
    """
    BLEUスコア評価指標
    """

    def __init__(self):
        """
        初期化メソッド
        """
        super().__init__(name="bleu")
        try:
            from sacrebleu import BLEU
            self.bleu = BLEU()
        except ImportError:
            raise ImportError("sacrebleu is required for BLEUScore metric")

    def calculate(self, hypothesis: str, reference: str) -> float:
        """
        BLEUスコアで評価する

        Args:
            hypothesis: モデルの予測出力
            reference: 正解出力

        Returns:
            float: 評価スコア（0.0-1.0）
        """
        return self.bleu.sentence_score(hypothesis.strip(), [reference.strip()]).score / 100.0