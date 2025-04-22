"""
集合ベースのF1スコア評価指標モジュール
"""
from ..base import BaseMetric


class SetF1(BaseMetric):
    """
    集合ベースのF1スコア評価指標
    """

    def __init__(self, delimiter: str = "\n"):
        """
        初期化メソッド

        Args:
            delimiter: 項目の区切り文字
        """
        super().__init__(name="set_f1")
        self.delimiter = delimiter

    def calculate(self, hypothesis: str, reference: str) -> float:
        """
        集合ベースのF1スコアで評価する

        Args:
            hypothesis: モデルの予測出力
            reference: 正解出力

        Returns:
            float: 評価スコア（0.0-1.0）
        """
        set_pred = {x.strip() for x in hypothesis.split(self.delimiter) if x.strip()}
        set_ref = {x.strip() for x in reference.split(self.delimiter) if x.strip()}

        if not set_pred and not set_ref:
            return 1.0
        if not set_pred or not set_ref:
            return 0.0

        true_positives = len(set_pred.intersection(set_ref))
        precision = true_positives / len(set_pred)
        recall = true_positives / len(set_ref)

        if precision + recall == 0:
            return 0.0

        f1 = 2 * precision * recall / (precision + recall)
        return f1