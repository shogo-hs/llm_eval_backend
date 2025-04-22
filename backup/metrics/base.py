"""
評価指標の抽象基底クラスを定義するモジュール
"""
from abc import ABC, abstractmethod


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