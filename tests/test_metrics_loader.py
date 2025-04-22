"""
メトリクス動的ロードのテスト
"""
import pytest
from app.metrics import get_metrics_functions, METRIC_REGISTRY
from app.metrics.base import BaseMetric, register_metric


def test_metrics_registration():
    """メトリクスが正しく登録されているかテスト"""
    metrics = get_metrics_functions()
    
    # 最低限含まれるべきメトリクス
    required_metrics = ["char_f1", "exact_match", "bleu", "contains_answer", 
                        "exact_match_figure", "set_f1"]
    
    for metric in required_metrics:
        assert metric in metrics, f"メトリクス '{metric}' が登録されていません"
        assert callable(metrics[metric]), f"メトリクス '{metric}' は呼び出し可能ではありません"


def test_registry_structure():
    """メトリクスレジストリの構造をテスト"""
    # レジストリが空でないことを確認
    assert len(METRIC_REGISTRY) > 0, "メトリクスレジストリが空です"
    
    # 全てのエントリが正しい型であることを確認
    for name, cls in METRIC_REGISTRY.items():
        assert isinstance(name, str), f"メトリクス名 '{name}' は文字列ではありません"
        assert issubclass(cls, BaseMetric), f"メトリクスクラス '{cls.__name__}' はBaseMetricのサブクラスではありません"
        
        # インスタンス化して名前とクラスが一致することを確認
        instance = cls()
        assert instance.name == name, f"メトリクス '{cls.__name__}' のnameプロパティ '{instance.name}' がレジストリキー '{name}' と一致しません"


def test_custom_metric_registration():
    """カスタムメトリクスを動的に登録できることをテスト"""
    # テスト用のカスタムメトリクスを定義
    @register_metric
    class TestMetric(BaseMetric):
        def __init__(self):
            super().__init__(name="test_metric")
            
        def calculate(self, hypothesis, reference):
            return 1.0 if hypothesis == reference else 0.0
    
    # 登録されたことを確認
    metrics = get_metrics_functions()
    assert "test_metric" in metrics, "カスタムメトリクスが登録されていません"
    
    # 正しく計算できることを確認
    assert metrics["test_metric"]("test", "test") == 1.0
    assert metrics["test_metric"]("test", "different") == 0.0


def test_metrics_functionality():
    """登録されたメトリクスが実際に機能するかテスト"""
    metrics = get_metrics_functions()
    
    # char_f1のテスト
    if "char_f1" in metrics:
        assert metrics["char_f1"]("テスト", "テスト") == 1.0
        assert 0.0 < metrics["char_f1"]("テスト", "テスト文字列") < 1.0
    
    # exact_matchのテスト
    if "exact_match" in metrics:
        assert metrics["exact_match"]("テスト", "テスト") == 1.0
        assert metrics["exact_match"]("テスト", "違う文字列") == 0.0
    
    # contains_answerのテスト
    if "contains_answer" in metrics:
        assert metrics["contains_answer"]("これはテスト文字列です", "テスト") == 1.0
        assert metrics["contains_answer"]("これは別の文字列です", "テスト") == 0.0
