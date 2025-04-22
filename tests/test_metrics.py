"""
評価メトリクスのテスト
"""
import pytest
from app.metrics.char_f1 import CharF1
from app.metrics.exact_match import ExactMatch


def test_char_f1():
    """
    CharF1メトリクスのテスト
    """
    metric = CharF1()
    
    # 完全一致
    assert metric.calculate("これはテストです。", "これはテストです。") == 1.0
    
    # 部分一致
    assert 0 < metric.calculate("これはテストです。", "これはテストだ。") < 1.0
    
    # 全く異なる
    assert metric.calculate("これはテストです。", "全く関係のない文章。") < 0.5
    
    # 空入力
    assert metric.calculate("", "") == 1.0
    assert metric.calculate("テスト", "") == 0.0
    assert metric.calculate("", "テスト") == 0.0


def test_exact_match():
    """
    ExactMatchメトリクスのテスト
    """
    metric = ExactMatch()
    
    # 完全一致
    assert metric.calculate("これはテストです。", "これはテストです。") == 1.0
    
    # スペースや改行の違いは無視
    assert metric.calculate("これはテストです。", "これはテストです。\n") == 1.0
    assert metric.calculate("これはテストです。", " これはテストです。 ") == 1.0
    
    # 部分一致は不一致
    assert metric.calculate("これはテストです。", "これはテストだ。") == 0.0
    
    # 空入力
    assert metric.calculate("", "") == 1.0
    assert metric.calculate("テスト", "") == 0.0
    assert metric.calculate("", "テスト") == 0.0
