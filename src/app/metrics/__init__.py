"""
��꯹�ñ��

��꯹�����Մk���WAPI�ЛY�
"""

import os
import importlib
import inspect
from pathlib import Path
from typing import Dict, Any, List, Type

from .base import BaseMetric, register_metric, get_metrics_functions, METRIC_REGISTRY


def _load_metrics() -> None:
    """
    ��꯹ǣ���K��Մk���뒤����Wf{2Y���p
    """
    # �(nǣ���metrics	nѹ�֗
    current_dir = Path(__file__).parent
    
    # ��꯹ǣ���nYyfn.pyա��"
    for file_path in current_dir.glob("*.py"):
        # ա��K������֗
        module_name = file_path.stem
        
        # y�ա�뒹���
        if module_name.startswith("__") or module_name == "base":
            continue
        
        # ���뒤����
        module_path = f"app.metrics.{module_name}"
        try:
            module = importlib.import_module(module_path)
            
            # ����nBaseMetric��Y�Yyfn�鹒"
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (issubclass(obj, BaseMetric) and 
                    obj.__module__ == module.__name__ and 
                    obj is not BaseMetric):
                    
                    # ~`{2U�fDjD4{2ǳ����i(
                    if not any(cls is obj for cls in METRIC_REGISTRY.values()):
                        register_metric(obj)
        
        except (ImportError, AttributeError) as e:
            print(f"Failed to import metric module {module_name}: {e}")


# Bk��꯹��Մk���
_load_metrics()

# APIhWfl�Y��p�	p
__all__ = [
    "BaseMetric", 
    "register_metric", 
    "get_metrics_functions",
    "METRIC_REGISTRY"
]
