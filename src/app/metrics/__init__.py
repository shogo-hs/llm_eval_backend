"""
áÈê¯¹ÑÃ±ü¸

áÈê¯¹â¸åüë’êÕ„kíüÉWAPI’Ğ›Y‹
"""

import os
import importlib
import inspect
from pathlib import Path
from typing import Dict, Any, List, Type

from .base import BaseMetric, register_metric, get_metrics_functions, METRIC_REGISTRY


def _load_metrics() -> None:
    """
    áÈê¯¹Ç£ì¯ÈêK‰êÕ„kâ¸åüë’¤óİüÈWf{2Y‹…è¢p
    """
    # ş(nÇ£ì¯Èêmetrics	nÑ¹’Ö—
    current_dir = Path(__file__).parent
    
    # áÈê¯¹Ç£ì¯Èê…nYyfn.pyÕ¡¤ë’"
    for file_path in current_dir.glob("*.py"):
        # Õ¡¤ëK‰â¸åüë’Ö—
        module_name = file_path.stem
        
        # yŠÕ¡¤ë’¹­Ã×
        if module_name.startswith("__") or module_name == "base":
            continue
        
        # â¸åüë’¤óİüÈ
        module_path = f"app.metrics.{module_name}"
        try:
            module = importlib.import_module(module_path)
            
            # â¸åüë…nBaseMetric’™Y‹Yyfn¯é¹’"
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (issubclass(obj, BaseMetric) and 
                    obj.__module__ == module.__name__ and 
                    obj is not BaseMetric):
                    
                    # ~`{2UŒfDjD4{2Ç³ìü¿’i(
                    if not any(cls is obj for cls in METRIC_REGISTRY.values()):
                        register_metric(obj)
        
        except (ImportError, AttributeError) as e:
            print(f"Failed to import metric module {module_name}: {e}")


# BkáÈê¯¹’êÕ„kíüÉ
_load_metrics()

# APIhWfl‹Y‹¢pû	p
__all__ = [
    "BaseMetric", 
    "register_metric", 
    "get_metrics_functions",
    "METRIC_REGISTRY"
]
