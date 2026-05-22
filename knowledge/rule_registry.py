# knowledge/rule_registry.py
import inspect
import sys
from typing import Dict, List, Callable, Optional

# 全局注册表
_RULE_REGISTRY: Dict[str, Callable] = {}
_RULE_ID_MAP: Dict[str, int] = {}   # rule_name -> int id
_RULE_NAME_LIST: List[str] = []
_LAST_ID = -1

def register_rule(rule_name: Optional[str] = None):
    """
    装饰器：将函数注册为可用的积分规则。
    用法:
        @register_rule()
        def rule_power_integral(integral): ...
    或:
        @register_rule("my_custom_name")
        def some_func(integral): ...
    """
    def decorator(func):
        nonlocal rule_name
        name = rule_name if rule_name is not None else func.__name__
        if name in _RULE_REGISTRY:
            # 允许重复注册（热重载时），但打印警告
            print(f"⚠️ 规则 {name} 已存在，将被覆盖")
        _RULE_REGISTRY[name] = func
        return func
    return decorator

def build_action_space():
    """
    根据当前注册的函数列表，构建动作空间（ID映射和列表）。
    在每次热重载后调用。
    """
    global _RULE_ID_MAP, _RULE_NAME_LIST, _LAST_ID
    names = list(_RULE_REGISTRY.keys())
    # 保持字典序确定性
    names.sort()
    _RULE_NAME_LIST = names
    _RULE_ID_MAP = {name: idx for idx, name in enumerate(names)}
    _LAST_ID = len(names) - 1
    return _RULE_NAME_LIST, _RULE_ID_MAP

def get_rule_by_name(name: str) -> Callable:
    return _RULE_REGISTRY[name]

def get_rule_by_id(rule_id: int) -> Callable:
    name = _RULE_NAME_LIST[rule_id]
    return _RULE_REGISTRY[name]

def get_rule_id(name: str) -> int:
    return _RULE_ID_MAP[name]

def get_all_rule_names() -> List[str]:
    return _RULE_NAME_LIST.copy()

def get_num_rules() -> int:
    return len(_RULE_NAME_LIST)

def is_valid_rule_id(rule_id: int) -> bool:
    return 0 <= rule_id < len(_RULE_NAME_LIST)

def clear_registry():
    """清空注册表（用于热重载前的清理）"""
    global _RULE_REGISTRY, _RULE_ID_MAP, _RULE_NAME_LIST, _LAST_ID
    _RULE_REGISTRY.clear()
    _RULE_ID_MAP.clear()
    _RULE_NAME_LIST.clear()
    _LAST_ID = -1

def reload_module(module_name: str = "knowledge.rules"):
    """
    热重载规则模块：清空注册表，重新导入模块，重建动作空间。
    """
    clear_registry()
    if module_name in sys.modules:
        import importlib
        importlib.reload(sys.modules[module_name])
    else:
        import importlib
        importlib.import_module(module_name)
    build_action_space()
    print(f"✅ 热重载完成，当前动作空间大小: {get_num_rules()}")

# 兼容旧代码的接口（供训练时使用）
RULE_NAMES = _RULE_NAME_LIST
RULE_DICT = _RULE_REGISTRY

# 如果直接运行本文件，可以测试
if __name__ == "__main__":
    @register_rule()
    def dummy_rule(integral):
        return (integral, "rewrite")
    build_action_space()
    print("RULE_NAMES:", RULE_NAMES)
    print("RULE_DICT:", RULE_DICT)