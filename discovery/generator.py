# discovery/generator.py
import os
import re
import ast
import sys
import shutil
import importlib
from collections import defaultdict
from typing import List, Tuple, Optional

# 全局宏动作使用计数器（用于淘汰）
macro_usage_counter = defaultdict(int)

def generate_macro_rule_code(rule_combination_names: List[str], new_rule_id: int) -> Tuple[str, str]:
    """
    生成组合规则函数的代码字符串。
    rule_combination_names: 规则名称列表，如 ["TrigProductToSum", "ExtractConstant"]
    new_rule_id: 整数 ID，用于生成唯一函数名
    返回 (new_rule_name, code_string)
    """
    new_rule_name = f"rule_auto_macro_{new_rule_id}"
    code_lines = [
        f"def {new_rule_name}(integral):",
        f"    \"\"\"Automatically generated macro rule (ID: {new_rule_id}).\"\"\"",
        f"    from discovery.generator import macro_usage_counter",
        f"    macro_usage_counter['{new_rule_name}'] = macro_usage_counter.get('{new_rule_name}', 0) + 1",
        f"    current_integral = integral",
        f"    current_status = 'rewrite'",
        f""
    ]
    for rule in rule_combination_names:
        code_lines.extend([
            f"    # 执行子规则: {rule}",
            f"    res_{rule} = {rule}(current_integral)",
            f"    if res_{rule} is None:",
            f"        return None",
            f"    current_integral, current_status = res_{rule}",
            f"    if current_status == 'solved':",
            f"        return (current_integral, 'solved')",
            f""
        ])
    code_lines.append("    return (current_integral, current_status)\n")
    code_string = "\n".join(code_lines)
    return new_rule_name, code_string

def append_rule_to_source_file(file_path: str, code_string: str, new_rule_name: str) -> None:
    """
    将新规则函数追加到源文件，并自动更新 RULE_NAMES / RULE_DICT（如果存在），
    最后触发 build_action_space() 重建动作空间。
    具有备份恢复能力。
    """
    backup_path = file_path + ".bak"
    shutil.copyfile(file_path, backup_path)
    try:
        # 追加函数定义
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write("\n\n" + code_string)
        # 读取全文
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        # 在 RULE_NAMES 列表中添加新规则名
        if "RULE_NAMES" in content:
            pattern_names = r'(RULE_NAMES\s*=\s*\[)([^\]]*)(\])'
            def repl_names(m):
                before = m.group(1)
                middle = m.group(2)
                after = m.group(3)
                if f'"{new_rule_name}"' in middle:
                    return m.group(0)
                new_middle = middle.rstrip() + f',\n    "{new_rule_name}"'
                return before + new_middle + after
            content = re.sub(pattern_names, repl_names, content, flags=re.DOTALL)
        # 在 RULE_DICT 中添加映射
        if "RULE_DICT" in content:
            pattern_dict = r'(RULE_DICT\s*=\s*\{)([^\}]*)(\})'
            def repl_dict(m):
                before = m.group(1)
                middle = m.group(2)
                after = m.group(3)
                if f'"{new_rule_name}":' in middle:
                    return m.group(0)
                new_middle = middle.rstrip() + f',\n    "{new_rule_name}": {new_rule_name}'
                return before + new_middle + after
            content = re.sub(pattern_dict, repl_dict, content, flags=re.DOTALL)
        # 确保文件末尾有 build_action_space() 调用（用于注册表重建）
        if "build_action_space" not in content:
            content += "\n\n# 重建动作空间\nfrom knowledge.rule_registry import build_action_space\nbuild_action_space()\n"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    except Exception as e:
        # 发生错误时回滚
        if os.path.exists(backup_path):
            shutil.copyfile(backup_path, file_path)
        raise e
    finally:
        if os.path.exists(backup_path):
            os.remove(backup_path)

def verify_generated_code(new_rule_name: str, file_path: str = "knowledge/rules.py") -> bool:
    """
    编译和运行时验证新规则，失败则回滚文件。
    检查：
    - 语法正确性（AST）
    - 模块导入成功且新函数存在
    - 新规则已注册到 rule_registry
    """
    backup_path = file_path + ".bak"
    # 语法检查
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            current_content = f.read()
        ast.parse(current_content)
    except SyntaxError as se:
        print(f"❌ 语法错误: {se}，触发回滚")
        if os.path.exists(backup_path):
            shutil.copyfile(backup_path, file_path)
            os.remove(backup_path)
        return False
    # 动态重载模块
    try:
        module_name = "knowledge.rules"
        if module_name in sys.modules:
            imported_module = importlib.reload(sys.modules[module_name])
        else:
            imported_module = importlib.import_module(module_name)
        if not hasattr(imported_module, new_rule_name):
            raise AttributeError(f"函数 {new_rule_name} 未找到")
        # 检查是否在注册表中
        from knowledge.rule_registry import get_all_rule_names, build_action_space
        # 确保注册表已更新（文件末尾的 build_action_space 已被执行）
        build_action_space()
        if new_rule_name not in get_all_rule_names():
            raise ValueError(f"新规则 {new_rule_name} 未注册到 rule_registry")
    except Exception as e:
        print(f"❌ 重载验证失败: {e}，触发回滚")
        if os.path.exists(backup_path):
            shutil.copyfile(backup_path, file_path)
            os.remove(backup_path)
        return False
    # 清理备份
    if os.path.exists(backup_path):
        os.remove(backup_path)
    print(f"✅ 新规则 {new_rule_name} 验证通过，已激活")
    return True

def prune_inactive_macros(threshold: int = 50, file_path: str = "knowledge/rules.py") -> None:
    """
    淘汰使用次数低于 threshold 的宏规则（规则名以 rule_auto_macro_ 开头）。
    从源文件中删除相应的函数定义，并从注册表内存中移除。
    """
    global macro_usage_counter
    # 筛选不活跃的宏规则
    inactive = [name for name, cnt in macro_usage_counter.items()
                if cnt < threshold and name.startswith("rule_auto_macro_")]
    if not inactive:
        print("没有需要淘汰的宏规则")
        return
    print(f"淘汰以下宏规则（使用次数 < {threshold}）: {inactive}")
    # 备份原文件
    backup_path = file_path + ".bak"
    shutil.copyfile(file_path, backup_path)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        new_lines = []
        skip_until_next_def = False
        for line in lines:
            # 检测是否是要删除的函数定义开始
            if any(f"def {name}" in line for name in inactive):
                skip_until_next_def = True
                continue
            if skip_until_next_def:
                # 跳过函数体直到下一个非缩进行（即下一个函数或全局代码）
                if line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                    skip_until_next_def = False
                else:
                    continue
            new_lines.append(line)
        # 重新写入
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        # 同时从注册表的内存中移除
        from knowledge.rule_registry import _RULE_REGISTRY, _RULE_NAME_LIST, _RULE_ID_MAP, build_action_space
        for name in inactive:
            if name in _RULE_REGISTRY:
                del _RULE_REGISTRY[name]
            if name in _RULE_ID_MAP:
                del _RULE_ID_MAP[name]
            if name in _RULE_NAME_LIST:
                _RULE_NAME_LIST.remove(name)
            macro_usage_counter.pop(name, None)
        # 重建动作空间
        build_action_space()
        print("✅ 宏规则淘汰完成，动作空间已更新")
    except Exception as e:
        print(f"❌ 淘汰过程出错: {e}，回滚")
        if os.path.exists(backup_path):
            shutil.copyfile(backup_path, file_path)
    finally:
        if os.path.exists(backup_path):
            os.remove(backup_path)

# 测试入口（可选）
if __name__ == "__main__":
    # 单元测试示例
    os.makedirs("knowledge", exist_ok=True)
    mock_content = """from knowledge.rule_registry import register_rule
@register_rule()
def rule_power(integral): return (integral, "rewrite")
RULE_NAMES = ["rule_power"]
RULE_DICT = {"rule_power": rule_power}
"""
    test_file = "knowledge/rules.py"
    with open(test_file, "w") as f:
        f.write(mock_content)
    combo = ["rule_power", "rule_power"]
    name, code = generate_macro_rule_code(combo, 999)
    append_rule_to_source_file(test_file, code, name)
    ok = verify_generated_code(name, test_file)
    print(f"测试结果: {ok}")
    # 模拟使用计数
    macro_usage_counter[name] = 5
    prune_inactive_macros(threshold=10, file_path=test_file)
    # 清理
    if os.path.exists("knowledge"):
        shutil.rmtree("knowledge")