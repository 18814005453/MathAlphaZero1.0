# discovery/generator.py
import os
import re
import ast
import sys
import shutil
import importlib
from collections import defaultdict

# 全局宏动作使用计数器（用于衰减）
macro_usage_counter = defaultdict(int)

def generate_macro_rule_code(rule_combination_names, new_rule_id):
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

def append_rule_to_source_file(file_path, code_string, new_rule_name):
    """
    将新规则函数追加到源文件，并更新 RULE_NAMES 和 RULE_DICT。
    注意：假设目标文件已经使用了 rule_registry 装饰器，但为了兼容旧代码，
    我们仍会修改 RULE_NAMES 和 RULE_DICT（如果存在），并自动调用 build_action_space。
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
        # 如果文件末尾有 build_action_space() 调用，我们需要在添加规则后重新调用
        # 更简单：在 RULE_NAMES 列表和 RULE_DICT 中插入新规则（用于旧代码兼容）
        if "RULE_NAMES" in content:
            # 在 RULE_NAMES = [ ... ] 中添加新名称
            pattern_names = r'(RULE_NAMES\s*=\s*\[)([^\]]*)(\])'
            def repl_names(m):
                before = m.group(1)
                middle = m.group(2)
                after = m.group(3)
                # 如果已经有该名称，跳过
                if f'"{new_rule_name}"' in middle:
                    return m.group(0)
                new_middle = middle.rstrip() + f',\n    "{new_rule_name}"'
                return before + new_middle + after
            content = re.sub(pattern_names, repl_names, content, flags=re.DOTALL)
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
        # 确保文件末尾有 build_action_space() 调用
        if "build_action_space" not in content:
            content += "\n\n# 重建动作空间\nfrom knowledge.rule_registry import build_action_space\nbuild_action_space()\n"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    except Exception as e:
        if os.path.exists(backup_path):
            shutil.copyfile(backup_path, file_path)
        raise e
    finally:
        if os.path.exists(backup_path):
            os.remove(backup_path)

def verify_generated_code(new_rule_name, file_path="knowledge/rules.py"):
    """
    编译和运行时验证新规则，失败则回滚。
    """
    backup_path = file_path + ".bak"
    # AST 语法检查
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
        from knowledge.rule_registry import get_all_rule_names
        if new_rule_name not in get_all_rule_names():
            raise ValueError(f"新规则 {new_rule_name} 未注册到 rule_registry")
    except Exception as e:
        print(f"❌ 重载验证失败: {e}，触发回滚")
        if os.path.exists(backup_path):
            shutil.copyfile(backup_path, file_path)
            os.remove(backup_path)
        return False
    if os.path.exists(backup_path):
        os.remove(backup_path)
    print(f"✅ 新规则 {new_rule_name} 验证通过，已激活")
    return True

def prune_inactive_macros(threshold=50, rule_names_list=None, file_path="knowledge/rules.py"):
    """
    删除连续 threshold 轮未被使用的宏规则。
    rule_names_list: 当前所有规则名称列表（用于过滤出宏规则）。
    此函数会从源文件中移除对应的函数定义和注册条目。
    """
    global macro_usage_counter
    inactive = [name for name, cnt in macro_usage_counter.items() if cnt < threshold and name.startswith("rule_auto_macro_")]
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
                # 跳过直到下一个函数定义（非缩进行）
                if line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                    skip_until_next_def = False
                else:
                    continue
            new_lines.append(line)
        # 重新写入
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        # 同时从注册表的内存中移除（下次重载时自动生效）
        from knowledge.rule_registry import _RULE_REGISTRY, _RULE_NAME_LIST, _RULE_ID_MAP
        for name in inactive:
            if name in _RULE_REGISTRY:
                del _RULE_REGISTRY[name]
            if name in _RULE_ID_MAP:
                del _RULE_ID_MAP[name]
            if name in _RULE_NAME_LIST:
                _RULE_NAME_LIST.remove(name)
        # 重置计数器
        for name in inactive:
            macro_usage_counter.pop(name, None)
        # 重新构建动作空间
        from knowledge.rule_registry import build_action_space
        build_action_space()
        print("✅ 宏规则淘汰完成，动作空间已更新")
    except Exception as e:
        print(f"❌ 淘汰过程出错: {e}，回滚")
        if os.path.exists(backup_path):
            shutil.copyfile(backup_path, file_path)
    finally:
        if os.path.exists(backup_path):
            os.remove(backup_path)

# 测试入口
if __name__ == "__main__":
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
    # 清理
    if os.path.exists("knowledge"):
        shutil.rmtree("knowledge")