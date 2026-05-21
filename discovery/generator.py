import os
import re
import ast
import sys
import shutil
import importlib


def generate_macro_rule_code(rule_combination_names, new_rule_id):
    """
    1. 动态宏规则代码生成器
    将一组旧规则名称拼装组合，生成符合规则库规范的、顺序流水线执行的 Python 函数代码字符串。

    :param rule_combination_names: 规则名称列表, 例如 ["TrigProductToSum", "ExtractConstant"]
    :param new_rule_id: 新规则的唯一标识 ID
    :return: (new_rule_name, code_string) 新规则名及完整的函数体字符串
    """
    new_rule_name = f"rule_auto_macro_{new_rule_id}"

    # 构建动态函数的骨架
    code_lines = [
        f"def {new_rule_name}(integral):",
        f"    \"\"\"",
        f"    Automatically generated macro rule (ID: {new_rule_id}).",
        f"    Pipeline: {' -> '.join(rule_combination_names)}",
        f"    \"\"\"",
        f"    current_integral = integral",
        f"    current_status = 'rewrite'",
        f""
    ]

    # 串联执行流逻辑
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

    # 最终返回流水线处理完的表达式及状态
    code_lines.append("    return (current_integral, current_status)\n")

    code_string = "\n".join(code_lines)
    return new_rule_name, code_string


def append_rule_to_source_file(file_path, code_string, new_rule_name):
    """
    2. 源码黑客改写器
    将新生成的宏动作函数追加到文件末尾，并强行改写 RULE_NAMES 列表与 RULE_DICT 字典注册表。

    :param file_path: 目标规则源码文件路径 (如 'knowledge/rules.py')
    :param code_string: 步骤 1 生成的函数体字符串
    :param new_rule_name: 新规则的方法名字符串
    """
    # 建立安全备份，供第三步的验证失败回滚使用
    backup_path = file_path + ".bak"
    shutil.copyfile(file_path, backup_path)

    try:
        # 第一步：以追加模式将完整的函数体“烫进”文件末尾
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write("\n\n" + code_string)

        # 第二步：读取全文，利用正则表达式精准定位并重写注册表
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 2.1 强行插入 RULE_NAMES 列表（在匹配到 'RULE_NAMES = [' 后换行插入新动作）
        if "RULE_NAMES" in content:
            content = re.sub(
                r'(RULE_NAMES\s*=\s*\[)',
                r'\1\n    "' + new_rule_name + '",',
                content
            )
        else:
            raise ValueError("在源码中未找到 RULE_NAMES 注册列表，无法扩容动作空间。")

        # 2.2 强行插入 RULE_DICT 字典（在匹配到 'RULE_DICT = {' 后换行插入键值对）
        if "RULE_DICT" in content:
            content = re.sub(
                r'(RULE_DICT\s*=\s*\{)',
                r'\1\n    "' + new_rule_name + '": ' + new_rule_name + ',',
                content
            )
        else:
            raise ValueError("在源码中未找到 RULE_DICT 注册字典，无法映射动作句柄。")

        # 将重写后的全新内容覆盖写入文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

    except Exception as e:
        # 如果改写过程本身报错，立即从临时备份恢复
        if os.path.exists(backup_path):
            shutil.copyfile(backup_path, file_path)
        raise e


def verify_generated_code(new_rule_name, file_path="knowledge/rules.py"):
    """
    3. 编译安全性防御（语法检查与回滚机制）
    利用 AST 静态解析与动态重载，确保新合成的代码未破坏系统稳定性。若失败则全自动无感回滚。

    :param new_rule_name: 刚刚注册的新规则方法名
    :param file_path: 目标规则源码文件路径
    :return: bool 验证通过返回 True，失败回滚返回 False
    """
    backup_path = file_path + ".bak"

    # 核心防御 1：AST 静态抽象语法树解析，严防少括号或缩进崩溃
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            current_content = f.read()
        ast.parse(current_content)
    except SyntaxError as se:
        print(f"❌【安全防御】检测到语法编译错误: {se}。正在触发自动回滚...")
        if os.path.exists(backup_path):
            shutil.copyfile(backup_path, file_path)
            os.remove(backup_path)
        return False

    # 核心防御 2：动态重载模块，确保双塔网络等下游组件能顺利 import 并读取到扩容后的 len(RULE_NAMES)
    try:
        # 获取模块在 sys.modules 中的键名
        module_name = "knowledge.rules"

        if module_name in sys.modules:
            # 强行刷新并重载内存中的模块
            imported_module = importlib.reload(sys.modules[module_name])
        else:
            imported_module = importlib.import_module(module_name)

        # 检查函数句柄是否存在
        if not hasattr(imported_module, new_rule_name):
            raise AttributeError(f"模块中未检测到函数句柄: {new_rule_name}")

        # 检查注册表内是否确实存在
        if new_rule_name not in getattr(imported_module, "RULE_NAMES", []):
            raise ValueError(f"RULE_NAMES 中未成功包含新规则: {new_rule_name}")
        if new_rule_name not in getattr(imported_module, "RULE_DICT", {}):
            raise ValueError(f"RULE_DICT 中未成功包含映射: {new_rule_name}")

    except Exception as e:
        print(f"❌【安全防御】模块重载验证失败: {e}。正在触发自动回滚...")
        if os.path.exists(backup_path):
            shutil.copyfile(backup_path, file_path)
            os.remove(backup_path)
        return False

    # 验证全部通过，安全移除备份
    if os.path.exists(backup_path):
        os.remove(backup_path)

    print(f"🚀【安全防御】新规则 [{new_rule_name}] 静态编译与动态重载全部通过！动作空间已成功扩容。")
    return True


# ==========================================
# 模块联动与闭环测试演示 (Pipeline Workflow)
# ==========================================
if __name__ == "__main__":
    # 为了演示，我们在本地临时创建一个模拟的 knowledge/rules.py 文件
    os.makedirs("knowledge", exist_ok=True)
    mock_rules_content = """# 基础原子规则定义
def TrigProductToSum(integral):
    return (integral, "rewrite")

def ExtractConstant(integral):
    return (integral, "rewrite")

# 核心注册表（双塔网络读取的源头）
RULE_NAMES = [
    "TrigProductToSum",
    "ExtractConstant",
]

RULE_DICT = {
    "TrigProductToSum": TrigProductToSum,
    "ExtractConstant": ExtractConstant,
}
"""
    test_file = "knowledge/rules.py"
    with open(test_file, "w", encoding="utf-8") as f:
        f.write(mock_rules_content)

    print("--- 闭环流程开始 ---")

    # 模拟从 pattern_miner.py 传过来的黄金套路名字和全局最新规则递增 ID
    combination = ["TrigProductToSum", "ExtractConstant"]
    next_id = 101

    # Step 1: 生成流水线代码
    r_name, r_code = generate_macro_rule_code(combination, next_id)
    print(f"1. 成功生成宏函数代码，期望方法名: {r_name}")

    # Step 2: 改写底层源码并注入注册表
    append_rule_to_source_file(test_file, r_code, r_name)
    print("2. 源码及注册表改写注入完成。")

    # Step 3: 进行编译与重载安全性校验
    # 将当前的 'knowledge' 目录加进环境路径确保测试时能 import
    sys.path.append(os.getcwd())
    success = verify_generated_code(r_name, file_path=test_file)
    print(f"3. 闭环最终状态: {'成功缝合补丁！' if success else '回滚成功，主程序安全。'}")

    # 打印看一眼被改写后的文件变成什么样了
    with open(test_file, "r", encoding="utf-8") as f:
        print("\n--- 注入改写后的 rules.py 实际效果展示 ---")
        print(f.read())

    # 清理测试产生的临时文件目录
    if os.path.exists("knowledge"):
        shutil.rmtree("knowledge")