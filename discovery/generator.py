# discovery/generator.py
"""Automatic macro-rule discovery — AST-based code generation (v5.0)"""

import ast
import os
import sys
import shutil
import importlib
from collections import defaultdict
from typing import List, Tuple, Optional

macro_usage_counter = defaultdict(int)


def generate_macro_rule_code(rule_names: List[str], new_rule_id: int) -> Tuple[str, str]:
    """Generate a composite rule function from a chain of sub-rules."""
    new_rule_name = f"rule_auto_macro_{new_rule_id}"

    lines = [
        f"def {new_rule_name}(integral):",
        f'    """Auto-generated macro rule (ID: {new_rule_id})."""',
        f"    from discovery.generator import macro_usage_counter",
        f"    macro_usage_counter['{new_rule_name}'] = macro_usage_counter.get('{new_rule_name}', 0) + 1",
        f"    current = integral",
        f"    label = 'rewrite'",
        f"",
    ]
    for rule in rule_names:
        lines.extend([
            f"    res = {rule}(current)",
            f"    if res is None:",
            f"        return None",
            f"    current, label = res",
            f"    if label == 'solved':",
            f"        return (current, 'solved')",
            f"",
        ])
    lines.append("    return (current, label)\n")
    return new_rule_name, "\n".join(lines)


def append_rule_to_rules_file(file_path: str, code_string: str, new_rule_name: str) -> None:
    """Insert new rule function before the build_action_space() call using AST."""
    backup_path = file_path + ".bak"
    shutil.copyfile(file_path, backup_path)

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Parse as AST
        tree = ast.parse(content)

        # Parse the new function as AST
        new_func_ast = ast.parse(code_string).body[0]

        # Find the position of the last import or assignment before build_action_space()
        lines = content.split('\n')
        insert_lineno = len(lines)

        # Find build_action_space() call position
        for node in ast.walk(tree):
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                func = node.value.func
                if isinstance(func, ast.Name) and func.id == 'build_action_space':
                    insert_lineno = node.lineno - 1
                    break

        # Reconstruct file with new rule inserted
        result_lines = lines[:insert_lineno]
        # Add blank line before new rule
        if result_lines and result_lines[-1].strip():
            result_lines.append('')
        result_lines.append(code_string.strip())
        result_lines.append('')
        result_lines.extend(lines[insert_lineno:])

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(result_lines))

    except Exception as e:
        if os.path.exists(backup_path):
            shutil.copyfile(backup_path, file_path)
        raise e
    finally:
        if os.path.exists(backup_path):
            os.remove(backup_path)


def verify_generated_code(new_rule_name: str, file_path: str = "knowledge/rules.py") -> bool:
    """Verify new rule: syntax check, import, and registry registration."""
    backup_path = file_path + ".bak"

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        ast.parse(content)
    except SyntaxError as se:
        print(f"Syntax error: {se}, rolling back")
        if os.path.exists(backup_path):
            shutil.copyfile(backup_path, file_path)
            os.remove(backup_path)
        return False

    try:
        module_name = "knowledge.rules"
        if module_name in sys.modules:
            importlib.reload(sys.modules[module_name])
        else:
            importlib.import_module(module_name)

        from knowledge.rule_registry import get_all_rule_names, build_action_space
        build_action_space()

        if new_rule_name not in get_all_rule_names():
            raise ValueError(f"Rule {new_rule_name} not in registry")
    except Exception as e:
        print(f"Verification failed: {e}, rolling back")
        if os.path.exists(backup_path):
            shutil.copyfile(backup_path, file_path)
            os.remove(backup_path)
        return False

    if os.path.exists(backup_path):
        os.remove(backup_path)
    print(f"New rule {new_rule_name} verified and activated")
    return True


def prune_inactive_macros(threshold: int = 50, file_path: str = "knowledge/rules.py") -> None:
    """Remove inactive macro rules using AST-based approach."""
    global macro_usage_counter
    inactive = [name for name, cnt in macro_usage_counter.items()
                if cnt < threshold and name.startswith("rule_auto_macro_")]
    if not inactive:
        print("No inactive macros to prune")
        return

    print(f"Pruning {len(inactive)} inactive macros: {inactive}")
    backup_path = file_path + ".bak"
    shutil.copyfile(file_path, backup_path)

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        tree = ast.parse(content)
        to_remove = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in inactive:
                to_remove.add(node.lineno)
                # Mark all lines from function start to end for removal
                for i in range(node.lineno, node.end_lineno + 1):
                    to_remove.add(i)

        lines = content.split('\n')
        new_lines = []
        for i, line in enumerate(lines, start=1):
            if i not in to_remove:
                new_lines.append(line)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(new_lines))

        # Clean registry in memory
        from knowledge.rule_registry import _RULE_REGISTRY, build_action_space
        for name in inactive:
            _RULE_REGISTRY.pop(name, None)
            macro_usage_counter.pop(name, None)

        # Reload and rebuild
        module_name = "knowledge.rules"
        if module_name in sys.modules:
            importlib.reload(sys.modules[module_name])
        else:
            importlib.import_module(module_name)
        build_action_space()

        print("Macro pruning complete")
    except Exception as e:
        print(f"Pruning failed: {e}, rolling back")
        if os.path.exists(backup_path):
            shutil.copyfile(backup_path, file_path)
    finally:
        if os.path.exists(backup_path):
            os.remove(backup_path)
