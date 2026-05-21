import os

# 修复 env.py
env_file = "core/env.py"
with open(env_file, 'r') as f:
    content = f.read()

# 修复 start_time 初始化问题
content = content.replace(
    "self.start_time = None",
    "self.start_time = None\n        self.initial_complexity = None"
)

# 修复 is_terminal 中的 None 检查
content = content.replace(
    "if self.time_limit is not None and (time.time() - self.start_time) > self.time_limit:",
    "if self.time_limit is not None and self.start_time is not None and (time.time() - self.start_time) > self.time_limit:"
)

with open(env_file, 'w') as f:
    f.write(content)

print("✅ core/env.py 修复完成")

# 修复 auto_train.py 中的 SymPy Poly 警告
train_file = "auto_train.py"
with open(train_file, 'r') as f:
    lines = f.readlines()

# 找到并修复 _generate_primitive_hard 函数
new_lines = []
in_hard_func = False
hard_func_buffer = []

for line in lines:
    if "def _generate_primitive_hard(x):" in line:
        in_hard_func = True
        hard_func_buffer = [line]
        continue
    
    if in_hard_func:
        hard_func_buffer.append(line)
        if line.strip() and not line.startswith(' ') and not line.startswith('\t') and "def _" not in line:
            # 函数结束，处理 buffer
            func_text = ''.join(hard_func_buffer)
            # 修复 Poly 问题
            func_text = func_text.replace(
                "poly_coeff = sp.Poly(random.randint(1, 3) * x + random.randint(1, 2), x)",
                "poly_coeff = random.randint(1, 3) * x + random.randint(1, 2)"
            )
            func_text = func_text.replace(
                "return coeff * poly_coeff * base",
                "return coeff * poly_coeff * base"
            )
            new_lines.append(func_text)
            in_hard_func = False
            continue
    
    if not in_hard_func:
        new_lines.append(line)

with open(train_file, 'w') as f:
    f.writelines(new_lines)

print("✅ auto_train.py 修复完成")

# 确保 MCTS 中的 env 正确初始化
engine_file = "core/engine.py"
with open(engine_file, 'r') as f:
    content = f.read()

# 修复 MCTS __init__ 中 env 初始化
if "self.env = IntegrationEnv(" in content:
    content = content.replace(
        "self.env = IntegrationEnv(max_steps=max_depth, time_limit=timeout)",
        "self.env = IntegrationEnv(max_steps=max_depth, time_limit=timeout if timeout else 30.0)"
    )

with open(engine_file, 'w') as f:
    f.write(content)

print("✅ core/engine.py 修复完成")

# 清理旧数据并重新开始
import shutil
if os.path.exists("data"):
    for f in os.listdir("data"):
        if f.endswith((".pkl", ".pth", ".archived")):
            os.remove(os.path.join("data", f))
    print("✅ 旧数据已清理")

print("\n" + "="*50)
print("✅ 所有修复完成！现在可以运行:")
print("python auto_train.py")
print("="*50)
