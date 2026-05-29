#!/usr/bin/env python3
"""
MathAlphaZero 统一训练脚本 v7.0
- 错题本集成 (ErrorBook)
- 错题优先重练 (50%训练时间做旧错题)
- 失败原因诊断
- 断点续训 + 课程学习 + 优先经验回放 + 性能追踪
"""
import os, sys, time, json, math, random, pickle, argparse, logging
from collections import defaultdict, deque

import numpy as np
import torch, torch.nn as nn, torch.optim as optim
import sympy as sp

import knowledge.rules
from core.state import IntegrationState, set_default_preprocessor
from core.env import IntegrationEnv
from core.network import MathNet
from core.engine import MCTS
from utils.preprocessor import MathPreprocessor
from utils.validator import MathValidator
from knowledge.rule_registry import build_action_space, get_all_rule_names, get_num_rules, reload_module
from performance_tracker import PerformanceTracker
from error_book import ErrorBook

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
L = logging.getLogger("train")


# ===================== 题目生成器 =====================
def _rc():
    if random.random() < 0.7:
        return random.randint(1, 5) * random.choice([-1, 1])
    return sp.Rational(random.randint(1, 3), random.randint(2, 4)) * random.choice([-1, 1])


def generate_problem(diff="easy"):
    x = sp.Symbol('x')
    if diff == "easy":
        base = random.choice([x**n for n in range(1, 5)] +
                             [sp.sin(k*x) for k in range(1, 3)] +
                             [sp.cos(k*x) for k in range(1, 3)] +
                             [sp.exp(k*x) for k in range(1, 3)] + [sp.exp(-x)])
        return sp.diff(_rc() * base, x)
    elif diff == "medium":
        base = random.choice([x*sp.sin(x), x*sp.cos(x), x*sp.exp(x), x**2*sp.exp(x),
                              x*sp.sin(2*x), sp.sin(x)**2, sp.cos(x)**2,
                              x/(x**2+1), sp.log(x+1), x*sp.log(x)])
        return sp.simplify(sp.diff(_rc() * base, x))
    else:
        base = random.choice([1/(x**2+1), x/(x**2+1), sp.atan(x), x*sp.atan(x),
                              sp.exp(x)*sp.sin(x), sp.exp(x)*sp.cos(x),
                              x**2*sp.sin(x), sp.sin(x)*sp.cos(x)])
        return sp.simplify(sp.diff(_rc() * base, x))


def gen_with_depth(max_d, attempts=30):
    def _d(e): return 1 if e.is_Atom else 1 + max(_d(a) for a in e.args)
    for _ in range(attempts):
        d = random.choices(["easy","medium","hard"], weights=[0.2,0.4,0.4])[0]
        e = generate_problem(d)
        if _d(e) <= max_d: return e
    return generate_problem("easy")


# ===================== Buffer =====================
class PERBuffer:
    def __init__(self, cap=100000, alpha=0.6, beta=0.4, beta_inc=0.001):
        self.b = []; self.p = []; self.pos = 0
        self.cap = cap; self.alpha = alpha; self.beta = beta; self.beta_inc = beta_inc

    def push(self, item, prio=1.0):
        prio = max(prio, 1e-6)
        if len(self.b) < self.cap: self.b.append(item); self.p.append(prio)
        else: self.b[self.pos] = item; self.p[self.pos] = prio
        self.pos = (self.pos + 1) % self.cap

    def sample(self, n):
        if not self.b: return [], [], []
        probs = np.array(self.p) ** self.alpha; probs /= probs.sum()
        idxs = np.random.choice(len(self.b), min(n, len(self.b)), p=probs)
        samples = [self.b[i] for i in idxs]
        total = len(self.b)
        w = (total * probs[idxs]) ** (-self.beta); w /= w.max()
        self.beta = min(1.0, self.beta + self.beta_inc)
        return samples, w, idxs

    def update_priorities(self, idxs, prios):
        for i, p in zip(idxs, prios): self.p[i] = max(p, 1e-6)

    def __len__(self): return len(self.b)
    def save(self, path):
        with open(path,'wb') as f: pickle.dump((self.b, self.p, self.pos), f)
    def load(self, path):
        if os.path.exists(path) and os.path.getsize(path)>0:
            with open(path,'rb') as f: self.b, self.p, self.pos = pickle.load(f)


# ===================== Curriculum =====================
class Curriculum:
    def __init__(self, start=2, max_d=20, w=50):
        self.p = defaultdict(lambda: deque(maxlen=w))
        self.d = start; self.max = max_d

    def update(self, d, ok):
        self.p[d].append(1.0 if ok else 0.0)

    def step(self):
        dq = self.p[self.d]
        if len(dq) >= 10 and np.mean(dq) >= 0.6: self.d = min(self.d+1, self.max)
        elif len(dq) >= 20 and np.mean(dq) < 0.25: self.d = max(2, self.d-1)
        return self.d


# ===================== 检测失败原因 =====================
def diagnose_failure(expr, trajectory, reward, done, net, preprocessor, N):
    """分析失败原因"""
    if not trajectory:
        return "no_trajectory", []
    if not done:
        return "search_incomplete", []
    if reward <= 0:
        # Check if any rule matched at all
        last_step = trajectory[-1]
        env = IntegrationEnv(max_steps=30)
        legal = env.legal_actions(last_step["state"])
        if not legal:
            return "no_legal_action", []
        # Check if network gave bad priorities
        return "reward_failed", [a.name for a in legal[:3]]
    return "unknown", []


# ===================== 主训练 =====================
def pad_tensor(t, target_len):
    if t.shape[0] < target_len:
        return torch.cat([t, torch.zeros(target_len - t.shape[0], dtype=t.dtype)])
    return t[:target_len]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--problems_per_epoch", type=int, default=40)
    p.add_argument("--simulations", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--gamma", type=float, default=0.96)
    p.add_argument("--max_depth", type=int, default=30)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--checkpoint", type=str, default="data/checkpoint.pth")
    p.add_argument("--memory_path", type=str, default="data/memory.pkl")
    p.add_argument("--error_path", type=str, default="data/error_book.json")
    p.add_argument("--eval_every", type=int, default=20)
    p.add_argument("--error_review_ratio", type=float, default=0.4)
    p.add_argument("--reset", action="store_true")
    args = p.parse_args()

    L.info("=" * 60)
    L.info("MathAlphaZero v7.0 — Error Book Training")
    L.info(f"Config: {vars(args)}")
    L.info("=" * 60)

    os.makedirs("data", exist_ok=True)
    build_action_space()
    pre = MathPreprocessor(max_len=128)
    set_default_preprocessor(pre)
    validator = MathValidator()
    NUM_RULES = get_num_rules()
    MAX_LEN = 128
    L.info(f"Rules: {NUM_RULES}")

    device = torch.device("cpu")
    L.info(f"Device: {device}")

    net = MathNet(vocab_size=pre.vocab_size, d_model=128, nhead=8, num_layers=3,
                  rule_num_layers=3, max_len=MAX_LEN, dropout=0.1, temperature=0.07,
                  learn_temperature=False, use_depth_embedding=True, max_depth=32).to(device)
    net.refresh_rule_cache(get_all_rule_names(), pre._string_to_ids, action_ids=list(range(NUM_RULES)))
    L.info(f"Net: {sum(pm.numel() for pm in net.parameters()):,} params")

    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=40, min_lr=1e-6)

    memory = PERBuffer(cap=100000)
    error_book = ErrorBook(save_path=args.error_path, max_errors=500)
    tracker = PerformanceTracker(save_dir="data/performance")

    # Resume
    start_epoch = 1
    curriculum = Curriculum(start=2, max_d=15)
    solved_set = set()

    if not args.reset and os.path.exists(args.checkpoint):
        try:
            ckpt = torch.load(args.checkpoint, map_location=device)
            ms = net.state_dict()
            filt = {k: v for k, v in ckpt["model"].items() if k in ms and v.shape == ms[k].shape}
            net.load_state_dict(filt, strict=False)
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = ckpt.get("epoch", 1) + 1
            curriculum.depth = ckpt.get("curriculum_depth", 2)
            memory.load(args.memory_path)
            L.info(f"Resumed from epoch {start_epoch}")
        except Exception as e:
            L.warning(f"Failed to load checkpoint: {e}")

    x = sp.Symbol('x')
    base_problems = [
        sp.Integral(x**2, x), sp.Integral(x**3, x), sp.Integral(sp.sin(x), x),
        sp.Integral(sp.cos(x), x), sp.Integral(sp.exp(x), x), sp.Integral(1/x, x),
        sp.Integral(2*x, x), sp.Integral(3*x**2, x), sp.Integral(sp.sin(2*x), x),
        sp.Integral(sp.cos(3*x), x), sp.Integral(sp.exp(2*x), x),
        sp.Integral(sp.sin(x)**2, x), sp.Integral(sp.cos(x)**2, x),
        sp.Integral(1/(x**2+1), x), sp.Integral(x/(x**2+1), x),
        sp.Integral(sp.log(x), x), sp.Integral(sp.exp(x)*sp.sin(x), x),
        sp.Integral(x*sp.sin(x), x), sp.Integral(x*sp.exp(x), x),
        sp.Integral(sp.sin(x)*sp.cos(x), x),
    ]

    BEST_SR = 0
    pretrain_epochs = 30
    t_start = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        current_depth = curriculum.step()
        L.info(f"--- Epoch {epoch}/{args.epochs} | Depth: {current_depth} | "
               f"Errors: {error_book.stats()['total']} | Mem: {len(memory)} ---")

        # ===== 生成训练题（含错题重练）=====
        problems = []

        # 错题重练：40% 的训练时间做旧错题
        error_review_cnt = int(args.problems_per_epoch * args.error_review_ratio)
        if error_book.errors:
            review = error_book.sample_review_set(n=error_review_cnt, bias_recent=True)
            problems.extend([sp.Integral(sp.sympify(e.expr_str), x) for e in review])

        # 新题 + 基础题
        new_cnt = args.problems_per_epoch - len(problems)
        for expr in base_problems:
            if len(problems) >= args.problems_per_epoch: break
            if expr not in problems:
                problems.append(expr)
        for _ in range(new_cnt):
            if len(problems) >= args.problems_per_epoch: break
            expr = gen_with_depth(current_depth)
            if str(expr) not in solved_set:
                problems.append(sp.Integral(expr, x))

        if not problems: continue

        sc, r_losses, l_losses, v_losses = 0, [], [], []

        for expr in problems:
            state = IntegrationState(expr)
            mcts = MCTS(net, pre, num_simulations=args.simulations, gamma=args.gamma,
                        max_depth=args.max_depth, timeout=args.timeout,
                        tree_depth_limit=6, device=str(device))
            traj = mcts.get_trajectory(state, temperature=args.temperature)

            if not traj:
                err_str = str(expr.function)
                error_book.record_fail(err_str, difficulty="easy",
                                       reason="no_trajectory", epoch=epoch)
                curriculum.update(current_depth, False)
                continue

            last = traj[-1]
            env = IntegrationEnv(max_steps=args.max_depth)
            ns, reward, done, _ = env.step(last["state"], last["action"])
            ok = (done and reward > 0.8 and validator.verify_integral(expr.function, ns.expr))

            if ok:
                sc += 1
                solved_set.add(str(expr))
                curriculum.update(current_depth, True)

                # 如果之前是错题，标记为通过
                err_str = str(expr.function)
                if error_book.is_error(err_str):
                    error_book.record_pass(err_str, epoch=epoch)

                # Store experience
                T = len(traj)
                vals = [step.get("value_target", 0.0) for step in traj]
                returns = []
                for t_i in range(T):
                    g = 0.0
                    for k in range(min(5, T - t_i)):
                        g += (args.gamma ** k) * vals[t_i + k]
                    if t_i + 5 < T: g += (args.gamma ** 5) * vals[t_i + 5]
                    returns.append(g)

                for idx, step in enumerate(traj):
                    tok, dep = pre.state_to_tensor_with_depth(step["state"].expr)
                    dep = pad_tensor(dep.squeeze(0), MAX_LEN).unsqueeze(0)
                    rt = pad_tensor(torch.tensor(step["rule_policy_target"], dtype=torch.float32), NUM_RULES)
                    lt = pad_tensor(torch.tensor(step["location_policy_target"], dtype=torch.float32), MAX_LEN)
                    val_tgt = returns[idx] if idx < len(returns) else reward
                    td_err = abs(step.get("value_target", 0.0) - val_tgt)
                    memory.push((tok, dep, rt, lt, torch.tensor([val_tgt])),
                                priority=td_err**2 + 1e-6)
            else:
                curriculum.update(current_depth, False)

                # 记录错题
                err_str = str(expr.function)
                reason, blocked_rules = diagnose_failure(expr, traj, reward, done, net, pre, NUM_RULES)

                # 尝试得到标准答案
                try:
                    target = sp.integrate(expr.function, x)
                    target_str = str(target)
                except Exception:
                    target_str = ""

                complexity = expr.function.count_ops() if hasattr(expr.function, 'count_ops') else 0
                error_book.record_fail(
                    err_str, target_answer=target_str,
                    difficulty="easy" if complexity < 5 else "medium" if complexity < 10 else "hard",
                    reason=reason, complexity=complexity,
                    rules_blocked=blocked_rules, epoch=epoch
                )

        # ===== 网络训练 =====
        if len(memory) >= args.batch_size:
            for _ in range(8):
                batch, weights, indices = memory.sample(args.batch_size)
                if not batch: continue

                sb = torch.cat([b[0] for b in batch], dim=0).to(device)
                db = torch.cat([b[1] for b in batch], dim=0).to(device)
                rt = torch.stack([b[2] for b in batch]).to(device)
                lt = torch.stack([b[3] for b in batch]).to(device)
                vt = torch.stack([b[4] for b in batch]).to(device)
                wt = torch.tensor(weights, dtype=torch.float32).to(device)

                rm = torch.ones(len(batch), NUM_RULES, dtype=torch.bool, device=device)
                lm = (sb != 0)

                rl, ll, pv = net(sb, depth=db, rule_mask=rm, location_mask=lm)
                loss_r = -(rt * nn.LogSoftmax(dim=1)(rl)).sum(dim=1) * wt
                loss_r = loss_r.mean()

                if epoch <= pretrain_epochs:
                    loss_l = torch.tensor(0.0, device=device)
                else:
                    log_l = nn.LogSoftmax(dim=1)(ll)
                    loss_l = -(lt * log_l * lm.float()).sum(dim=1) * wt
                    loss_l = loss_l.mean()

                loss_v = (pv - vt).pow(2).squeeze() * wt
                loss_v = loss_v.mean()

                lw = 0.3 if epoch <= pretrain_epochs else 0.5
                loss = loss_r + lw * loss_l + loss_v

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()

                r_losses.append(loss_r.item()); l_losses.append(loss_l.item()); v_losses.append(loss_v.item())

                with torch.no_grad():
                    td = (pv - vt).abs().squeeze().cpu().numpy()
                memory.update_priorities(indices, td**2 + 1e-6)

            avg_r = np.mean(r_losses); avg_l = np.mean(l_losses); avg_v = np.mean(v_losses)
            scheduler.step(avg_r + avg_l + avg_v)
            L.info(f"Loss | R:{avg_r:.4f} L:{avg_l:.4f} V:{avg_v:.4f}")

        sr = sc / max(1, len(problems)) * 100
        marker = " *** NEW BEST ***" if sr > BEST_SR else ""
        if sr > BEST_SR: BEST_SR = sr
        elapsed = time.time() - t_start
        L.info(f"Epoch {epoch} done | Success: {sr:.1f}% | Solved: {len(solved_set)} | "
               f"Errors: {error_book.stats()['total']} | Time: {elapsed:.0f}s{marker}")

        # 记录性能
        tracker.record_training(epoch,
            np.mean(r_losses) if r_losses else 0,
            np.mean(l_losses) if l_losses else 0,
            np.mean(v_losses) if v_losses else 0,
            sr, get_num_rules())

        # 保存
        ckpt = {"model": net.state_dict(), "optimizer": optimizer.state_dict(),
                "epoch": epoch, "curriculum_depth": curriculum.depth}
        torch.save(ckpt, args.checkpoint)
        memory.save(args.memory_path)

        # 定期打印错题报告
        if epoch % 20 == 0 and error_book.errors:
            error_book.print_report()

        if epoch % 50 == 0:
            L.info(tracker.show_improvement_summary())

    torch.save(net.state_dict(), "data/brain.pth")
    L.info("Training complete!")
    error_book.print_report()


if __name__ == "__main__":
    main()
