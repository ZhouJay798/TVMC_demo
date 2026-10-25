#!/usr/bin/env python
"""One command runner. Exit 0: selected scope passes; 2: measured target fails;
3: execution blocked/error. All three retain metrics/report/full traceback log.
"""
import argparse
import contextlib
import importlib.metadata
import json
import platform
import os
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


class Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log
    def write(self, text):
        self.console.write(text)
        self.log.write(text)
        self.log.flush()
        return len(text)
    def flush(self):
        self.console.flush()
        self.log.flush()
    def isatty(self):
        return False


def report(metrics, output):
    core, model = metrics.get('core', {}), metrics.get('model', {})
    lines = ['# TVMC Demo 实测报告', '', f"- 运行 ID：`{metrics['run_id']}`",
             f"- UTC 开始时间：{metrics['started_utc']}", f"- 总状态：**{metrics['status']}**",
             f"- 模型阶段：**{model.get('status', 'not_run')}**", '',
             '## 设计与范围', '',
             'Shamir (2,5) 使用一次多项式；Feldman 承诺和绑定会话的 AND-Schnorr Fiat–Shamir NIZK；'
             'HKDF-SHA256 派生 AES-256-GCM 密钥。模型阶段加密、恢复并加载完整 SmolLM2 safetensors 权重。', '',
             'Tardos 使用截断 arcsine 分布、对称累加得分和运行前固定的 Bernstein 阈值。'
             '5 个用户对应 5 个发布会话，不能据此宣称某个托管方就是泄露者。'
             '单次批量/流式请求可含多个输入，严格单输入盒只有一个符号，证据不足时拒绝裁决。', '',
             '原文未规定“三种协议”的具体实现，本 Demo 将其解释为 batch / stream / single。'
             '原文的 t 与 t+1 记号冲突按用户要求统一为两份恢复。'
             '公开 Judge 信任注册表快照与采集者签名；不声称签名可证明远端模型执行。'
             '复制完整码字或恶意采集者超出该防栽赃实验，片段拒绝不等于无条件不可嫁祸证明。', '',
             'LoRA 只更新输出头两个词元行，秩 8；缓存冻结主干特征用于训练，'
             '最终读出总是重新运行完整模型。效用为 12 条固定留出文本的困惑度和下一词元一致率，'
             '通过标准预先固定为困惑度相对增加 ≤10% 且一致率 ≥90%；不是通用能力基准。', '',
             '## 密码学与码层', '', '| 检查 | 实测 |', '|---|---|']
    for k,v in core.get('crypto', {}).items():
        if isinstance(v,bool):
            lines.append(f'| {k} | {"PASS" if v else "FAIL"} |')
    for k,v in core.get('bridge', {}).items():
        if isinstance(v,bool):
            lines.append(f'| {k} | {"PASS" if v else "FAIL"} |')
    code = core.get('tardos', {})
    for k in ('length','threshold','single_hit_rate','pair_hit_rate'):
        if k in code:
            lines.append(f'| {k} | {code[k]} |')
    mc = code.get('monte_carlo')
    if mc:
        lines += ['', f"码层误告率：{mc['innocent_accusations']}/{mc['innocent_opportunities']} = "
                  f"{mc['false_accusation_rate']:.4%}；≤1%：{mc['empirical_fpr_le_1pct']}。",
                  f"独立场景出现误告的比例之 Wilson 95% 区间：{mc['scenario_false_rate_ci95']}。",
                  '该统计来自独立模拟码本、标记假设下三种策略，不是未运行的模型攻击误告率。']
    lines += ['', '## 模型、一次性盒与攻击', '']
    if model.get('status') != 'completed':
        lines += ['**模型相关指标未完成，不得作为通过。**',
                  f"原因：`{model.get('error', '仅选择 core 范围，未运行模型')}`", '']
    if model.get('embeddings'):
        lines += ['| 用户 | 位准确率 | 完整读出 | 效用通过 |', '|---|---:|---|---|']
        for x in model['embeddings']:
            lines.append(f"| {x['user']} | {x['bit_accuracy']:.3%} | {x['fully_readable']} | {x['utility_pass']} |")
    if model.get('protocols'):
        lines += ['', '| 协议 | 请求数 | 符号数 | Judge 接受 |', '|---|---:|---:|---|']
        for k,v in model['protocols'].items():
            lines.append(f"| {k} | {v['requests']} | {v['returned_symbols']} | {v['judge_accepted']} |")
    if model.get('attacks'):
        lines += ['', '| 攻击 | 指纹存活 | 位准确率 | PPL 相对变化 | 效用通过 |', '|---|---|---:|---:|---|']
        for k,v in model['attacks'].items():
            lines.append(f"| {k} | {v['fingerprint_survives']} | {v['bit_accuracy']:.3%} | "
                         f"{v['relative_perplexity_increase']:.3%} | {v['utility_pass']} |")
        lines += ['', '攻击各自从相同水印模型恢复后依次运行，不是累计破坏。量化是实际权重 INT8 舍入后'
                  '反量化，执行仍为浮点；蒸馏是干净预训练同架构学生的新 LoRA 头，不是从零训练整个学生。']
    if model.get('end_to_end'):
        lines += ['', '## 三层归因', '', '```json',
                  json.dumps(model['end_to_end'], indent=2, ensure_ascii=False), '```']
    if model.get('model_false_accusation'):
        fp = model['model_false_accusation']
        lines += ['', f"模型合谋实验误告：{fp['innocent_accusations']}/{fp['innocent_opportunities']} "
                  f"= {fp['false_accusation_rate']:.4%}。30 个相关场景不能证明总体误告概率上界。"]
    lines += ['', '## 复现', '', 'Windows PowerShell（仓库根目录）：', '', '```powershell',
              '.\\.venv\\Scripts\\python.exe run_demo.py --device cpu',
              '.\\.venv\\Scripts\\python.exe -m pytest -q', '```', '',
              'Linux/macOS：`./.venv/bin/python run_demo.py --device cpu`。详见 README 的安装与离线步骤。',
              '详细参数、逐场景结果和版本见同目录 metrics.json；完整输出和异常原文见 demo_output.log。', '',
              '## 局限', '', '研究 Demo，不是已审计的生产托管系统。Python 大整数不保证常数时间；'
              '演示追踪密钥为固定可复现实验值；真实系统应安全生成并保护。无 Refresh、TEE、链上 Judge 或通用 SNARK。'
              '不复制原文占位指标。Linux 实测不能视为 Windows 已实测。']
    (output/'DEMO_REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core-only', action='store_true', help='explicitly skip all model experiments')
    parser.add_argument('--model-path', help='offline local SmolLM2 directory')
    parser.add_argument('--revision', default='main', help='HF commit/tag; resolved revision is recorded')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--length', type=int, default=256)
    parser.add_argument('--steps', type=int, default=800)
    parser.add_argument('--attack-steps', type=int, default=12)
    parser.add_argument('--trials', type=int, default=2000)
    parser.add_argument('--cache', type=Path, default=ROOT/'.cache'/'huggingface')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs')
    args = parser.parse_args()
    for key in ('threads','length','steps','attack_steps','trials'):
        if getattr(args,key) < 1:
            parser.error(f'--{key.replace("_","-")} must be positive')
    os.environ.setdefault('HF_HOME', str(args.cache.resolve()))
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:6]
    output = args.output.resolve()/run_id
    output.mkdir(parents=True, exist_ok=False)
    metrics = {'run_id':run_id, 'started_utc':datetime.now(timezone.utc).isoformat(),
               'status':'running', 'scope':'core' if args.core_only else 'full',
               'parameters':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
               'runtime':{'python':sys.version, 'platform':platform.platform()},
               'model':{'status':'not_run'}}
    for package in ('torch','transformers','numpy','cryptography','huggingface-hub'):
        try:
            metrics['runtime'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            metrics['runtime'][package] = 'not_installed'
    def persist():
        (output/'metrics.json').write_text(json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
        report(metrics, output)
    exit_code = 3
    started = time.perf_counter()
    with (output/'demo_output.log').open('w', encoding='utf-8', buffering=1) as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr,log)):
            print('TVMC run:',run_id, '\nOutput:', output, '\nParameters:',metrics['parameters'],flush=True)
            try:
                from tvmc.experiment import core_experiment, model_experiment
                core,state = core_experiment(args.length,args.trials)
                metrics['core'] = core
                persist()
                core_pass = (all(v for v in core['crypto'].values() if isinstance(v,bool))
                             and all(v for v in core['bridge'].values() if isinstance(v,bool))
                             and core['tardos']['single_hit_rate']==1
                             and core['tardos']['all_pair_strategies_hit']
                             and core['tardos']['monte_carlo']['empirical_fpr_le_1pct'])
                if args.core_only:
                    metrics['model'] = {'status':'not_run','reason':'explicit --core-only'}
                    metrics['status'] = 'core_passed_model_unrun' if core_pass else 'core_failed'
                    exit_code = 0 if core_pass else 2
                else:
                    metrics['model'] = {'status':'running'}
                    persist()
                    metrics['model'] = model_experiment(args,state,output)
                    passed = core_pass and metrics['model']['acceptance']=='passed'
                    metrics['status'] = 'passed' if passed else 'completed_with_failed_targets'
                    exit_code = 0 if passed else 2
            except Exception as exc:
                traceback.print_exc()
                metrics['status'] = 'blocked_or_error'
                partial = output/'model_progress.json'
                if partial.exists():
                    metrics['model'] = json.loads(partial.read_text(encoding='utf-8'))
                metrics['model'].update(status='blocked_or_error', error=f'{type(exc).__name__}: {exc}')
                exit_code = 3
            finally:
                metrics['seconds'] = time.perf_counter()-started
                metrics['exit_code'] = exit_code
                persist()
                print(f"Final status: {metrics['status']}; exit={exit_code}; artifacts={output}", flush=True)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
