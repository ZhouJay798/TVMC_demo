# TVMC Demo 实测报告

- 运行 ID：`20261026T173212Z-0a7b2d`
- UTC 开始时间：2026-10-26T17:32:12.116173+00:00
- 总状态：**completed_with_failed_targets**
- 模型阶段：**completed**

## 设计与范围

Shamir (2,5) 使用一次多项式；Feldman 承诺和绑定会话的 AND-Schnorr Fiat–Shamir NIZK；HKDF-SHA256 派生 AES-256-GCM 密钥。模型阶段加密、恢复并加载完整 SmolLM2 safetensors 权重。

Tardos 使用截断 arcsine 分布、对称累加得分和运行前固定的 Bernstein 阈值。5 个用户对应 5 个发布会话，不能据此宣称某个托管方就是泄露者。单次批量/流式请求可含多个输入，严格单输入盒只有一个符号，证据不足时拒绝裁决。

原文未规定“三种协议”的具体实现，本 Demo 将其解释为 batch / stream / single。原文的 t 与 t+1 记号冲突按用户要求统一为两份恢复。公开 Judge 信任注册表快照与采集者签名；不声称签名可证明远端模型执行。复制完整码字或恶意采集者超出该防栽赃实验，片段拒绝不等于无条件不可嫁祸证明。

LoRA 只更新输出头两个词元行，秩 8；缓存冻结主干特征用于训练，最终读出总是重新运行完整模型。效用为 12 条固定留出文本的困惑度和下一词元一致率，通过标准预先固定为困惑度相对增加 ≤10% 且一致率 ≥90%；不是通用能力基准。

## 密码学与码层

| 检查 | 实测 |
|---|---|
| all_10_recovered | PASS |
| single_rejected | PASS |
| duplicate_rejected | PASS |
| tampered_share_rejected | PASS |
| tampered_ciphertext_rejected | PASS |
| tampered_nizk_rejected | PASS |
| cross_session_nizk_rejected | PASS |
| all_5_nizk_verified | PASS |
| aes_roundtrip | PASS |
| valid_accepted | PASS |
| seed_mismatch_rejected | PASS |
| cm_mismatch_rejected | PASS |
| missing_reg_rejected | PASS |
| expired_rejected | PASS |
| unsigned_fragment_frame_rejected | PASS |
| signed_fragment_frame_rejected | PASS |
| edited_receipt_rejected | PASS |
| replay_nonce_rejected | PASS |
| length | 256 |
| threshold | 66.15590890666381 |
| single_hit_rate | 1.0 |
| pair_hit_rate | 1.0 |

码层误告率：0/6000 = 0.0000%；≤1%：True。
独立场景出现误告的比例之 Wilson 95% 区间：[1.0842021724855044e-19, 0.001917047281252934]。
该统计来自独立模拟码本、标记假设下三种策略，不是未运行的模型攻击误告率。

## 模型、一次性盒与攻击

| 用户 | 位准确率 | 完整读出 | 效用通过 |
|---|---:|---|---|
| 1 | 100.000% | True | True |
| 2 | 99.609% | False | True |
| 3 | 99.219% | False | True |
| 4 | 100.000% | True | True |
| 5 | 100.000% | True | True |

| 协议 | 请求数 | 符号数 | Judge 接受 |
|---|---:|---:|---|
| batch | 1 | 256 | True |
| stream | 1 | 256 | True |
| single | 1 | 1 | False |

| 攻击 | 指纹存活 | 位准确率 | PPL 相对变化 | 效用通过 |
|---|---|---:|---:|---|
| finetuning | True | 97.266% | 1.800% | True |
| quantization | True | 100.000% | 18.398% | False |
| pruning | True | 88.281% | 455.358% | False |
| mixing | True | 85.938% | 0.053% | True |
| filtering | False | 0.000% | 1.296% | True |
| distillation | False | 47.656% | 0.000% | True |

攻击各自从相同水印模型恢复后依次运行，不是累计破坏。量化是实际权重 INT8 舍入后反量化，执行仍为浮点；蒸馏是干净预训练同架构学生的新 LoRA 头，不是从零训练整个学生。

## 三层归因

```json
{
  "key_layer": {
    "verified_participants": [
      1,
      2
    ],
    "asset_decrypted": true,
    "sha256": "80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1"
  },
  "deployment_layer": {
    "sessions": [
      "session-1"
    ],
    "consumers": [
      1
    ],
    "judge_accepted": true
  },
  "bridge_layer": {
    "binding_accepted": true,
    "attribution_scope": "session/consumer; participating custodians are not proven leakers"
  },
  "status": "passed",
  "failure_layer": null
}
```

模型合谋实验误告：0/90 = 0.0000%。30 个相关场景不能证明总体误告概率上界。

## 复现

Windows PowerShell（仓库根目录）：

```powershell
.\.venv\Scripts\python.exe run_demo.py --device cpu
.\.venv\Scripts\python.exe -m pytest -q
```

Linux/macOS：`./.venv/bin/python run_demo.py --device cpu`。详见 README 的安装与离线步骤。
详细参数、逐场景结果和版本见同目录 metrics.json；完整输出和异常原文见 demo_output.log。

## 局限

研究 Demo，不是已审计的生产托管系统。Python 大整数不保证常数时间；演示追踪密钥为固定可复现实验值；真实系统应安全生成并保护。无 Refresh、TEE、链上 Judge 或通用 SNARK。不复制原文占位指标。Linux 实测不能视为 Windows 已实测。
