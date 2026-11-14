# TVMC：模型指纹分发与追踪演示

本项目按新版论文流程运行：**五名托管方登记 → 双门限密钥 → 真实模型共享载体指纹 → 三人批准交付 → 本人解密 → 疑似模型筛查与归因 → 证据复核／API 转发申辩。**

本版本以“可复跑的研究演示”为目标。旧的全会话模型大包、重建会话私钥和逐人 LoRA 指纹入口已移除。旧代码可在提交 `e858e3d` 中找回；`results/` 内旧日期报告是历史结果，不属于新方案。

[最新整链路与攻击实验（2026-11-14）](results/acceptance_20261114/README.md)：真实模型文件、直接 HTTPS API、转发反驳三组主流程通过；模型平均、基座复位及公开基座截断暴露鲁棒性缺口。另见[历史验收记录](results/reproduction_20261113/README.md)。

2026-11-14：新增[成员独立核算得分的解密批准入口](docs/MEMBER_SCORE_APPROVAL.md)，拒绝指纹密文冒充得分、绑定整批请求，并在公开核验中检查成员批准。仍为同进程角色模拟，非独立密钥保管服务。

新增[CA 证书认证](docs/CA_CERTIFICATES.md)：分发者、成员身份和个人接收公钥均先验 X.509 证书及撤销状态，再使用公钥。

[CA 与 API 防护开关对照实验](results/security_controls_20261114/README.md)：13 类协议攻击各 5 轮，以及真实 HTTPS 的副本归属、转发、伪造响应对照；包含正常业务和无法防御的信任边界。

## 1. 安装

推荐 Python 3.11/3.12。Windows 把下面 `.venv/bin/python` 改成 `.venv\Scripts\python.exe`。

```bash
python -m venv .venv
.venv/bin/python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest -q
```

密码流程和 HTTP/TLS 测试不需下载模型。实际模型实验需要后面的本地模型和文本文件。

## 2. 准备模型和文本

当前实验使用本地 `SmolLM2-135M-Instruct`，由固定 Q4_1 GGUF 解量化为浮点权重；**不是 GPT-2，也不是原始全精度 SmolLM2**。复跑请保持相同来源。

```bash
.venv/bin/python -m pip install -r requirements-public-model.txt
.venv/bin/python -m pip download --no-deps llm-smollm2==0.1.2 -d .cache/public_model
.venv/bin/python prepare_public_model.py --wheel .cache/public_model/llm_smollm2-0.1.2-py3-none-any.whl --output models/SmolLM2-135M-Instruct-Q4_1-converted
```

转换器检查固定 SHA256，已经存在的模型目录无需再次转换。模型仅作为本地实验资产，不上传 GitHub。

实验语料为 Tiny Shakespeare。可用下列脚本下载固定 Git 修订的原文，下载失败则停止，不用占位文本替换：

```bash
.venv/bin/python prepare_data.py
```

训练、通用输入统计、探针、载体强度校准和最终效用评估使用不重叠的文本区间；有限的单语料测试不能代表全部业务任务。

## 3. 一键实验

正常文件交付与追踪：

```bash
.venv/bin/python run_experiment.py --output work/file_demo
```

默认运行先完成正常流程。需要单独检查版本截断时添加 `--sweep`。

增加剪枝、INT8 量化/解量化、短程全参数微调、合谋平均和公开基座复位对照：

```bash
.venv/bin/python run_experiment.py --output work/attack_demo --attacks
```

实际本地 HTTPS API，多人分别查询并核对结果：

```bash
.venv/bin/python run_experiment.py --output work/api_demo --evidence api
```

模拟“疑似 API 转发到 U1 合法服务”，验证申辩后撤销权重泄漏指认：

```bash
.venv/bin/python run_experiment.py --output work/relay_demo --evidence relay
```

每次必须使用新输出目录，避免覆盖证据。默认 5 人、3 人门限、3 个指纹副本、128 位归因指纹、16 位同源筛查、每位 3 个探针；不等于论文的 100 副本/2048 位配置。CPU 运行需要数分钟以上，文件和密码验证会增加耗时。

## 4. 查看结果

- `report.json`：总体结果、每项检查、模型能力、指纹得分和限制。
- `copies.json`：每个接收人、批准人、指纹读回和能力指标。
- `carrier_calibration.json`：独立测试码字的载体强度选择过程。
- `version_sweep.json`：使用 `--sweep` 或 `--attacks` 时记录截断检查，默认明确标为未运行；失败不会伪装成抗攻击成功。
- `trace.json`：加密评分、解密证明、归因结果。
- `public_journal.jsonl`、`trust.json`：日志及本次实验固定的公钥与见证检查点。
- API 模式另有 `api_evidence.json`；转发模式另有 `rebuttal.json`。
- `recipients/v2-U*/model.zip`：实际接收并校验后的完整指纹模型。

`passed` 表示该次演示流程的检查通过，**不是所有攻击都有效防御**。API 转发模式中，命中 U1 是来源线索；有效申辩后输出转发关系，不直接说 U1 泄漏了权重。

## 5. 代码位置

| 模块 | 职责 |
|---|---|
| `run_experiment.py` | 唯一模型实验主入口 |
| `tvmc/shared_carriers.py` | 实际模型 MLP 闭式共享载体、秘密 token 对、自然探针 |
| `tvmc/paper_threshold.py` | Pedersen/Feldman DKG 本地演示、TDH2 混合胶囊、部分解密证明 |
| `tvmc/paper_hpke.py` | HPKE 个人封装，官方 RFC 测试向量校验 |
| `tvmc/paper_workflow.py`、`stream_delivery.py` | 身份签名、两阶段批准、分块文件交付 |
| `tvmc/paper_trace.py` | 加密码本、门限评分、探针托管、公开复核、申辩后重算 |
| `tvmc/evidence.py`、`api_server.py` | 实际 HTTPS 取证、相同码字要求、服务日志申辩 |
| `tvmc/transparency.py` | 持久化日志、Merkle 包含性、签名与见证检查点 |
| `tvmc/version_attacks.py` | 截断扫描、变换和合谋对照 |

[流程说明](docs/FLOW.md) · [限制与论文差异](docs/LIMITATIONS.md) · [旧代码清理记录](docs/CHANGES.md)

仅使用公钥和证据重新核验（文件模式会再次加载真实模型）：

```bash
.venv/bin/python verify_experiment.py work/file_demo --ca-root /trusted/tvmc-root.pem
```

必须通过独立可信渠道预置 CA 根并传入 `--ca-root`，或预先在目录外保存 `trust.json` 的 SHA256 并通过 `--trust-sha256` 传入；两者都没有时拒绝验收。结果目录中的根副本不能自行证明可信。API 原始来源判定与转发申辩后结论分别输出。
