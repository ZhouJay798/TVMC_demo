# 份额刷新、身份轮换与周期模型实验

`run_epoch_experiment.py` / `run_epoch_crypto.py` / `verify_epoch_history.py` 是独立实验入口，不替换原 `run_demo.py`。`tvmc/epochs.py` 在原密码学原语上增加指定参与方的任意次数 Feldman 多项式、严格会话/周期标签、完整成员恢复及原子刷新。A 使用托管方 1、2 的 2/2 份额，B 使用 2、3、4 的 3/3 份额。

每个系列登记 E0、E1、E2 三个交付会话，分别生成独立密钥。指纹种子采用原项目的 canonical JSON 编码，绑定会话、交付周期及有序参与方身份承诺。每个交付模型包含完整基座、分词器和训练好的 HeadLoRA 适配器，并打包整体加密。刷新 E0/E1 的现有会话时不重新生成模型或密钥；身份轮换单独执行，历史公开记录和证明保留。

## 环境与运行

沿用项目 `.venv`、`requirements-research.txt`、固定版本 `models/SmolLM2-135M` 和 `.cache/huggingface` 内 WikiText-2 缓存。无需新增依赖，默认 CPU。输出目录必须不存在。下面的三个实验步骤顺序执行；报告步骤只读取已完成结果。

```bash
.venv/bin/python -m pip check
.venv/bin/python -m pytest -q
.venv/bin/python run_epoch_experiment.py --output work/epoch_experiment/new_run
.venv/bin/python run_epoch_crypto.py --run-dir work/epoch_experiment/new_run
.venv/bin/python verify_epoch_history.py --run-dir work/epoch_experiment/new_run
.venv/bin/python run_epoch_report.py --run-dir work/epoch_experiment/new_run --output outputs/epoch_report
```

模型阶段：六个模型各1200步训练一次，每次使用新的随机身份种子；训练初始化种子为71001至71006。负对照是无指纹基座。一次HTTP请求提交768个自然文本前缀，返回 `logit(1)-logit(0)` 分数，每3个输入聚合为1位，总计256位。每次使用独立本地服务，第二次请求应返回409。`serve.make_server` 新增可选 `max_prompts` 与 `response_field` 参数，原默认256输入及 `bits` 响应不变。

密码学阶段：5组独立随机性，每组同时测试重抽及不重抽身份。E0资产刷新两次，E1资产刷新一次，E2不刷新。只第一组重抽分支对应真实训练的六个身份种子；其他密码学分支与重复复用这六个固定模型包作为加密载荷，不构成新模型训练。跨周期混合组合穷举各参与方可选的所有已保存周期，排除全体来自同一周期的组合。

历史复测：密码学阶段完成后解密实际交付包，重新加载完整模型，在新的一次性HTTP服务上用历史登记记录追踪，保留原始响应分数。该复测没有重新训练模型。

## 输出与解释

- `model_metrics.json`：PPL、可观察/正确位、六候选分数、定位和HTTP状态。
- `crypto_metrics.json`：5组检查明细，公开承诺、历史证明、上下文和密文哈希。
- `historical_metrics.json`：实际解密模型的重新加载、旧证明验证、旧种子重算和HTTP复测。
- `*.zip` / `*.aesgcm`：完整模型明文包及第一组重抽分支密文。
- `private_setup.pkl`：本地测试私密状态，仅可加载本实验自行生成的文件，不得当成公开证据分享。`work/` 被 Git 忽略。
- 报告目录：Markdown、CSV、Excel、源文件及公开证据压缩包；不包含身份秘密、密钥、份额或权重。

退出码0代表脚本执行完成，是否达标须检查指标中逐项 `passed`、`correct_localization` 和 `failed`；不使用模型训练重试或选优来隐藏失败。直接混用份额的碰撞/解密次数仅为观测结果。持有完整旧份额仍可恢复旧密钥，指纹定位只识别登记名单，不能裁定哪个托管方实际泄漏。

私密认证通道、可信历史登记、真实分布式故障处理和安全擦除仍是部署假设。这里的删除只模拟隔离字典移除旧引用，测试工具故意保留被窃取的旧秘密用于边界验证。Python 大整数不是恒时实现。
