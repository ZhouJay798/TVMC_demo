# 多用户鲁棒性与剪枝感知指纹实验

这是研究原型，不是已经改造或重新签发的正式交付流程。实验保持历史 U1/U2/U3 交付包不变，另行训练受限的输出头行。原运行中的签名使用同次保存的公钥核验，因此只证明记录内部一致，不证明独立身份信任或当前状态。

实验结果位于 `work/multiuser_suite/` 和 `work/prune_resilience/`，包含模型、码字相关信息、秘密检测键或实验权重，不能提交仓库或公开。目录由项目 `.gitignore` 排除。报告生成脚本只导出统计数据。

## 原型设计

- 第一通道：保留交付模型原有的 0/1 输出分数和 Tardos 码。
- 第二通道：用 32 字节私有键选择 16 对不同输出词元，并为每个码位选择一对；仅训练 32 行输出权重。两个通道共享探针和模型隐藏特征，不可假设统计独立。
- 对照：只在原模型探针特征上训练备用输出行。
- 幅度剪枝增强：在原模型、10% 和 30% 每矩阵幅度剪枝的特征上训练备用输出行；20%、40% 和结构化剪枝未参与该轮训练。
- 结构化增强：再将 2%、5% 的 MLP 神经元成组置零样本加入训练，3%、7% 结构化置零未参与该轮训练。
- 训练每码位使用三个相关前缀中的前两个；第三个仅对这次适配训练留出，它已参与历史模型的初次嵌入，不能称为独立新探针。
- 所用阈值仍是原单通道演示阈值，多个通道取并集还需用独立负样本、多个用户和模型重新校准；本原型不声称可获得理论误报率。
- 原 `aggregate` 会将平均分数刚好为零判为位 1。新加的 `aggregate_with_abstention` 是可选研究函数，将零或经校准的弱信号作为擦除，不能让已删去的指纹重新出现；历史检测流程和本次结果未启用它。迁移前须版本化并重新校准阈值。

原交付流程对左填充的历史读取位置存在偏差。本实验原型显式读取最后一个非填充词元，但原交付模型、登记和嵌入训练并未因此自动修复。正式方案须统一重训、版本化、重新签发并验证。

## 研究入口

- `run_multiuser_robustness_suite.py`：三用户基线、真实 PyTorch 动态逐行 INT8、不同剪枝、全参/LoRA 微调、已知探针的定向输出头优化以及组合操作。该去指纹攻击不是 MEraser 复现。
- `run_multiuser_gguf_suite.py`：真实 GGUF Q8_0、Q4_0、Q4_1 文件及 llama.cpp CPU 推理，对比是否量化输出头。
- `run_multiuser_collusion_suite.py`：实际模型权重平均和公共基座差分，逐候选人报告 Tardos 得分。
- `run_prune_resilient_experiment.py`：训练及测试第一个剪枝感知备用通道。
- `run_structured_pruning_resilience.py`：对比再加入结构化剪枝样本的备用通道。
- `run_low_structured_pruning.py`：测试更温和的逐层 MLP 结构化置零及单层置零，比例按神经元数量取整。
- `run_activation_pruning_test.py`：用普通文本校准 MLP 激活贡献，在相同比例下比较激活感知结构化置零；没有使用指纹探针计算剪枝掩码。
- `run_prune_difference_attack.py`：仅通过精确公共基座和带指纹模型的输出头差异选择要置零的行；选行不读取检测键或探针，验证阶段再使用它们。
- `build_multiuser_experiment_report.py`：导出 Word、Excel 和 CSV 汇总。

各入口先验证本地签名、登记与载荷哈希的一致性。原始受限材料需要在本地已存在；报告不包含密钥、探针、码字或可用于直接恢复实验模型的权重。

## 复现参数

使用源模型 `SmolLM2-135M-Instruct-Q4_1` 量化后反量化为 FP32 的本地副本；指纹嵌入发生在此后。评估为 TinyShakespeare 窗口 384–399、最多 64 词元，总计约 1008 个预测词元；普通微调使用窗口 448–479、32 步。窗口不重叠，但来自同一文本，不是独立业务文档。

本次环境：Python 虚拟环境中的 PyTorch 2.5.1+cpu、Transformers 4.46.3、gguf 0.19.0、llama-cpp-python 0.3.36；GGUF 转换和量化工具是本地 llama.cpp 提交 `f3b407ca1bfeeecebf9e9ef19fcd6eed565bd70a`。真实 GGUF 测试用实际整数权重文件和 llama.cpp CPU 内核；PyTorch 动态量化还会处理激活，不能与纯权重舍入模拟混为一谈。

额外依赖列在 `requirements-experiments.txt`。其中 `llama-cpp-python` 可能需要本机 C/C++ 编译器；GGUF 转换及量化还需要上述版本的 llama.cpp 工具链。它们不属于项目核心交付依赖。

在本地受限目录创建 32 字节随机键后，可分别对三个用户运行：

```bash
umask 077
mkdir -p work/prune_resilience/new_run
.venv/bin/python -c "import secrets; from pathlib import Path; Path('work/prune_resilience/new_run/detector.key').write_bytes(secrets.token_bytes(32))"
for u in 1 2 3; do
  .venv/bin/python run_prune_resilient_experiment.py \
    --recipient "$u" --key work/prune_resilience/new_run/detector.key \
    --output work/prune_resilience/new_run --threads 1
done
```

下一轮 `run_structured_pruning_resilience.py --prior work/prune_resilience/new_run` 应使用同一检测键及相应先前输出。不要把检测键放进版本库、公开文档或交付包。所有测试均为小样本探索，不构成对任意剪枝、微调、量化、蒸馏或白盒攻击的保证。
