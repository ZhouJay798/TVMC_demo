# TVMC 鲁棒性配对实验报告

运行：`20261025T083201Z-54e20b`

状态：**completed_improvements_verified**；退出码：0
全部所测攻击均解决：**False**

这是模型层配对对照；不是新公开身份归因证据。原密码学端到端 Demo 保持独立。
通过改进门槛只说明所列比较有效，不等于所有研究目标成立。

## 固定设置

SmolLM2-135M；5 用户；256 码位；固定 Tardos 阈值；原触发 256 次模型输入，新版 3 倍冗余为 768 次。自然前缀错误可能相关，不声称独立重复带来的理论保证。
效用标准保持 PPL 相对增加 ≤10% 且下一词元一致率 ≥90%。测试文档、模型、源码的哈希见 metrics.json。
WikiText 校准、蒸馏和触发文档相互分开。测试按固定 offset 取 64 个段落，不同段落可能来自同一文章，结果不代表全数据集或不同主题的独立泛化。

干净测试模型：{"nll": 3.6599151445499034, "perplexity": 38.85804541449733, "tokens": 6073, "next_token_agreement": 1.0}

## 嵌入结果

| 配置 | 用户 | 位准确率 | 定位 | PPL 增幅 | 一致率 | 效用通过 |
|---|---:|---:|---|---:|---:|---|
| original | 1 | 99.22% | [1] | 16.90% | 92.56% | False |
| original | 2 | 99.22% | [2] | 15.28% | 89.23% | False |
| original | 3 | 100.00% | [3] | 17.08% | 90.60% | False |
| original | 4 | 99.22% | [4] | 23.43% | 89.56% | False |
| original | 5 | 100.00% | [5] | 24.89% | 89.87% | False |
| natural_redundant | 1 | 97.27% | [1] | 1.54% | 96.30% | True |
| natural_redundant | 2 | 98.44% | [2] | 3.25% | 95.26% | True |
| natural_redundant | 3 | 96.88% | [3] | 2.17% | 95.59% | True |
| natural_redundant | 4 | 96.48% | [4] | 4.13% | 95.46% | True |
| natural_redundant | 5 | 96.48% | [5] | 2.82% | 95.98% | True |

## 压缩：相同位宽/相近实际稀疏率的配对比较

| 方法 | 真实零权重率 | 干净模型 PPL 增幅 | 新版定位 | 新版 PPL 增幅 | 新版一致率 | 新版效用通过 |
|---|---:|---:|---|---:|---:|---|
| row_int8 | 1.236% | 2.64% | [1] | 4.19% | 92.33% | True |
| group_int8 | 0.795% | 0.51% | [1] | 1.96% | 94.07% | True |
| magnitude30 | 30.098% | 90.14% | [1] | 99.33% | 53.35% | False |
| sparsegpt30 | 30.000% | 10.53% | [1] | 12.01% | 80.75% | False |
| sparsegpt_mlp30 | 30.000% | 19.04% | [1] | 20.54% | 74.41% | False |

压缩行均有同一变换下的无水印对照、旧版指纹对照，以及原 12 句语料指标，详见 JSON。INT8 是权重舍入后浮点执行，不是 INT8 内核或压缩存储。剪枝恢复始终不使用秘密码字/触发器。所有模型攻击以用户 1 为代表；其他用户只完成干净嵌入评估。

## 输入过滤与改写

| 配置/攻击 | 码位覆盖 | 触发拒绝率 | 正常请求拒绝率 | 定位 |
|---|---:|---:|---:|---|
| original/prefix | 0.00% | 100.00% | 0.00% | [] |
| original/adaptive_nll | 0.00% | 100.00% | 4.69% | [] |
| original/rewrite | 100.00% | 0.00% | 0.00% | [1] |
| natural_redundant/prefix | 100.00% | 0.00% | 0.00% | [1] |
| natural_redundant/adaptive_nll | 92.19% | 13.80% | 3.12% | [1] |
| natural_redundant/rewrite | 100.00% | 0.00% | 0.00% | [1] |

NLL 阈值来自校准集的 95% 分位，不按秘密触发标签调阈值。拒绝率是正常服务可用性的代价；若 JSON 提供 rewrite_utility，它比较同样归一化的测试文本与干净模型，不能据此证明原任务语义完全保留。

## 蒸馏

| 配置/攻击 | 位准确率 | 定位 | PPL 增幅 | 效用通过 |
|---|---:|---|---:|---|
| original/soft | 98.83% | [1] | 13.45% | False |
| original/hard | 74.22% | [1] | 157.39% | False |
| natural_redundant/soft | 86.72% | [1] | 1.56% | True |
| natural_redundant/hard | 57.42% | [] | 439.88% | False |

学生使用相同干净预训练基座，仅训练新输出头 LoRA。软蒸馏使用精确完整词表 KL；硬蒸馏使用教师实际 argmax。两者均不接收秘密触发样本。这不是从零训练整个学生，也不能代表所有蒸馏算法。

## 改进门槛

```json
{
  "group_int8_utility": true,
  "same_rate_pruning_better": true,
  "adaptive_filter_better": true,
  "natural_embedding_utility": true,
  "no_innocent_accusations": true
}
```

## 复现（Windows PowerShell）

```powershell
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: -r requirements-research.txt
.\.venv\Scripts\python.exe run_robustness.py --device cpu
```

模型和数据已缓存时追加 `--offline`。当前 Python 与依赖版本见 metrics.json；提供 Windows 命令不代表所有平台均已实测。测试集不能再用于调参；改动方法后应另设未使用的评估数据。

报错：无执行异常。未达标指标仍保留，不降低标准。
