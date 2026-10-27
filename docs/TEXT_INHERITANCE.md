# 可选的生成文字指纹分支

本分支保留 `run_distillation_research.py`、原统计指纹权重与原三种蒸馏协议。新入口 `run_text_inheritance.py` 只在实际文字生成时，对词表施加秘密符号向量对应的轻微概率偏置，再真正逐词采样。学生只接收生成文字，不接收密钥、秘密探测输入或老师概率表。默认原入口不会启用这个功能。

这是**生成接口层保护**，不是新训练进权重的防护。绕过接口、直接拿走模型权重重新生成，仍可能移除指纹。学生检测仍需要完整词表概率和干净基座，尚未实现只凭学生聊天文字定位。研究中公开随机种子只用于复现，不能直接当生产密钥。

## 实验口径

- 保持现有 `dense_0.5` 普通输出统计指纹老师；新增采样偏置 `strength × key`，temperature=1，无 top-k/top-p。
- 64 个普通提示，各生成 32 个词元，学生训练 500 步、干净基座 KL 系数 10；学生与数据预算不变。
- 验证集上优先选择只定位用户 1 的候选，再选老师输出 PPL 最低者；没有成功候选时仍保存失败结果。
- 同时报告老师**实际加偏置后的输出分布 PPL**、蒸馏学生 PPL、定位结果、无指纹负对照。不能只报告学生低 PPL 而隐藏老师损失。
- 重跑 soft12、soft500、hard-KL10，仍使用原老师接口。这是保留原分支的回归，不是证明所有接口同时叠加偏置后也完全不变。
- 主验收要求：老师和学生 PPL 相对干净基座均增加不超过 10%，正确且仅定位用户 1，三项旧实验通过，负对照不被指认。10% 是工程目标，不是论文要求。下一词元一致率另行报告，不冒充所有能力无损。
- 返回码 0 为上述有限验收通过，2 为目标未通过；报错仍需查看 traceback。结果不构成普适抗蒸馏证明。

## Windows 操作

在已有 Python 3.12 环境的项目根目录运行：

```powershell
git pull --ff-only
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: -r requirements-research.txt
.\.venv\Scripts\python.exe -m pytest -q
```

需要先有原统计指纹实验的 `statistical_user1.pt`。若没有，执行原入口生成（会运行完整研究实验，耗时较长）：

```powershell
.\.venv\Scripts\python.exe run_distillation_research.py --model-path ".\outputs\你的原始运行目录\decrypted_model"
```

找到 `outputs\distillation_research\运行编号\statistical_user1.pt` 后，填入下面的实际路径；不要把学生检查点误当老师。输出目录必须不存在，以防覆盖已有实验。

```powershell
.\.venv\Scripts\python.exe run_text_inheritance.py --model-path ".\outputs\你的原始运行目录\decrypted_model" --teacher ".\outputs\distillation_research\运行编号\statistical_user1.pt" --strengths 0 0.1 0.2 --test-offset 256 --output ".\outputs\text_inheritance_local"
```

模型和 WikiText-2 缓存完整后可加 `--offline`。GitHub 仅含代码、文本实验记录，不包含大模型和 `.pt` 检查点。使用既有研究依赖，无新依赖。CPU 运行需等待，阶段日志打印在终端，指标保存在输出目录的 `metrics.json`，生成 token ID 保存在 `text_*.json`。

## 固定参数的多随机种子配对复测

使用 `run_text_repeats.py` 新增五组实验，不搜索强度、不更改阈值。生成种子4502–4506、学生种子3302–3306；每组原方式与0.2偏置方式共享种子。此前4501/3301单次结果不计入五组统计。固定用户1、原64段提示和offset256评估集，验证的是随机种子变化下的表现，不代表多用户或跨数据泛化。

```powershell
.\.venv\Scripts\python.exe run_text_repeats.py --model-path ".\outputs\你的原始运行目录\decrypted_model" --teacher ".\outputs\distillation_research\运行编号\statistical_user1.pt" --output ".\outputs\text_repeats_local"
```

缓存完整可加 `--offline`。每组约需数分钟，阶段输出在终端；统计保存在输出目录 `metrics.json` 的 `summary`。原老师权重及分布固定，因此老师PPL测一次，不随生成随机种子变化；学生PPL报告均值、样本标准差和范围。定位成功定义为只指认用户1，错误指认另计。
