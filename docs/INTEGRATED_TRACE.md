# 自然语言冗余指纹、二阶补偿剪枝与文字蒸馏的完整追踪实验

入口 `run_integrated_trace.py`。原入口和原蒸馏实验保留。本实验不直接加载旧统计指纹老师，而是新建注册会话后，从干净SmolLM2-135M为5个用户训练自然语言冗余指纹。不能沿用先前dense统计老师的成功率。

## 角色与事先登记

五个接收用户分别绑定session-1至session-5。授权恢复参与方分别为[1,2]、[2,3]、[3,4]、[4,5]、[1,5]。Shamir(2,5)恢复相同基础模型解密密钥；逐会话证明绑定会话、epoch和基础模型SHA256。每个会话实际恢复并解密验证SHA256。5个注册Tardos码字绑定身份种子与承诺；统计密钥映射也在训练前写入 `registration_before_training.json`。

用户ID是登记接收方，不是实名身份认证，也不是证明参与恢复的托管方泄露。

## 四个阶段

1. natural：768个自然文本输入，每位3个前缀，软合并读取256位；1200步训练，正常文本效用约束。
2. pruned：使用独立校准语料计算二阶统计，SparseGPT式误差补偿，约30% backbone Linear权重置零。不是仅删除30%小权重；embedding/head不计入该分母。
3. text：未剪枝自然指纹老师使用0.2密钥选词偏置，64个提示各生成32词元。干净基座学生训练全词表输出头LoRA，500步，clean KL=10。
4. pruned_text：剪枝老师以同样偏置生成文字，再蒸馏到干净基座学生。学生不保留老师的剪枝稀疏性；它是剪枝后再进行文字蒸馏，不是学生也被剪枝。

所有用户固定生成种子4501、学生种子3301，数据量与训练预算一致。学生只接收生成token，不接收密钥、秘密触发输入或老师概率表。学生特征从自己的干净基座重新计算，尤其不使用剪枝老师的隐藏状态。

效用统一在WikiText-2 test offset256的64段、5616预测词元上计算。这是固定子集，不是完整标准基准。剪枝后重新计算真实主干特征，不使用干净模型特征冒充剪枝PPL。老师另测施加0.2偏置后实际输出分布的PPL。

## 两种公开验证，不能混称

- `natural_verification`：原论文式Tardos Judge；密钥层、自然指纹部署层、会话桥接层。与原 `verify_evidence.py` 兼容。
- `statistical_verification`：新增研究性统计验证；采集者签名绑定整个公开上下文摘要、endpoint、nonce和词表漂移向量。验证者重新计算随机密钥检验，并检查登记用户和密钥会话对应。不是原论文Tardos不可嫁祸/抗合谋证明，也不是自然指纹恢复成功的替代说法。

两种证据都依赖可信登记快照和诚实采集者，不是远程模型执行证明。公开测试种子不适合生产密钥。新增采样偏置不在模型权重中，绕过接口生成不受该层保护。统计验证需要完整词表概率和干净基座，不是纯聊天文字检测。

证据目录 `user_N/阶段名/` 包含公开上下文、自然指纹证据、统计响应签名证据和各层结果。失败也导出，不伪造通过。干净模型负对照单独输出。

## Windows

```powershell
git pull --ff-only
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: -r requirements-research.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe run_integrated_trace.py --model-path ".\outputs\你的原始运行目录\decrypted_model" --offline --output ".\outputs\integrated_trace_local"
```

替换实际模型目录。`--offline`要求WikiText-2缓存完整；缺少数据缓存时去掉该参数。输出目录必须不存在。无需旧指纹.pt文件，会重新注册、嵌入、生成文字和训练学生。CPU可能需要十余分钟或更久，模型文件与学生权重不上传GitHub。

复核某个阶段（无需下载模型）：

```powershell
.\.venv\Scripts\python.exe verify_integrated_evidence.py --run-dir ".\outputs\integrated_trace_local\user_1\pruned"
```

将user_1换为user_2至user_5，阶段可选natural、pruned、text、pruned_text。该命令同时显示两种验证结果，退出0仅表示自然指纹原式三层全部通过，退出2表示这条原式证据链未通过；研究性统计通过不会把退出码改成0。

运行主脚本成功只表示完成实验，不能理解为所有论文要求通过。查看metrics.json和实际结果表，10% PPL工程目标不是原文规定。

`evaluate`输出的next_token_agreement是相对当前主干的未加LoRA输出；剪枝时该主干本身已改变，因此它不能衡量剪枝相对原模型的一致率。本次表格仅据统一干净PPL基准比较效用，不用该局部一致率宣称压缩无损。
