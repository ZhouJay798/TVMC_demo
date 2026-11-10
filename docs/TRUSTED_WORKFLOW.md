# 逐人模型分发、签名登记与追踪实验

这是**单机研究演示**，包含本地模拟的发行方、登记方、见证方和三个接收方。它可检验调用路径及受试篡改是否被拒绝，不能代表跨机构系统已部署或具有真实世界身份保证。

## 流程

1. 五位托管方生成演示身份承诺及知识证明，U1/U2/U3 申请会话。准入后按 3/3 生成份额和承诺，并绑定每位参与者的 RSA 公钥及持有证明。
2. 为每位参与者生成不同的 256 位 Tardos 码，登记方在训练前签名登记会话、接收人、个人公钥哈希、指纹码承诺以及偏置向量和解码阈值。本地模拟见证方签发包含随机挑战与有效期的检查点。
3. 为每个人分别训练指纹，使用 `--merged-delivery` 将 LoRA 权重更新合并到交付模型中。每个模型包使用不同的随机 AES-256-GCM 数据密钥，密钥使用相应接收人的 RSA-OAEP/SHA-256 公钥封装。
4. 发行方用 Ed25519 签名清单，绑定各密文、密钥封装、模型包哈希、接收人和会话；登记方检查清单和预登记一致后签名发行快照，本地模拟见证方再签发检查点。三个个人密文与签名清单合为一包，用 3/3 会话秘密派生的密钥进行外层加密。
5. 全体参与者提交份额和请求绑定证明，恢复会话秘密并产生本地签名回执；验证外层、发行方清单、登记记录和检查点，然后各人以自己的 RSA 私钥解封本人的 AES 数据密钥，解密并校验明文包哈希。
6. 模拟拿到疑似模型，在本机 HTTP 接口用一次请求提交 768 个探针，汇总为 256 位观测并用登记的码本与阈值解码。空白基座、单输入、转交副本和三对 50/50 参数混合分别对照。

指纹输出只指示**登记副本的来源线索**。不能据此单独认定实际泄漏者；一份副本可以被转交或窃取。单输入不会生成 256 位证据；一次批量请求包含 768 个输入，两者不能混为一谈。

这些追踪探针也是训练时用于嵌入指纹的探针，不是训练未见过的独立触发输入。每三条来自同一个文本窗口的不同前缀，彼此相关，不能作为 768 个独立样本估算成功率或误告率；测试文本窗口主要用于模型效用对照。

合并前后的浮点计算顺序不同，不能要求二者数值逐位相等。当前实验记录所有探针分数，使用 `atol=1e-4, rtol=1e-5` 比较，再额外要求汇总后的 256 位指纹观测完全相同；报告给出最大分数差和位变化数量，差异超出任一条件都会将该检查标记为未通过。这是演示使用的数值容差，并非模型安全性门槛。

## 运行

在项目根目录安装 README 的依赖后，离线运行合成集成验证：

```bash
HF_HOME="$PWD/.cache/huggingface" .venv/bin/python -m pytest -q
HF_HOME="$PWD/.cache/huggingface" .venv/bin/python run_bundle_experiment.py \
  --offline-fixture --merged-delivery --signed-registry --persist-bundle \
  --output work/trusted_workflow/fixture_run
```

输出目录不能事先存在。`--offline-fixture` 使用随机初始化的小型 Transformer 和合成文本，并非 SmolLM2、WikiText-2，也不代表真实模型精度或鲁棒性。密码学小载荷重复 5 次；模型训练及完整三模型包各运行一轮。`WORKFLOW_REPORT.md` 汇总实际指标，原始数据在 `crypto_metrics.json`、`model_metrics.json`、登记记录、签名清单和各 HTTP 记录中。

运行真实 SmolLM2 时，先按下述命令下载固定版本并校验权重：

```bash
.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('HuggingFaceTB/SmolLM2-135M', revision='93efa2f097d58c2a74874c7e644dbc9b0cee75a2', local_dir='models/SmolLM2-135M', allow_patterns=['*.json','model.safetensors','merges.txt','vocab.json'])"
.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('Salesforce/wikitext', repo_type='dataset', revision='b08601e04326c79dfdd32d625aee71d232d685c3', cache_dir='.cache/huggingface', allow_patterns=['wikitext-2-raw-v1/*.parquet'])"
printf '%s  %s\n' 80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1 models/SmolLM2-135M/model.safetensors | sha256sum -c -
HF_HOME="$PWD/.cache/huggingface" .venv/bin/python run_bundle_experiment.py \
  --merged-delivery --signed-registry --persist-bundle \
  --output work/trusted_workflow/real_run
```

真实模型运行需要较多内存和磁盘。若网络受限，应在云环境网络允许列表保留已有项并添加 `huggingface.co`、`*.huggingface.co`、`*.hf.co`；设置环境变量或密钥不能代替网络授权。具体部署若出现其他下载域，应按实际错误单独检查，不要关闭证书验证。

### 使用 PyPI 上的公开量化模型和公开文本

如果无法访问 Hugging Face，但能够访问 PyPI 和 GitHub，可以使用公开的 [llm-smollm2 0.1.2](https://pypi.org/project/llm-smollm2/0.1.2/) 包，其中包含由 QuantFactory 量化的 SmolLM2-135M-Instruct Q4_1 权重；包的说明指出原始模型来自 HuggingFaceTB。这里使用**量化后再反量化为 float32 的模型**，不能称为原始未量化检查点。wheel 的固定 SHA-256 是 `bcc81830d10ce7d9e76640cad826a4b79ed3e4547c78a0be5c4f2fb0e2448c70`，内含 GGUF 的固定 SHA-256 是 `b179c9523d0e6a0f98a330c7562b682750a6f8c8c15e5bc70ea373728110db53`。代码仅读取归档中指定的权重文件，不安装或执行该软件包。

这是第三方分发的量化包：wheel 哈希与 PyPI 该版本元数据一致，GGUF 哈希用于固定本次文件；这里没有独立核对原模型作者对该量化文件的签名或官方哈希。哈希固定文件内容，不单独证明发布者身份或模型无恶意。

GGUF 中没有独立输出头，必须与 token 嵌入权重绑定：转换器会检查输入文件布局并明确绑定。转换模型使用自身的 tokenizer；转换完成后的浮点文件不能与前述原始 SmolLM2 校验和混淆。转换只需做一次：

```bash
.venv/bin/python -m pip install -r requirements-public-model.txt
mkdir -p .cache/public_model .cache/public_data
.venv/bin/python -m pip download --no-deps --only-binary=:all: --dest .cache/public_model llm-smollm2==0.1.2
HF_HOME="$PWD/.cache/huggingface" .venv/bin/python prepare_public_model.py \
  --wheel .cache/public_model/llm_smollm2-0.1.2-py3-none-any.whl \
  --output models/SmolLM2-135M-Instruct-Q4_1-converted
curl -fsSL https://raw.githubusercontent.com/karpathy/char-rnn/6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e/data/tinyshakespeare/input.txt -o .cache/public_data/tinyshakespeare-6f9487a.txt
printf '%s  %s\n' 86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed .cache/public_data/tinyshakespeare-6f9487a.txt | sha256sum -c -
HF_HOME="$PWD/.cache/huggingface" .venv/bin/python run_bundle_experiment.py \
  --model-path models/SmolLM2-135M-Instruct-Q4_1-converted \
  --text-corpus .cache/public_data/tinyshakespeare-6f9487a.txt \
  --merged-delivery --signed-registry --persist-bundle --output work/trusted_workflow/public_run
```

本模式将 Tiny Shakespeare 切成互不重叠的文本窗口，前 256 窗口生成探针、接下来 128 窗口校准、再后 64 窗口测试。窗口来自同一文本集合，不能称作互相独立的文档，也不是 WikiText-2；测试结果不能直接代表其他语言或数据分布的表现。若上述网络仍不可用，可在安全可信的联网环境准备并复制文件，继续验证哈希。

本流程的 `--merged-delivery` 不生成原有独立 `*.pt` 适配器。`run_c123_validation.py` 依赖旧的独立适配器，不能直接用于本模式；不能将历史 C1/C2/C3 结果当作本次合并模型的新结果。

可另用 `run_c1_offline_control.py --run-dir <运行目录>` 执行 5 次独立小载荷对照，检验完整份额能否绕过服务离线插值；该对照不会重新训练或解密真实模型。如要将对照写入本次报告，应在报告生成后再次调用 `tvmc.workflow_report.write_workflow_report` 读取更新后的结果。

## 复核和可信边界

`audit_signed_registry.py` 需要从独立可信渠道获得发行方、登记方、见证方公钥以及预期挑战值；默认要求检查点在当前时间仍有效，过期会拒绝。`--historical` 仅按见证者自报的时间重放检查，**不能证明当前有效或时间可信**。为演示可以使用运行目录下的本地公钥和挑战，但它们不是独立信任根。例如：

```bash
.venv/bin/python audit_signed_registry.py --run-dir work/trusted_workflow/fixture_run \
  --trusted-anchors work/trusted_workflow/fixture_run/local_demo_trust_anchors.json \
  --challenge '<从可信渠道获得的 registry_checkpoint.json 对应 challenge>' --historical
```

本地演示没有独立操作的见证方、持久化防回滚/防分叉、可信时钟、真实身份认证、联网端点的远程证明或不可重放的分布式服务。本地检查点仅拒绝与其所固定的快照不匹配的已知旧记录，不能证明看到的是全局最新记录。历史预登记时间同样仅由本地见证进程声称。

演示码本/种子可预测且会被记录，无法抵抗了解材料的攻击者伪造或完整复制指纹。合并权重避免交付包中直接附带可丢弃的独立适配器，但不能阻止公开基座差分、继续训练、蒸馏或转交。应进一步研究秘密管理、独立见证服务、远程模型来源验证和多种攻击的实际检出率。签名回执证明本地服务对请求的声明，无法阻止份额持有人另行离线插值；密钥泄漏和恶意发行方也不在保障范围内。

`work/` 含码本、开启值、模型包和内部元数据，应按敏感数据管理，不提交或公开。完整生成进程仍可访问模型明文；3/3 只约束按本实验接口恢复外层，不保证进程、宿主机或托管方环境独立。

**本次并未实现可长期恢复的真实交付服务。** 脚本在进程中生成并使用身份秘密、份额、会话密钥和三把 RSA 私钥；它们不持久保存。即使设置 `--persist-bundle` 保存外层密文，进程退出后也不能仅凭该密文再次解开；个人明文副本仅在受控临时目录中加载，随后被清理。实际部署需要让各接收方通过认证的独立通道保管自己的份额和私钥，并设计密钥备份、权限撤销及恢复，不应把这组演示密钥写入公共代码库。
