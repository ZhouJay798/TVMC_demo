# TVMC 本地 Demo

向老师汇报可直接阅读：[实验总览、问题原因、前后对照与答辩问答](docs/EXPERIMENT_DEFENSE_GUIDE.md)。

根据 `docs/TVMC_PAPER.md` 和 `docs/DEMO_PROMPT.md` 实现的 **Python 命令行研究 Demo**。默认用 HuggingFaceTB/SmolLM2-135M，无 GPU 自动使用 CPU。入口 `run_demo.py` 一条命令完成托管恢复、5 个用户的 LoRA 指纹、黑盒读出、Tardos 解码、身份裁决及攻击评估。

**结果以实际输出为准，不能把“脚本完成”当作“安全目标全部达标”。** 原文的数值均为占位，本项目不会复用这些数值。严格单输入一次性盒无法给出 256 个码位；输入过滤和蒸馏也可能移除指纹，报告如实记录失败。

## 新增：鲁棒性改进与配对对照实验

**2026-10-26 已重新训练并验证**：[最新结果、参数及 PPL 对照](results/robustness_20261026/README.md)。被拒绝输入不会进入指纹响应推理；NLL 过滤拒绝 106/768 个新版探测输入后，仍正确定位用户 1，正常请求拒绝率 2/64。新版分组 INT8 后 PPL 39.621，二阶补偿 30% 剪枝后 PPL 43.524，同一评估语料的干净基座为 38.858。原始日志、指标及可用 Excel 打开的 CSV 一并归档。过滤与压缩是独立实验，尚未验证两者联合攻击。

原 `run_demo.py` 保留为密码学端到端基线。新增 `run_robustness.py` 比较改进前后真实模型行为：

- 分组 INT8（仍为 8 位）与原逐行 INT8 对照；
- 同样约 30% 总稀疏率的幅度剪枝与 SparseGPT 式二阶补偿剪枝；
- 普通自然文本触发、每位 3 个输入的软读出，以及完整词表 KL 效用约束；
- 固定前缀、校准 NLL 异常输入过滤、统一大小写/空格改写三种过滤/改写；
- 不含秘密触发样本的软概率蒸馏、硬标签蒸馏；
- 无水印模型、原指纹、新指纹的配对指标，数千 token 的 WikiText-2 效用测试。

方法依据、数据划分及限制见 [ROBUSTNESS_DESIGN.md](docs/ROBUSTNESS_DESIGN.md)。
已完成的真实对照、原始日志和失败反例见 [2026-10-25 实验结果](results/robustness_20261025/README.md)：新版 5 用户均能定位，PPL 增幅 1.54%–4.13%；分组 INT8 后代表用户增幅 1.96%。30% 剪枝仍未满足全部效用标准；短程软蒸馏和保持效用的硬蒸馏仍能移除指纹。
**这不是“全部攻击已解决”的声明**：每种攻击的实际定位、PPL 和失败状态均写入新报告。
10% PPL / 90% 下一词元一致率是本 Demo 自设的工程效用门槛，**不是论文规定或行业统一标准**；本次保持门槛不变，也不将增加查询次数隐藏起来。
新版输入是 768 次模型输入，原版为 256 次；严格单输入盒的问题不因此消失。

### Windows 详细运行步骤

在现有项目目录的 PowerShell 中运行（保持 Python 3.12，不重建已经可用的环境）：

```powershell
cd "C:\Users\14509\Desktop\TVMC_demo"
git status
git pull --ff-only
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: -r requirements-research.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe run_robustness.py --device cpu
```

`git status` 有自己的修改时先保留修改，不要使用强制 reset。若 pip 再次遇到 SSL 网络中断，可将安装一行替换为：

```powershell
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: -r requirements-research.txt --index-url https://pypi.tuna.tsinghua.edu.cn/simple --timeout 120 --retries 5
```

新增依赖只有 parquet 读取所需的 `pyarrow==19.0.1` 及已有基础依赖；确认官方提供 Windows x64 / Python 3.12 的预编译 wheel。不要安装 Rust 或关闭 TLS 校验。

首次研究运行除模型外，还下载固定版本 WikiText-2 的 train、validation、test 三个 parquet 文件。原模型缓存会复用。
这个实验比原 Demo 更重：10 个模型配置、多种全模型压缩、两类蒸馏及数千 token 评估，CPU 可能需要数十分钟，建议预留 16 GB 内存。各阶段会持续写入日志。

模型已经在本地时，可以避免再次找模型下载源：

```powershell
.\.venv\Scripts\python.exe run_robustness.py --model-path .\models\SmolLM2-135M --device cpu
```

只有模型和 WikiText 都已缓存后，才使用完全离线模式：

```powershell
.\.venv\Scripts\python.exe run_robustness.py --device cpu --offline
```

输出位于 `outputs\robustness\<运行ID>\`：

```text
metrics.json                  所有配对指标、数据文档哈希、源码哈希、改进门槛和失败
DEMO_REPORT.md                可读结果表与边界说明
demo_output.log               实际运行日志和异常
benchmark_codebook.json       公开的研究码本，不是真实托管秘密
original_user*.pt             旧版适配器
natural_redundant_user*.pt    新版适配器
```

过滤结果中的 `accepted_requests` / `rejected_requests` 记录放行和拒绝数量，`observed_code_bits` / `correct_code_bits` 记录可观测和正确码位数量。被拒绝输入不进入指纹响应推理，而以擦除交给解码器；NLL 过滤器自身的输入评分不作为指纹响应使用。
报告同时显示压缩后的 PPL 绝对值及增幅。WikiText 指标与原 12 句语料指标分别比较，不能跨语料把百分比变化当作算法收益。

查看最新结果：

```powershell
$research = Get-ChildItem .\outputs\robustness -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
Get-Content (Join-Path $research.FullName "DEMO_REPORT.md") -Encoding UTF8
```

另开窗口可追踪运行日志：

```powershell
Get-Content (Join-Path $research.FullName "demo_output.log") -Tail 20 -Wait
```

补做更严格的蒸馏对照（先完成上面的研究主实验；仍使用同一批保存适配器，不重新训练防御）：

```powershell
.\.venv\Scripts\python.exe run_distillation_controls.py --run-dir $research.FullName --offline
```

若使用独立模型目录，可追加 `--model-path .\models\SmolLM2-135M`。附加结果在该研究运行目录的 `distillation_controls\<运行ID>\`。
包括未学习教师的干净学生、12 步软蒸馏、带效用保持约束的硬标签蒸馏。附加脚本退出码 0 表示测量完成，**不表示水印通过攻击**；逐项看 `fingerprint_survives` 和 `utility_pass`。

补做同等 30% 剪枝后的恢复对照（CPU，额外耗时和内存）：

```powershell
.\.venv\Scripts\python.exe run_pruning_recovery.py --run-dir $research.FullName --offline
```

幅度剪枝与二阶补偿剪枝均使用 50 步、相同学习率的干净教师 KL 恢复，固定零权重掩码，不向恢复过程提供秘密触发器。结果保存在 `pruning_recovery\<运行ID>\metrics.json`；退出 0 同样只代表测量完成。可追加相同的 `--model-path` 参数。

退出码 `0` 仅表示报告中列出的 **improvement_gates（改进门槛）通过**；还须检查 `all_attacks_solved` 和每行结果。
退出码 `2` 表示存在未通过的改进门槛，`3` 表示执行异常。不能把门槛通过改写成所有攻击均已解决。
此入口的研究码本与原身份注册表分开；不要将研究输出目录传给旧版 `serve.py` 或 `verify_evidence.py`。
原端到端托管/API 演示仍按下文 `run_demo.py` 的步骤执行。

### 为完全离线 Windows 机器准备研究依赖和数据

在相同 Python 3.12 / Windows x64 的联网准备机上，执行原文离线安装节的 PyTorch wheel 下载，再执行：

```powershell
.\.venv\Scripts\python.exe -m pip download --only-binary=:all: --dest .\wheelhouse -r requirements-research.txt
.\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('HuggingFaceTB/SmolLM2-135M', revision='93efa2f097d58c2a74874c7e644dbc9b0cee75a2', cache_dir='.cache/huggingface', allow_patterns=['*.json','model.safetensors','merges.txt','vocab.json'])"
.\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('Salesforce/wikitext', repo_type='dataset', revision='b08601e04326c79dfdd32d625aee71d232d685c3', cache_dir='.cache/huggingface', allow_patterns=['wikitext-2-raw-v1/*.parquet'])"
```

将源码、`wheelhouse` 和整个 `.cache\huggingface` 复制到离线 Windows 项目目录。若缓存使用符号链接，拷贝工具必须将链接目标的实际文件一起保留，不能只复制链接。不要拷贝 `.venv`。
离线机创建 Python 3.12 虚拟环境后执行：

```powershell
.\.venv\Scripts\python.exe -m pip install --no-index --find-links .\wheelhouse torch==2.5.1 -r requirements-research.txt
.\.venv\Scripts\python.exe run_robustness.py --device cpu --offline
```

代码和报告上传 GitHub；模型、数据缓存、wheelhouse 与生成适配器仍在本地准备，不上传大型文件或私有材料。

## Windows：从终端安装并运行

建议 Windows 10/11 x64、**Python 3.11 或 3.12 x64**，至少 8 GB 内存（建议 16 GB），预留 6 GB 磁盘。CPU 可运行，实际耗时取决于处理器；首次需要联网下载约 270 MB 模型权重及 PyTorch。此交付的运行验证使用 Linux CPU，Windows 命令按跨平台接口提供，未声称已在 Windows 实测。

### 1. 获取项目

以下命令在 **Windows PowerShell** 中逐段执行。先检查工具：

```powershell
py -3.12 --version
git --version
```

没有 Python 或 Git 时，可从 python.org 和 git-scm.com 安装，或在支持 winget 的 Windows 中执行：

```powershell
winget install --exact --id Python.Python.3.12
winget install --exact --id Git.Git
```

安装后关闭并重新打开 PowerShell。Python 安装器应安装 Python Launcher；只有 Python 3.11 时，把下文 `py -3.12` 改成 `py -3.11`。推荐固定使用同一 Python 小版本。

克隆 GitHub 仓库到桌面（自动适配当前 Windows 用户，不写死用户名）：

```powershell
$desktop = [Environment]::GetFolderPath("Desktop")
Set-Location $desktop
git clone https://github.com/ZhouJay798/TVMC_demo.git tvmc_demo
Set-Location .\tvmc_demo
```

如果已经存在 `tvmc_demo`，不要覆盖；进入已有克隆，先 `git status` 检查本地修改，再执行 `git pull --ff-only`。也可在 GitHub 仓库页面选择 **Code → Download ZIP**，解压后进入含 `run_demo.py`、`requirements.txt` 的目录。私有仓库需要先用自己的 GitHub 账号登录 Git Credential Manager 或浏览器，不能使用云环境中的代理凭据。

### 仓库内有哪些，哪些必须在 Windows 本地准备

| 内容 | GitHub 中是否包含 | 本地操作 |
|---|---|---|
| Python 源码、测试、README、原文 | 是 | 克隆或下载 ZIP |
| 已运行的日志、指标、报告、公开验证材料 | 是，位于 `results/final` | 可直接查看，不等于本机已运行 |
| Python、Git、PyTorch 和其他依赖 | 否 | 按下文安装，或使用离线 wheel 包 |
| SmolLM2-135M 权重与 tokenizer | 否 | 第一次运行下载，或手动下载完整模型目录 |
| `.venv`、模型下载缓存 | 否 | 在当前 Windows 机器创建，不能复制 Linux 虚拟环境 |
| 用户 LoRA 适配器、解密模型、加密资产 | 否，属于 `outputs` | 运行完整实验生成，之后才能启动模型 API |
| 持续运行的 API 进程 | 否 | 本地运行 `serve.py`；关闭终端或重启电脑后需重新启动 |

**下载状态说明：** 云端最初访问 Hugging Face 返回 403，允许所需域名后已成功下载并完成真实模型实验；本次交付没有仍未下载成功的必需模型文件。GitHub 不上传模型、大型缓存和运行产物，因此你在 Windows 上仍需要完成下面的下载或离线拷贝步骤。所有网络安装命令失败时应先处理错误，不要把缺依赖误认为实验未达标。

### 2. 安装（不用激活虚拟环境）

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

这些命令直接调用虚拟环境里的 Python，**不需要修改 PowerShell 执行策略**。不要使用其他窗口中的全局 `pip` 混装。NVIDIA GPU 用户可以按 PyTorch 官方安装说明安装适配自己驱动的 PyTorch 2.5.1，然后使用 `--device cuda`；默认 CPU 安装无须 CUDA。

### 3. 一条命令运行完整实验

```powershell
.\.venv\Scripts\python.exe run_demo.py --device cpu
```

默认参数：5 用户、2 人合谋、Shamir (2,5)、256 码位、每用户 800 个 LoRA 优化步、每类训练攻击 12 步、2000 个独立码本误告实验。日志每 200 个嵌入优化步输出一次。默认缓存存放在项目 `.cache\huggingface`；重新执行可复用已下载模型。

运行结果保存在 `outputs\<UTC时间-运行ID>\`，不会覆盖旧实验：

```text
metrics.json           全部指标、版本、固定阈值、参数、分母、退出码
DEMO_REPORT.md         设计、结果表、限制、复现命令
demo_output.log        真实 stdout/stderr 与异常 traceback
user_1_lora.pt ...     5 个训练后的实际输出头 LoRA 适配器（模型阶段成功后）
public_evidence.json   首个会话的裁决证据
public_context.json    公开份额承诺、采集者公钥、注册表快照与 NIZK
model.aesgcm           真实模型权重的认证加密副本
decrypted_model\      托管恢复后实际用于推理的权重与 tokenizer
```

模型阶段被网络阻断时，仍生成前三个文件，并明确标记未运行。每次完整运行可能额外占用约 600 MB；不再需要的旧 `outputs` 子目录可自行删除。

查看最近结果：

```powershell
$run = Get-ChildItem .\outputs -Directory | Where-Object { Test-Path (Join-Path $_.FullName "metrics.json") } | Sort-Object LastWriteTime -Descending | Select-Object -First 1
Get-Content (Join-Path $run.FullName "DEMO_REPORT.md") -Encoding UTF8
Get-Content (Join-Path $run.FullName "metrics.json") -Encoding UTF8
```

退出码：`0` = 所选范围通过；`2` = 实验运行完成但目标未全部满足（包括严格单输入盒证据不足）；`3` = 执行失败或外部阻断。用 `$LASTEXITCODE` 查看。报告保留失败，不能通过改阈值把结果改成通过。

### 4. 测试与只运行密码学部分

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe run_demo.py --core-only
```

`--core-only` **不运行模型**，不能替代完整实验；pytest 中的小型随机 Transformer 仅测试实现接口，不冒充 SmolLM2 效果数据。

可独立验证公开证据（不需要模型、私有份额或私有验证密钥）：

```powershell
.\.venv\Scripts\python.exe verify_evidence.py --run-dir .\results\final
```

`public_context.json` 是可信初始化材料，必须从可信拥有者取得；若攻击者可以同时替换证据和全部公钥/注册表，验证没有安全意义。验证器检查密钥证明、部署证据，以及二者的会话/周期/参与集合绑定。演示中的时间为显式逻辑时间。

### 5. 离线部署或下载受限

第一次下载需要访问 `huggingface.co`、`*.huggingface.co` 和 `*.hf.co`（权重 CDN 可能使用这些域）。保持 TLS 校验；不要使用 `verify=False`。本模型公开，无须在代码里填写令牌。

可在能联网的 Windows 机器上安装上述依赖，然后下载完整模型目录：

```powershell
.\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('HuggingFaceTB/SmolLM2-135M', revision='93efa2f097d58c2a74874c7e644dbc9b0cee75a2', local_dir='models/SmolLM2-135M', allow_patterns=['*.json','model.safetensors','merges.txt','vocab.json'])"
.\.venv\Scripts\python.exe run_demo.py --model-path .\models\SmolLM2-135M --offline --device cpu
```

可把完整 `models\SmolLM2-135M` 目录复制到离线机器。离线机器仍需预装依赖；不要直接跨操作系统复制 `.venv`。本地目录必须含 `model.safetensors`、`config.json`、`tokenizer.json` 及 tokenizer 配置。缺文件会明确报错，不自动换成假模型。

已缓存模型时也可运行：

```powershell
.\.venv\Scripts\python.exe run_demo.py --offline --device cpu
```

需要锁定权重版本时，在 `metrics.json` 中找到 `resolved_revision`，使用 `--revision <完整commit SHA>` 重跑。真实资产 SHA-256 同时写入报告数据。

### 5.1 下载中断、超时或 403 的具体处理

1. 保留已下载的缓存，再次执行原下载命令；已完成的文件会复用，不必删除整个缓存。
2. 检查是否能访问 Hugging Face 以及权重跳转的 CDN。仅能打开网页不代表权重下载域名可达。受限网络由管理员允许 `huggingface.co`、`*.huggingface.co`、`*.hf.co`。Python 包还需要 `pypi.org`、`files.pythonhosted.org`、`download.pytorch.org` 和 `download-r2.pytorch.org`。
3. 慢速网络可在当前 PowerShell 会话增加等待时间，然后重试：

```powershell
$env:HF_HUB_DOWNLOAD_TIMEOUT = "120"
$env:HF_HUB_ETAG_TIMEOUT = "60"
.\.venv\Scripts\python.exe -m pip install --timeout 120 --retries 5 -r requirements.txt
```

4. 如果你有已经配置好的本机 HTTP 代理，按实际端口设置；下面的 7890 只是示例，没有该代理时不要执行：

```powershell
$env:HTTPS_PROXY = "http://127.0.0.1:7890"
$env:HTTP_PROXY = "http://127.0.0.1:7890"
```

不再使用代理时执行 `Remove-Item Env:HTTPS_PROXY, Env:HTTP_PROXY -ErrorAction SilentlyContinue`。不要把账号、密码或令牌写进 README、Git 仓库或截图。企业证书问题应安装/配置受信任 CA，不能通过关闭 TLS 校验解决。

5. 仍无法下载时，在另一台可联网的 **Windows x64、相同 Python 版本**机器准备模型和依赖，按下一节复制。不要将浏览器保存的 HTML 错误页面重命名为模型文件。

下载模型后，在模型所在的项目根目录检查真实权重校验和：

```powershell
$actual = (Get-FileHash .\models\SmolLM2-135M\model.safetensors -Algorithm SHA256).Hash.ToLowerInvariant()
$expected = "80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1"
if ($actual -ne $expected) { throw "权重校验失败：请重新下载上述固定版本，不要继续运行。" }
Get-ChildItem .\models\SmolLM2-135M
```

该校验和仅对应上文固定 commit 的权重；不能拿它验证任意其他版本。Windows 下 Hugging Face 的 symlink 警告通常表示缓存采用复制文件方式，会多占磁盘；不必为此关闭安全检查或强制管理员运行。

### 5.2 完全离线 Windows 安装（包括 Python 包）

**联网准备机**须使用与离线机相同的 Windows x64 / Python 3.12（或两台都用 3.11）。先按前文创建 `.venv`，然后在项目根目录下载 wheel：

```powershell
New-Item -ItemType Directory -Force .\wheelhouse | Out-Null
.\.venv\Scripts\python.exe -m pip download --only-binary=:all: --dest .\wheelhouse torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip download --only-binary=:all: --dest .\wheelhouse -r requirements.txt
```

还要执行第 5 节的 `snapshot_download` 命令，准备 `models\SmolLM2-135M`。将**项目源码、wheelhouse、整个模型目录**一起拷贝到离线机；不要复制 `.venv`。如果离线机尚无 Python，还需提前下载官方 Python 3.12 x64 的完整离线安装器；Git 不是必需的，可直接拷贝源码目录。

**离线机**在项目根目录执行：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --no-index --find-links .\wheelhouse torch==2.5.1 -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe run_demo.py --model-path .\models\SmolLM2-135M --offline --device cpu
$LASTEXITCODE
```

`--no-index` 保证安装时不访问包索引。出现 `No matching distribution found` 时，检查两台机器的 Python 小版本、x64 架构，以及 wheel 是否全部下载成功；回到联网准备机补齐，不能用 Linux 的 wheel 代替 Windows wheel。离线 `--model-path` 指向本地模型，不需要登录 Hugging Face。

### 6. 常见问题

- **403 / ProxyError**：检查环境的网络允许域和公司代理，不是 LoRA 失败。云环境需要在环境设置保存域名变更。可改用上述本地模型目录。
- **内存不足**：关闭其他大型程序，使用 `--threads 2` 限制 CPU 线程；仍不足时先运行 `--core-only`。减少 `--length` 会改变研究参数，报告会记录，不能当作默认参数通过。
- **CPU 训练较慢**：可用 `--steps 200 --attack-steps 4` 做工程试运行，但这改变训练预算，须对照报告。默认不悄悄减少实验。
- **CUDA 不可用**：CPU wheel 没有 CUDA 支持；直接用 `--device cpu`。
- **路径含空格**：在 PowerShell 中给路径加双引号。
- **依赖冲突**：使用独立 `.venv`，不装与本任务无关的 torchvision/torchaudio。

## 本地 API 部署（完整模型实验生成适配器后）

在第一个 PowerShell 窗口选择一次成功生成权重和适配器的运行目录，启动服务：

```powershell
$run = Get-ChildItem .\outputs -Directory | Where-Object { (Test-Path (Join-Path $_.FullName "user_1_lora.pt")) -and (Test-Path (Join-Path $_.FullName "decrypted_model\model.safetensors")) } | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $run) { throw "没有可部署模型：请先运行完整 run_demo.py；--core-only 不会生成模型。" }
.\.venv\Scripts\python.exe serve.py --run-dir $run.FullName --device cpu --port 8080
```

第二个 PowerShell 窗口检查并发起一次请求：

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8080/health"
$body = @{ protocol = "batch"; prompts = @("The capital of France is", "One plus one is") } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8080/query" -ContentType "application/json" -Body $body
```

响应中的 `bits` 是模型对 0/1 两个词元的概率比较结果，不是自然语言回答。上述普通提示只用于连通性验证；正式指纹提示由主脚本生成。一次有效 `/query` 后再次请求返回 HTTP 409。健康检查不消耗额度；要演示一个新盒，在第一个窗口 Ctrl+C 后重新启动。`stream` 在进程内分块推理，HTTP 响应汇总返回，不宣称 SSE 流式传输。服务只绑定本机，不提供公网托管或认证系统。

如果端口 8080 被其他程序占用，启动时改用 `--port 8081`，并把请求地址中的端口一起改为 8081。不要随意终止不属于本 Demo 的进程。API 窗口需要持续运行；这不是 Windows 系统服务，关闭窗口后服务停止。重新启动新盒后再做新的请求，HTTP 409 是设计行为，不是服务器崩溃。

## 当前尚未达到的目标

完整代码与下载成功，不代表研究目标全部通过。本次测量中用户 1–4 仍有 1–2 位指纹读错；严格 single 盒无法完整归因；过滤与蒸馏可移除指纹；量化、剪枝后的效用超过退化阈值。这些是**实测未达标项**，不是 Windows 少下载了某个组件。正常的完整运行因此可能返回退出码 2；查看 `DEMO_REPORT.md`，不要通过反复安装依赖或降低标准将它改写为“全部通过”。

## Linux / macOS

Linux CPU：

```bash
cd /workspace/TVMC_demo
python3 -m venv .venv
.venv/bin/python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest -q
.venv/bin/python run_demo.py --device cpu
```

macOS 请从默认 PyPI 安装 `torch==2.5.1`，不使用 Linux/Windows CPU wheel 索引；本项目未实测 macOS。

## 实验设计与准确边界

| 层/实验 | 实现 | 判断标准或边界 |
|---|---|---|
| 密钥托管 | RFC3526 2048-bit 群，Feldman + 一次 Shamir 多项式，5 份 | 10 个两份组合恢复；单份、重复、篡改份额拒绝 |
| NIZK | AND-Schnorr + Fiat–Shamir，绑定身份、会话、周期和资产 | 证明知道份额和身份秘密，不是一般电路 SNARK；Python 算术非恒时 |
| 资产加密 | HKDF-SHA256 → AES-256-GCM | 完整 safetensors 加密/恢复，AAD 绑定资产哈希；密文篡改拒绝 |
| Tardos | 截断 arcsine 偏置，对称加权得分 | 固定阈值；每个两人组合测试 interleave / AND / OR，不等于穷举所有策略 |
| LoRA | 秩 8 输出头更新，仅 0/1 两个输出行可训练 | 冻结主干允许精确缓存训练特征；最终黑盒读出仍跑完整模型 |
| 指纹读出 | 触发文本上的 0/1 next-token logits 比较 | 需要概率接口，不承诺普通聊天 API 文本会输出 0/1 |
| 效用 | 12 条独立于训练集的固定留出文本 | PPL 相对增加 ≤10%，下一词元一致率 ≥90%；不是完整通用能力测评 |
| 身份桥 | seed / cm_w / 可信注册表快照 | 合法接受；错 seed、错承诺、缺记录、过期拒绝 |
| 片段栽赃 | 完整覆盖检查 + 可信采集者 Ed25519 签名 | 拒绝伪造/片段证据；签名不证明模型来源，不能抵挡恶意采集者或完整复制指纹 |
| batch | 一次请求含 256 个触发输入 | 真正一次请求，但不是一个输入 |
| stream | 一次请求，分块返回多组符号 | 不重置盒，不把多个独立请求伪装成一次 |
| single | 严格一个输入，只返回一个符号 | 证据不足，必须拒绝归因；不宣称可完整追踪 |
| 微调 | 干净语料上 LoRA 头继续训练 | 真实语言模型交叉熵优化 |
| 量化 | 主干矩阵 per-row INT8 舍入/反量化 | 真实参数变换，但计算仍用 float，不是 INT8 内核性能测试 |
| 剪枝 | 主干矩阵逐矩阵移除 30% 小权重 | token embedding 不剪枝，报告明确范围 |
| 模型混合 | 用户 1/2 的有效低秩权重更新按 50/50 混合 | 不直接平均 A/B 因子；定位任一合谋用户算存活 |
| 数据/输入过滤 | 拒绝明显的 `TVMC probe` 触发输入 | 用于暴露 OOD 触发的脆弱性，不声称触发器隐蔽 |
| 蒸馏 | 干净预训练 SmolLM2 + 新 LoRA 头，在教师概率上 KL 优化 | 同架构学生、头部蒸馏；不是从零训练完整小学生模型 |

六类攻击按顺序独立运行，每轮恢复同一个基准水印模型，避免把上一轮损伤混入下一轮。码层误告统计报告分子/分母、独立场景 Wilson 区间；模型层合谋误告另记，绝不拿码层模拟代替模型实测。

原文的阈值 t/t+1 表述存在冲突，这里按提示词的 (2,5) 实现。原文也没有给出“三种一次性协议”的具体定义，以上三种为本 Demo 的显式解释。会话指纹对应消费者/参与集合，密钥层证明只证明谁参与恢复，**不能据此裁定某个参与方泄露模型**。公开注册表和采集者是本地可信组件，不是分布式共识或远程证明；未实现 Refresh、TEE、链上合约。

## 文件结构

```text
run_demo.py             一条命令、日志、指标和 Markdown 报告
serve.py                本机 HTTP 一次性推理 API
verify_evidence.py      只使用公开材料的独立裁决
requirements.txt        固定直接依赖版本（PyTorch 单独安装）
tvmc/crypto.py          托管、NIZK、AEAD
tvmc/tardos.py          码字、打分、解码、合谋策略
tvmc/bridge.py          登记、身份桥、Judge、一次性盒
tvmc/model.py           真实 SmolLM2 LoRA、效用与参数攻击
tvmc/experiment.py      全部实验及三层归因
tvmc/data.py            本地固定训练/效用语料
tests/                 密码学、预算与离线模型接口测试
docs/                  用户提供的原文与提示词
```

当前实例的实测摘要见 `VALIDATION.md`；三项原始产出见 `results/`（仅保留本次测量日志/指标/报告，不提交模型权重或私有份额）。

## 蒸馏指纹新实验

新增 `run_distillation_research.py`：比较三种普通输出统计指纹候选，用验证集选择后在新的测试文本上评估，并保留原自然触发指纹的配对对照。
[实测结果与失败边界](results/distillation_20261026/DEMO_REPORT.md)。短程软概率及保持效用的硬标签蒸馏已改善；实际生成文本蒸馏仍可移除指纹，不是通用抗蒸馏方案。新检测需要完整词表概率和干净基座，不能直接替代旧 Tardos/Judge。

Windows PowerShell（沿用 Python 3.12 虚拟环境与研究依赖）：

```powershell
git pull --ff-only
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: -r requirements-research.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe run_distillation_research.py --offline
```

只有模型和 WikiText 缓存齐备时使用 `--offline`；独立模型目录可追加 `--model-path .\models\SmolLM2-135M`。CPU 运行可能需要数十分钟。结果在 `outputs\distillation_research\<运行ID>`，包括报告、CSV、指标和适配器。公开研究密钥种子仅用于复现，不能作为生产密钥。退出 0 仅表示报告中的重点门槛通过，仍需阅读生成文本蒸馏的失败结果。
