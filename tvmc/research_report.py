"""Render measured paired outcomes without converting failures into passes."""
import json


def write_report(m,out):
    lines=['# TVMC 鲁棒性配对实验报告','',f"运行：`{m['run_id']}`",'',
           f"状态：**{m['status']}**；退出码：{m.get('exit_code','running')}",
           f"全部所测攻击均解决：**{m.get('all_attacks_solved',False)}**",'',
           '这是模型层配对对照；不是新公开身份归因证据。原密码学端到端 Demo 保持独立。',
           '通过改进门槛只说明所列比较有效，不等于所有研究目标成立。', '',
           '## 固定设置', '',
           'SmolLM2-135M；5 用户；256 码位；固定 Tardos 阈值；原触发 256 次模型输入，'
           '新版 3 倍冗余为 768 次。自然前缀错误可能相关，不声称独立重复带来的理论保证。',
           'Demo 自设的工程效用门槛为 PPL 相对增加 ≤10% 且下一词元一致率 ≥90%，不是论文指定阈值。测试文档、模型、源码的哈希见 metrics.json。',
           'WikiText 校准、蒸馏和触发文档相互分开。测试按固定 offset 取 64 个段落，'
           '不同段落可能来自同一文章，结果不代表全数据集或不同主题的独立泛化。', '',
           f"干净测试模型：{json.dumps(m.get('clean_test',{}),ensure_ascii=False)}",'',
           '## 嵌入结果','', '| 配置 | 用户 | 位准确率 | 定位 | PPL 增幅 | 一致率 | 效用通过 |',
           '|---|---:|---:|---|---:|---:|---|']
    for x in m['embeddings']:
        u=x['utility']
        lines.append(f"| {x['profile']} | {x['user']} | {x['bit_accuracy']:.2%} | {x['decoded_users']} | {u['relative_perplexity_increase']:.2%} | {u['next_token_agreement']:.2%} | {u['utility_pass']} |")
    lines+=['','## 压缩：相同位宽/相近实际稀疏率的配对比较','',
       '| 方法 | 真实零权重率 | 干净模型 PPL 增幅 | 新版定位 | 新版 PPL | 新版 PPL 增幅 | 新版一致率 | 新版效用通过 |',
       '|---|---:|---:|---|---:|---:|---:|---|']
    for method,x in m['compression'].items():
        n=x['natural_redundant']; u=n['utility']
        lines.append(f"| {method} | {x['parameters']['actual_zero_fraction']:.3%} | {x['clean_control']['relative_perplexity_increase']:.2%} | {n['decoded_users']} | {u['perplexity']:.3f} | {u['relative_perplexity_increase']:.2%} | {u['next_token_agreement']:.2%} | {u['utility_pass']} |")
    lines+=['','### 原 12 句语料：单独对照','',
            f"该语料的干净基座 PPL：{m.get('legacy_clean',{}).get('perplexity','未测')}。不能与上表不同语料的百分比直接比较。",'',
            '| 方法 | 新版 PPL | 相对该语料干净基座增幅 |','|---|---:|---:|']
    for method,x in m['compression'].items():
        u=x['natural_redundant'].get('legacy_utility')
        if u:
            lines.append(f"| {method} | {u['perplexity']:.3f} | {u['relative_perplexity_increase']:.2%} |")
    lines+=['','压缩行均有同一变换下的无水印对照、旧版指纹对照，以及原 12 句语料指标，详见 JSON。'
            'INT8 是权重舍入后浮点执行，不是 INT8 内核或压缩存储。'
            '剪枝恢复始终不使用秘密码字/触发器。所有模型攻击以用户 1 为代表；其他用户只完成干净嵌入评估。', '',
            '## 输入过滤与改写','',
            '| 配置/攻击 | 码位覆盖 | 触发拒绝率 | 正常请求拒绝率 | 定位 |', '|---|---:|---:|---:|---|']
    for name,x in m['filtering'].items():
        lines.append(f"| {name} | {x['coverage']:.2%} | {x['trigger_rejection_rate']:.2%} | {x['clean_rejection_rate']:.2%} | {x['decoded_users']} |")
    lines+=['','NLL 阈值来自校准集的 95% 分位，不按秘密触发标签调阈值。'
           '拒绝的输入不会进入指纹响应推理；输入评分仅供过滤器判断，不供解码器使用。'
           '拒绝率是正常服务可用性的代价；若 JSON 提供 rewrite_utility，它比较同样归一化的测试文本与干净模型，不能据此证明原任务语义完全保留。', '',
           '## 蒸馏','', '| 配置/攻击 | 位准确率 | 定位 | PPL 增幅 | 效用通过 |', '|---|---:|---|---:|---|']
    for name,x in m['distillation'].items():
        u=x['utility']
        lines.append(f"| {name} | {x['bit_accuracy']:.2%} | {x['decoded_users']} | {u['relative_perplexity_increase']:.2%} | {u['utility_pass']} |")
    lines+=['','学生使用相同干净预训练基座，仅训练新输出头 LoRA。软蒸馏使用精确完整词表 KL；'
            '硬蒸馏使用教师实际 argmax。两者均不接收秘密触发样本。'
            '这不是从零训练整个学生，也不能代表所有蒸馏算法。', '',
            '## 改进门槛','', '```json',json.dumps(m.get('improvement_gates',{}),indent=2),'```','',
            '## 复现（Windows PowerShell）','', '```powershell',
            '.\\.venv\\Scripts\\python.exe -m pip install --only-binary=:all: -r requirements-research.txt',
            '.\\.venv\\Scripts\\python.exe run_robustness.py --device cpu', '```','',
            '模型和数据已缓存时追加 `--offline`。当前 Python 与依赖版本见 metrics.json；提供 Windows 命令不代表所有平台均已实测。'
            '测试集不能再用于调参；改动方法后应另设未使用的评估数据。', '',
            '报错：'+m.get('error','无执行异常。未达标指标仍保留，不降低标准。')]
    (out/'DEMO_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
