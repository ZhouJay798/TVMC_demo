"""Human-readable report of one local delivery and tracing experiment."""
import json


def write_workflow_report(path, metrics):
    """Write results without overstating what local fixtures establish."""
    fixture = metrics['model_scope'].startswith('synthetic')
    scope = metrics.get('corpus_scope', '未记录')
    corpus_scope = {'synthetic fixture':'合成文本',
                    'disjoint windows of one public plaintext corpus':'单一公开文本集合的不重叠窗口',
                    'cached WikiText-2':'本地 WikiText-2 缓存'}.get(scope,scope)
    yes_no = lambda value: '通过' if value else '未通过'
    lines = [
        '# 大模型逐人分发与追踪：本次实验记录',
        '',
        f"运行状态：{metrics['status']}。",
        ('本次使用本地随机初始化的小型 Transformer 和人工合成文本，目的是检查端到端流程；'
         '它不是 SmolLM2 或 WikiText-2 的实验结果。' if fixture else
         f"本次使用所指定的本地模型；语料范围：{corpus_scope}。模型权重哈希如下。"
         '使用其他本地模型时，必须自行核对模型来源及版本。'),
        f"基座权重 SHA-256：`{metrics['base_sha256']}`。",
        '',
        '## 运行经过',
        '',
        '1. 在本地模拟五位托管方注册，选择 U1、U2、U3，验证会话申请和个人公钥绑定；'
        '生成本次 3/3 份额及其证明。真实身份和跨机构独立运行不在本实验范围。',
        '2. 为三个接收人生成不同的 256 位 Tardos 码，逐一训练模型头部；'
        + ('训练前记录登记方签名的指纹承诺和解码参数。' if metrics['signed_registry'] != 'not_run' else '本次未运行签名登记。'),
        '3. 将每份模型包用独立 AES-256-GCM 数据密钥加密，并以接收人 RSA 公钥封装对应数据密钥；'
        '发行方对清单签名，将三个密文和密钥封装装入外层会话加密包。',
        '4. 三份有效份额通过验证后恢复会话密钥，验证发行方签名'
        + ('及登记和检查点' if metrics['signed_registry'] != 'not_run' else '')
        + '；各接收人使用自己的私钥解封数据密钥，解密并核对模型包哈希。',
        '5. 将每个解密后的模型载入本机 HTTP 服务，发送一次包含 768 个探针的请求，'
        '汇总成 256 位观测并与登记码比较；另测单输入、空白模型、转交及参数混合。',
        '',
        '## 逐人结果',
        '',
        '| 接收人 | 本人解密及哈希 | 其他人私钥拒绝 | 码位匹配 | 批量指纹命中 |',
        '| --- | --- | --- | --- | --- |',
    ]
    for i, row in enumerate(metrics['rows']):
        hits = '、'.join(row['batch']['decoded_replicas']) or '无'
        matched = row['batch']['correct_bits'][i]
        lines.append(f"| U{row['recipient']} | {yes_no(row['owner_decrypt_hash_matches'])} | "
                     f"{yes_no(all(row['cross_key_rejected'].values()))} | {matched}/256 | {hits} |")
    lines += [
        '',
        f"空白基座模型命中：{metrics['negative_control']['decoded_replicas'] or '无'}。",
        '三份密文各用两个其他接收人的私钥尝试解封；具体结果见 `model_metrics.json`。',
        '各模型的 768 探针批量请求和单输入请求首次返回 HTTP 200，重复请求应返回 HTTP 409；'
        '具体状态见对应 HTTP 记录。单个输入不足以重建 256 位指纹。',
        f"U1 明文副本由模拟 U2 运行时的命中：{metrics['transfer_control']['decoded_replicas'] or '无'}。"
        '它指向登记副本来源，不证明谁实际实施了泄漏。',
        '',
        '## 实验检查及对照',
        '',
    ]
    if metrics['delivery_format'] == 'merged_model' and all('merged_reload_comparison' in row for row in metrics['rows']):
        lines.append('合并前后逐项比较浮点分数（绝对容差 0.0001、相对容差 0.00001），'
                     '并要求汇总得到的 256 位观测完全一致；不声称分数逐位完全相同。')
        lines.append('')
        for row in metrics['rows']:
            comparison=row['merged_reload_comparison']
            lines.append(f"- {row['replica']}：最大分数差 {comparison['max_abs_score_difference']:.8f}，"
                         f"汇总后变化 {comparison['different_aggregated_bits']} 位。")
        lines.append('')
    for name, passed in metrics['workflow_checks'].items():
        lines.append(f'- `{name}`：{yes_no(passed)}。')
    lines += [
        '',
        '密码学小载荷单独随机重复五轮；完整三模型包和每个模型训练只运行一轮。'
        '指纹参数混合对照只覆盖本次三组 50/50 模型头部参数混合：',
        '',
    ]
    for item in metrics['real_model_mixtures']:
        lines.append(f"- {' + '.join(item['pair'])}：命中 {', '.join(item['decoded']) or '无'}。")
    c1_path=path.parent/'c1_offline_control.json'
    if c1_path.is_file():
        c1=json.loads(c1_path.read_text(encoding='utf-8'))
        trials=c1['trials']
        recovered=sum(bool(row['offline_key_matches'] and all(row['own_payload_decrypts']) and
                           not row['reconstruction_receipt_emitted']) for row in trials)
        lines += ['', f'C1 独立离线控制：{recovered}/{len(trials)} 次小载荷试验中，持有全部份额者绕过'
                  '恢复服务插值并使用各自私钥解密，且未产生回执。它不代表对真实模型包重复了这些试验。']
    if metrics['signed_registry'] != 'not_run':
        lines += [
            '',
            '登记检查包括发行方清单和登记记录篡改、旧快照、错误挑战、过期检查点及错误接收人绑定。'
            '这些对照全部在本地模拟；见 `registry_checks.json`。',
        ]
    lines += [
        '',
        '## 结果边界',
        '',
        '- 指纹提示、登记信息及演示种子不构成生产级秘密；指纹可能被复制、模仿或移除。'
        '即使交付合并权重，公开基座差分、微调和蒸馏等风险仍需单独评估。',
        '- 登记方和见证方均由同一本地进程模拟；没有独立运营的见证方、持久防回滚、可信时间、'
        '真实身份认证或远程可信执行证明。打包运行目录中的公钥不自动构成独立信任根。',
        '- 指纹命中是副本来源线索，不能单独证明现实中的泄漏行为人；合谋混合也不保证全部来源均被检出。',
        '- 本次追踪使用与指纹训练相同的 768 条探针；每三条来自同一文本窗口，不能视为相互独立的样本，'
        '也不能证明对训练未见触发输入的泛化能力。',
        '- 本地代码、探针、码本和生成时的明文都必须妥善保护。签名与加密不能防止已获授权者转交明文。',
        '- 回执是本地服务的签名声明，不证明真实参与人身份，也不能阻止持有份额者离线插值。',
        '- 私钥、份额和会话密钥只在本次进程内存在，不会持久保存；进程结束后，不能仅凭保留的外层密文再次恢复模型。'
        '三个模型明文只在本机临时目录加载，不构成真实跨机构交付。',
        '- 复现实验需要准备并校验所选模型和语料；原有 C1/C2/C3 脚本依赖单独适配器，'
        '不能直接用于合并模型交付模式。',
        '',
        '详见 `crypto_metrics.json`、`model_metrics.json`、签名登记/清单和各 HTTP 原始响应。'
        '演示运行目录含敏感元数据，请勿公开上传。',
        '',
    ]
    path.write_text('\n'.join(lines), encoding='utf-8')
