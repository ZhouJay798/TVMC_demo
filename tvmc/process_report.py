"""Human-readable summary for the local multiprocess experiment."""


def write_process_report(path, metrics):
    fixture = metrics['model_scope'] == 'synthetic_fixture'
    label = lambda value: '通过' if value else '未通过'
    roles = metrics['processes']
    lines = [
        '# 模型资产分发与追踪：多进程实验', '',
        f"运行状态：{metrics['status']}。",
        ('使用随机初始化的小型 Transformer 与合成文本，属于流程联调；不是预训练模型实验。' if fixture
         else '使用调用时指定的本地模型和语料；请结合模型文件哈希和来源记录理解实验结果。'),
        f"基座权重 SHA-256：`{metrics['base_sha256']}`；每份副本训练 {metrics['training_steps_per_model']} 步。",
        '', '## 进程与操作', '',
        '| 角色 | 本次进程号 | 本次工作 |', '| --- | ---: | --- |',
        f"| 协调者 | {roles['coordinator']} | 模拟可信份额分配者和恢复服务，最后进行本地指纹读取 |",
        f"| 分发者 | {roles['issuer']} | 生成签名私钥；训练副本、加密和签名发行清单 |",
        f"| 登记管理员 | {roles['registrar']} | 生成签名私钥；验证准入与发行清单并分别签预登记、发行登记 |",
    ]
    for i in range(1, 6):
        action = '预登记；生成身份证明和RSA私钥，参与3/3恢复、独立验签并解密' if i <= 3 else '只参与身份预登记'
        lines.append(f"| 托管方 U{i} | {roles[f'U{i}']} | {action} |")
    lines += ['', '## 本次结果', '',
              '| 托管方 | 模型包哈希核对 | 该副本指纹码位匹配 | 指纹命中 |',
              '| --- | --- | --- | --- |']
    for row in metrics['recipients']:
        trace = row['trace']; i = row['recipient'] - 1
        lines.append(f"| U{row['recipient']} | {label(row['payload_sha256_matches'])} | "
                     f"{trace['correct_bits'][i]}/256 | {', '.join(trace['decoded_replicas']) or '无'} |")
    names = {
        'distinct_processes':'角色进程号均不同',
        'insufficient_shares_rejected':'不足三份的请求被拒绝',
        'duplicate_shares_rejected':'重复份额被拒绝',
        'cross_request_proofs_rejected':'其他请求的证明被拒绝',
        'modified_share_rejected':'篡改份额被拒绝',
        'all_three_recover_session_secret':'三份有效份额恢复会话秘密',
        'invalid_issuer_signature_rejected':'错误发行方签名被拒绝',
        'tampered_registry_rejected':'篡改管理员登记被拒绝',
        'all_six_wrong_recipient_packages_rejected':'六次错误接收人交付均被绑定检查拒绝',
        'all_owner_model_hashes_match':'三位托管方模型包哈希一致',
        'all_models_trace_to_their_copy':'三个模型分别命中自己的登记副本',
    }
    lines += ['', '## 检查结果', '']
    lines.extend(f"- {names.get(name, name)}：{label(value)}。" for name, value in metrics['checks'].items())
    lines += [
        '', '## 实验边界', '',
        '- 分发者、管理员和托管方的私钥在各自子进程生成，没有通过角色接口直接导出；不过它们属于同一操作系统用户，进程与文件并非对抗性安全隔离。',
        '- 协调者模拟可信份额分配者和恢复服务，生成并接触会话秘密，也接收所有提交的份额；这不是分布式密钥生成或门限解密。',
        '- 本次管理员和分发者公钥由本地协调者提供，不等于独立可信信任根；没有独立见证、可信时钟、持久防回滚或远程身份认证。',
        '- 三个托管方分别验签并解密，但留存的模型 ZIP 和本地审计码本是敏感文件，同一用户下的其他进程也可能访问。',
        '- 子进程退出后角色私钥和份额不持久化，不能仅凭加密外包重新恢复模型。回执使用分发者的签名密钥，不保证离线恢复都被记录。',
        '- 指纹读取由协调者执行本地模型推理，并非远程 HTTP 证据；使用训练时相同的 768 条探针，不能证明对未见输入或攻击的鲁棒性。',
        '- 文件路径用于进程间传递，但加密打包仍在内存构造完整外层包；大文件还受 AES-GCM 单次输入大小等限制。',
        '- 命中表示登记副本来源线索，不能单独证明现实中的泄漏行为人。', '',
        '原始指标、签名清单与敏感审计材料保存在本次运行目录。', '',
    ]
    path.write_text('\n'.join(lines), encoding='utf-8')
    path.chmod(0o600)
