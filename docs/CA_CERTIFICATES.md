# CA 证书与先验证后用钥

本版本使用 cryptography 的真实 X.509 证书及 CA 签名 CRL（撤销列表），不是用 JSON 自签名代替证书。新增 `tvmc/pki.py`。

## 三类证书

| 主体/用途 | 算法与约束 | 保存位置 |
|---|---|---|
| distributor / issuer | Ed25519，digitalSignature，非 CA | trust.json 的 pki.issuer_certificate |
| U1…U5 / identity | Ed25519，digitalSignature，非 CA | registry 的 identity_certificate |
| U1…U5 / receiving | X25519，keyAgreement，非 CA | registry 的 receiving_certificate |

使用 CA 签名的 SAN URI 绑定项目身份与用途，检查 subject CN 必须匹配期望身份；接收证书不能当签名证书，U2 证书不能当 U1 证书。根 CA 为 Ed25519，BasicConstraints CA=true/path_length=0；当前仅支持根直接签叶子，不支持中间 CA 链。

公钥原字段保留以便查阅，但每次业务使用都从已验证证书导出公钥并对照原字段；原字段被修改即拒绝。普通 dict 不能冒充带预置信任配置的 Registry。

## 初始化与存储

本地演示建立 CA，给程序预先指定的分发者和成员签发证书，并完成成员登记签名和接收密钥挑战。CA 私钥在内存内，仅用于本地演示，不导出到实验结果或 GitHub。初始化成员不代表现实身份核验；生产部署必须由独立 CA 核验身份、申请来源与私钥持有情况后签发。

新实验将证书、根证书副本与 CRL 导出到 trust.json，并将根证书公开副本写入 ca-root.pem。根副本的存在不代表验证者应该信任它；必须通过独立可信渠道预置信任根。

## 使用前检查

1. 用预置信任根验证叶子证书签名；检查根用途和有效期。
2. 检查主体、项目用途、算法、KeyUsage、有效期、非 CA 约束及未知关键扩展。
3. 验证 CRL 的 CA 签名、签发者、更新时间/下次更新时间和最低版本号；检查证书序列号未撤销。
4. 从证书提取公钥，对比登记公钥；只使用验证后的结果。

这些检查用于登记、HPKE 封装前、清单/请求/批准验签、API 采集证据、追踪得分批准、独立 Judge、见证检查点和反驳日志验签。两套 DKG 公钥仍是原门限协议生成，本次没有为它们新增证书，也没有以 CA 代替 DKG 正确性证明。

## 独立验收

新结果不再默认信任结果目录自己带来的根：

```bash
.venv/bin/python verify_experiment.py work/file_demo --ca-root /trusted/tvmc-root.pem
```

也可以沿用事先独立保存的 `--trust-sha256`，该哈希将证书、根副本和整个信任文件一起固定。没有两者之一时拒绝；不提供自动降级。旧的无证书结果请使用其历史版本验证程序。

`--ca-root work/file_demo/ca-root.pem` 只有在你已经通过可信渠道确认该文件时才有意义；不要相信陌生结果包同时提供的根和叶子证书。

## 撤销与回滚

本地测试接口为 `new_crl = custody.ca.revoke(certificate_pem)`，然后 `custody.registry.trust.update_crl(new_crl)`。更新只接受 CA 签名且当前有效的 CRL，并提高内存中的最低 CRL 序号。即使新 CRL 撤销了分发者，也会安装并使后续业务拒绝，不能退回旧 CRL 继续运行。

证书默认有效 30 天，CRL 24 小时。CRL 到期后拒绝服务，需要管理员更新。没有 OCSP/自动下载/续期服务。独立验证可用 `--min-crl-number` 指定在结果目录外保留的最低 CRL 序号，防止回滚到旧但尚未过期的 CRL。只有 CA 根并不能阻止有效期内的旧 CRL 回滚；生产系统必须持久保存可信撤销状态和密钥版本。

## 尚未覆盖

五个成员、分发者和 CA 仍在同一实验进程。CA 私钥与根配置如果随进程一起失陷，软件内校验无法保证安全。无独立现实身份核验、HSM、独立 CA 服务、密钥轮换及完整历史时间戳验证。验证采用当前时间和当前 CRL，已过期/撤销证书签署的历史记录不会自动被视为长期有效。到期和撤销是失败关闭，不会静默退回裸公钥。
