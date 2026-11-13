# TVMC v2 验证方式

当前测试：

```bash
.venv/bin/python -m pytest -q
```

测试覆盖 HPKE 官方向量、TDH2 胶囊绑定、Pedersen VSS、任意三人部分解密、少于门限和重复成员拒绝、身份和请求绑定、分块交付、加密码本证明、公开裁决、一次性探针、持久化日志、签名见证、实际 HTTPS 及证书固定、取证一致性、转发日志申辩和申辩后的加密评分重算。

真实模型验证：

```bash
.venv/bin/python run_experiment.py --output work/file_demo
.venv/bin/python run_experiment.py --output work/relay_demo --evidence relay
.venv/bin/python verify_experiment.py work/file_demo
.venv/bin/python verify_experiment.py work/relay_demo
```

不能用单元测试代替真实模型实验；不能把 `report.status == passed` 理解为所有攻击都防御成功。`source_candidates` 是来源线索，`final_attribution` 才包含申辩处理后的指认。

旧版测试与实验已保存在 Git 历史 `e858e3d`。新实验结果归档在 `results/reproduction_20261113/`，不混用旧 LoRA 模型实验数字。
