# 新流程已确认的实验结果

## 已通过

- 真实模型：SmolLM2-135M-Instruct（由固定 Q4_1 解量化），约 1.35 亿参数。
- 本次文件实验：5 名成员，3 人门限，3 份副本，64 位归因指纹、16 位筛查指纹，每位 3 个探针。
- U1、U2、U3 全部成功接收自己的模型；交付前后字节一致，指纹读错均为 0。
- 三份副本在本次效用集上的下一词一致率分别约 96.82%、97.03%、96.08%。
- 疑似 U1 文件仅命中 U1；分数 U1=80.4648、U2=-1.9424、U3=1.6504，原始门槛约 36.55，实际判定含定点舍入修正。
- 新进程无需私钥，重新加载实际模型并复核证据：verified=true。
- 当前测试：21 passed，1 个第三方依赖弃用警告。
- 文件实验含攻击对照耗时 640.7 秒；不代表纯密码操作耗时。

## 保留的失败结果

10% 幅度剪枝、INT8 数值转换和短程全参微调后，本次仍命中且通过效用门槛。30% 剪枝及通道置零虽然仍命中，但模型能力未达标。两副本平均、公开基座复位，以及部分公开基座截断能够逃过追踪；不能宣称抵抗全部合谋与去指纹攻击。

完整数据：[file_report.json](file_report.json)；攻击表：[attacks.csv](attacks.csv)；独立复核：[file_independent_verification.txt](file_independent_verification.txt)。

## API 状态

API 转发申辩：代码及组件测试已加入，但按用户要求提前结束本轮运行，未完成端到端验收，不计为通过。

## 复跑

```bash
.venv/bin/python -m pytest -q
.venv/bin/python run_experiment.py --output work/file_demo --bits 64 --attacks --threads 2
.venv/bin/python verify_experiment.py work/file_demo --threads 2
```

默认主入口使用 128 位；本次完整文件验收使用上述 64 位命令。正常流程可不加 --attacks 以减少耗时。

旧代码从当前版本清理，但仍保存在 Git 提交 e858e3d。未上传模型权重、成员私钥、数据密钥或完整工作目录。结果是流程演示，不是论文 2048 位目标配置或生产安全认证。
