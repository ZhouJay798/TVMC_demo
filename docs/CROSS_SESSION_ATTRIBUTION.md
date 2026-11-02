# 跨会话归因：会话级线索

目的：把已通过上游解码、映射至可信登记快照的会话标识，转换为可核对的参与集合线索。`tvmc.cross_session.session_leads` 不执行签名验证、会话登记真实性验证、模型来源证明或密钥层裁决；调用者必须先完成相应检查或明确声明信任假设。

1. 对整个登记快照验证会话ID唯一、周期非负、参与方非空且无重复；拒绝未知和重复的被指认会话，不返回部分集合。
2. 对每个命中会话输出会话ID、登记周期及登记参与方，输出全部命中会话的交集与并集。
3. 无命中：交集为 `null`（未定义），并集为空；单会话：保留完整登记名单，不收窄为个人。
4. 多会话交集非空：`intersection_heuristic`，将交集作为启发式线索，同时保留并集与各会话名单。交集为空：`union_fallback`，返回并集，仍为会话级线索。
5. 所有命中均标注“会话级线索”，`individual_attribution` 恒为 false。交集为单人也不能认定泄漏者，不改变 `judge` 或 `verify_evidence.py` 的密钥层/部署层/桥接裁决结果。

## 接入与运行

新增 `session_leads` 字段已接入周期模型读出、历史模型复测，以及 `tvmc.experiment` 的码层和模型读出位模拟合谋结果；旧的单会话 `judge` 保持原状。也可对现有周期实验结果运行：

```bash
.venv/bin/python run_cross_session_attribution.py --run-dir work/epoch_experiment/run1 --output work/epoch_experiment/run1/cross_session_attribution.json
.venv/bin/python -m pytest -q
```

回放入口复核登记记录的上下文、身份种子、参与方证明和码字；**这些检查不能证明登记快照真实可信**。原始HTTP得分结果也没有采集者签名，回放只说明在本地可信输入假设下的结果，不是独立公开裁决。

此外，入口对A-E0和B-E0先前真实模型读出的256个位进行固定随机种子的20次位交错，以及各一次AND和OR；这些是读出位层模拟，不是重新训练或测量合谋/混合模型，不能解释为模型攻击成功率。空交集、无命中、单会话及无效登记用规则测试夹具覆盖；周期实验A/B的参与集共有托管方2，本身没有不相交参与集。
