"""Short, measured distillation results; never silently hide failed controls."""
import csv


def write_report(m,out):
    base=m.get('clean_test',{})
    lines=['# 蒸馏后指纹定位：实测结果','',f"运行：`{m['run_id']}`；状态：`{m['status']}`。",'',
       f"选择：`{m.get('selection',{}).get('name','尚未选择')}`，只依据验证集；不是所有可能方法中的全局最优。",
       f"最终干净基座 PPL：{base.get('perplexity','未测')}；评分 token：{base.get('tokens','未测')}。",'',
       '## 相同学生、数据与预算的配对对照','',
       '| 蒸馏方式 | 旧自然触发指纹定位 | 新统计指纹定位 | 新学生 PPL | 新学生 PPL 增幅 | 下一词元一致率 |',
       '|---|---|---|---:|---:|---:|']
    rows=[]
    for name,x in m.get('final',{}).get('attacks',{}).items():
        old=m.get('baseline',{}).get('attacks',{}).get(name,{})
        u=x['utility'];new=x['detection']['decoded_users'];prior=old.get('detection',{}).get('decoded_users','未测')
        lines.append(f"| {name} | {prior} | {new} | {u['perplexity']:.3f} | {u['relative_perplexity_increase']:.2%} | {u['next_token_agreement']:.2%} |")
        rows.append([name,prior,new,u['perplexity'],100*u['relative_perplexity_increase'],100*u['next_token_agreement']])
    lines+=['','`[]` 表示未定位，`[1]` 表示定位用户 1。soft 是概率蒸馏；hard_utility 是逐位置最高概率词元训练加干净基座 KL；generated_utility 是实际自回归生成文本训练加干净基座 KL。kl1/kl10 是效用约束系数，均保留，不能只选有利结果。', '',
       '## 五用户嵌入检查','',
       '| 用户 | 定位 | PPL | PPL 增幅 | 下一词元一致率 |','|---|---|---:|---:|---:|']
    for x in m.get('final',{}).get('users',[]):
        u=x['utility'];lines.append(f"| {x['user']} | {x['detection']['decoded_users']} | {u['perplexity']:.3f} | {u['relative_perplexity_increase']:.2%} | {u['next_token_agreement']:.2%} |")
    lines+=['','## 结果边界','',
        f"所有所测蒸馏均能定位：**{m.get('all_tested_distillations_locate',False)}**。",
        '本轮重点是之前失败的短程软概率及保持效用的硬标签蒸馏。退出 0 只表示这些重点门槛通过；实际生成文本蒸馏等失败仍须查看。',
        'PPL ≤10% 是研究工程目标，不是论文要求。原 Demo 的 PPL≤10% 且一致率≥90% 联合指标仍保留在 utility_pass，不将未通过的联合指标改成通过。低 PPL 不等于所有任务能力保持。',
        '本轮模型为 SmolLM2-135M，学生只训练全词表输出头 LoRA、冻结同一公开基座；不是从零训练或跨架构蒸馏。攻击以用户 1 为代表；其他用户仅做嵌入检查。',
        '新检测器需要完整词表概率及公开干净基座作对照。旧检测器读秘密触发的两个词元；两种验证接口成本不同。新方法不是直接恢复旧 256 位码字，也不自动获得 Tardos 抗合谋或公开 Judge 保证。',
        '检测使用独立随机用户密钥和 2047 个随机对照密钥的一侧检验，5 用户 Bonferroni 阈值每用户 0.002；假设无辜密钥独立于可疑模型。不把少量无误告结果写成总体安全证明。',
        '本实验公开种子用于复现，不能作为生产秘密密钥。干净公开模型本身已有语言能力；本轮不证明学生提取了教师独有能力。', '',
        '详细指标、阴性对照、候选失败与参数见 metrics.json。']
    (out/'DEMO_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    with (out/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['attack','old_decoded_users','new_decoded_users','new_ppl','new_ppl_increase_percent','new_agreement_percent']);w.writerows(rows)
