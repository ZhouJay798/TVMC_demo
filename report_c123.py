"""Summarize observed counts, preserving failures and protocol boundaries."""
import argparse,json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args();d=a.run_dir
    m=json.loads((d/'model_metrics.json').read_text());c=json.loads((d/'c123_metrics.json').read_text());crypto=json.loads((d/'crypto_metrics.json').read_text())['trials'];offline=json.loads((d/'c1_offline_control.json').read_text())['trials']
    if m.get('status')!='completed' or c.get('status')!='completed':raise ValueError('both experiments must finish before reporting')
    def frac(values):return f'{sum(bool(x) for x in values)}/{len(values)}'
    lines=['# C1/C2/C3 本轮实测结果','',f"基座PPL：{m['baseline']['perplexity']:.6f}；WikiText-2固定测试集 {m['baseline']['tokens']} 个词元。每个副本新训练一次，密码学独立随机性重复5次。",'',
           '## 交付和原始副本定位','', '|登记接收人|本人解密|跨人解密拒绝|正确位/256|命中接收人|PPL|PPL增长|','|---|---|---|---|---|---|---|']
    for i,r in enumerate(m['rows']):
        lines.append(f"|{r['recipient']}|{int(r['owner_decrypt_hash_matches'])}/1|{frac(list(r['cross_key_rejected'].values()))}|{r['batch']['correct_bits'][i]}/256|{r['batch']['decoded_recipients']}|{r['ppl']:.6f}|{r['ppl_increase_pct']:.3f}%|")
    lines+=['','## 密码学对照','', '|项目|成功数/测试数|','|---|---|']
    for key,label in [('full_recovers','完整3/3恢复'),('own_decrypt','本人解密'),('wrong_person_rejections','跨人私钥拒绝'),('insufficient_interface_rejections','不足份额接口拒绝'),('direct_incomplete_decrypt_failed','直接不足份额插值解密失败（观测）'),('wrong_session_key_rejected','错误会话密钥拒绝'),('tampered_outer_rejected','外层篡改拒绝'),('wrong_issuer_rejected','错误发行公钥拒绝'),('modified_manifest_rejected','签名清单篡改拒绝'),('public_key_substitution_rejected','个人公钥替换拒绝')]:
        vals=[]
        for trial in crypto:vals.extend(trial[key] if isinstance(trial[key],list) else [trial[key]])
        lines.append(f'|{label}|{frac(vals)}|')
    rot=m['combined_refresh_rotation'];lines+=['',f"刷新+身份轮换：份额变化 {frac(rot['share_values_changed'])}；身份承诺变化 {frac(rot['identity_commitments_changed'])}；密钥不变 {rot['session_key_unchanged']}；同一完整外层包重新解密 {rot['full_outer_bundle_reopened_after_refresh']}。",'',
        '## 三类场景','', '|场景|实测|解释|','|---|---|---|',
        f"|C1 重构回执|{frac(list(c['C1']['controls'].values()))} 项检查通过|包括1个有效回执和6个拒绝对照；受信服务见证执行，知识证明见证秘密知识|"]
    # Build counts independently of labels; never equate missing hits with successful attribution.
    originals=[r['batch']['decoded_recipients']==[r['recipient']] for r in m['rows']]
    text=[r['registered_custodians']==[i+1] for i,r in enumerate(c['C2'])]
    strict=[r['strict_single']['decoded_recipients']==[r['recipient']] for r in m['rows']]
    lines += [f"|C1 绕过日志服务直接插值|恢复成功 {frac([r['offline_key_matches'] for r in offline])}；回执数 {sum(r['reconstruction_receipt_emitted'] for r in offline)}|不能强制所有离线重构留下记录|",f'|C2 一次请求、768个分数、无证明|正确定位 {frac(originals)}|依赖允许探针并返回真实分数|',f'|C2 一次请求、768个贪心文本词元|正确定位 {frac(text)}|普通文本不等同于0/1分数接口|',f'|C2 严格单输入单分数|正确定位 {frac(strict)}|当前256位协议信息不足|']
    for transform,label in [('merged_head','合并指纹更新到输出头'),('int8_delta','更新矩阵8位量化反量化'),('prune50_delta','更新矩阵剪枝50%'),('remove_delta','移除指纹更新')]:
        rows=[r for r in c['C3'] if r['transform']==transform]
        lines.append(f"|C3 {label}|正确定位 {frac([r['exact_source_match'] for r in rows])}；错误接收人命中 {sum(len(r['wrong_hits']) for r in rows)}|PPL："+', '.join(f"{r['ppl']:.4f}" for r in rows)+'|')
    http=[json.loads(f.read_text())['result'] for f in d.glob('*_http.json')]
    lines += ['',f"一次性端点第二次请求拒绝：{frac([r['second_http']==409 for r in http])}；无指纹基座命中接收人：{m['negative_control']['decoded_recipients']}。"]
    lines += ['', '## 逐副本变换明细','', '|托管方|变换|正确位/256|命中托管方|PPL|PPL增长|第二次HTTP|','|---|---|---|---|---|---|---|']
    for row in c['C3']:
        i=row['source_custodian']
        lines.append(f"|{i}|{row['transform']}|{row['correct_bits'][i-1]}/256|{row['registered_custodians']}|{row['ppl']:.6f}|{row['ppl_increase_pct']:.3f}%|{row['second_http']}|")
    lines += ['', '## 结论边界','',
        '指纹命中的是登记给具体托管方的副本。复制U1明文交由标记为U2的服务运行，仍命中 '+str(m['transfer_control']['decoded_recipients'])+'，因此不能由指纹证明现实泄漏者身份。',
        '三对50/50有效模型更新混合命中：'+str([r['decoded'] for r in m['real_model_mixtures']])+'。多源命中保留多个来源。',
        '当前交付包含基座和独立指纹适配器；忽略适配器即可恢复无指纹基座。因此当前结构没有解决该删除攻击，不能认定C3全面通过。',
        '本轮重新训练每份模型一次，不能估计跨训练成功率。更新矩阵量化/剪枝不等于全模型变换；任意蒸馏或攻击未被覆盖。',
        '完整外层密文在内存完成两次解密后释放，未保存原字节。保留的签名清单、回执和HTTP输出支持本地快照下一致性复核，不能证明远端服务或现实个人身份。']
    (d/'REPORT.md').write_text('\n'.join(lines)+'\n')
    crypto_ok=all(all(v) if isinstance(v,list) else bool(v) for trial in crypto for k,v in trial.items() if k!='repeat')
    cross_ok=all(all(r['cross_key_rejected'].values()) for r in m['rows'])
    passed=crypto_ok and cross_ok and not m['negative_control']['decoded_recipients'] and all(originals) and all(c['C1']['controls'].values()) and all(r['owner_decrypt_hash_matches'] for r in m['rows']) and rot['full_outer_bundle_reopened_after_refresh']
    print(json.dumps({'supported_delivery_and_score_interface_pass':passed,'arbitrary_text_or_transformation_solved':False,'report':str(d/'REPORT.md')},ensure_ascii=False))
    if not passed:raise SystemExit(1)

if __name__=='__main__':main()
