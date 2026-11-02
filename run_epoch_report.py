import csv,hashlib,json,shutil,subprocess,zipfile
from collections import defaultdict
from pathlib import Path
import argparse
parser=argparse.ArgumentParser(description='Build result tables and public evidence from completed epoch experiments')
parser.add_argument('--run-dir',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
root=Path(__file__).resolve().parent;run=args.run_dir.resolve();work=run.parent
out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
m=json.loads((run/'model_metrics.json').read_text());c=json.loads((run/'crypto_metrics.json').read_text());h=json.loads((run/'historical_metrics.json').read_text())
assert all(x['status']=='completed' for x in (m,c,h))
replay=json.loads((run/'public_replay_audit.json').read_text())
labels={'independent_delivery_keys':'六次交付密钥互异','distinct_delivery_seeds':'六次交付指纹种子互异','initial_decryption':'新交付完整模型包可解密',
'reject_nonzero_constant':'自洽但非零常数的刷新贡献被拒绝','reject_tampered_update_atomic':'篡改更新值被拒绝且不推进','reject_interruption_atomic':'参与方缺席不推进',
'legal_zero_refresh':'合法零常数刷新通过','refresh_preserves_identity':'单独刷新不改变身份承诺','refresh_preserves_key':'刷新保持原密钥','refresh_preserves_model_hash':'刷新解密后的完整模型包哈希一致',
'refreshed_model_reload':'刷新后 safetensors 与适配器可加载','old_proof_replay_rejected':'旧证明提交到新周期被拒绝','identity_rotation_independent':'轮换身份不改变份额值或份额承诺',
'identity_changed':'重抽分支身份承诺变化','identity_unchanged':'不重抽分支身份承诺不变','new_context_proof_valid':'新周期证明验证通过',
'old_proof_new_identity_rejected':'仅错换新身份承诺验证旧证明失败','old_proof_retained_identity_valid':'不重抽时同一身份可验证历史证明',
'forged_epoch_commitment_rejected':'旧份额伪装新周期无法通过新承诺','historical_proof_valid':'历史证明用历史上下文验证通过','historical_snapshot_unchanged':'历史公开快照保持不变',
'captured_complete_old_shares_still_decrypt':'已获得完整旧份额仍可解密','mixed_epochs_interface_rejected':'跨周期混合份额被正常接口拒绝',
'mixed_epochs_direct_observation':'直接混用插值的观测记录','cross_session_shares_rejected':'跨会话份额被拒绝','cross_session_proofs_rejected':'跨会话证明被拒绝',
 'tampered_old_context_rejected':'篡改历史会话上下文被拒绝','tampered_old_participants_rejected':'篡改历史参与方上下文被拒绝','tampered_old_identity_rejected':'篡改历史身份承诺被拒绝',
 'historical_seed_matches_trained_model':'历史身份种子对应原训练模型'}
groups=defaultdict(list)
for r in c['checks']:groups[r['test']].append(r)
crypto_rows=[]
for key,rs in groups.items():
    reps=sorted(set(r['repeat'] for r in rs));good=sum(all(r['passed'] for r in rs if r['repeat']==i) for i in reps)
    crypto_rows.append({'验证项':labels.get(key,key),'通过检查数':sum(r['passed'] for r in rs),'检查总数':len(rs),'通过独立重复':f'{good}/{len(reps)}','标识':key})
model_rows=[]
for r in m['rows']:
    model_rows.append({'模型':r['model'],'定位会话':','.join(r['decoded_sessions']) or '无','登记托管方':';'.join(','.join(map(str,x)) for x in r['decoded_custodians']) or '无',
        '可观察位':r['observed_bits'],'正确位':r['correct_bits'] if r['correct_bits'] is not None else '不适用','PPL':round(r['utility']['perplexity'],6),
        'PPL变化百分比':round(r['ppl_increase']*100,4),'误命中其他会话':','.join(r['wrong_session_hits']) or '无','第二次HTTP':r['second_http'],'正确定位':r['correct_localization']})
def csvfile(name,rows):
    with (out/name).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
csvfile('模型指纹结果.csv',model_rows);csvfile('密码学验证汇总.csv',crypto_rows)
csvfile('密码学逐项明细.csv',[dict(r,detail=json.dumps(r['detail'],ensure_ascii=False)) for r in c['checks']])
csvfile('历史模型重新加载验证.csv',[dict(r,decoded_sessions=','.join(r['decoded_sessions'])) for r in h['rows']])
mixed=[r for r in c['checks'] if r['test']=='mixed_epochs_direct_observation']
total=sum(r['detail']['total'] for r in mixed);same=sum(r['detail']['same_key'] for r in mixed);dec=sum(r['detail']['decrypted'] for r in mixed)
def table(rows,fields=None):
    fields=fields or list(rows[0]);return '\n'.join(['| '+' | '.join(fields)+' |','| '+' | '.join(['---']*len(fields))+' |']+['| '+' | '.join(str(r[f]) for f in fields)+' |' for r in rows])
test_log=(work/'all_tests.log').read_text() if (work/'all_tests.log').exists() else '未提供测试日志'
text=f'''# 份额刷新、身份轮换与周期模型交付实验结果

已实际运行；本次没有复用旧实验成功数字。执行环境 Linux CPU（2 vCPU 配额），Python 与依赖详见 environment.txt。使用本地缓存，无新增下载。本次完整回归日志：{test_log.strip().splitlines()[-1]}。

## 模型实验结果

六个会话模型各训练一次（1200 步），无指纹基座为负对照。SmolLM2-135M 固定版本 `{m['model_revision']}`，基座 SHA-256 `{m['base_sha256']}`。每个服务第一次 HTTP 请求含 768 个自然文本前缀（256 位，每位3个前缀），返回 `logit(1)-logit(0)`；聚合每组三个分数的均值读取一位。使用六候选 Tardos 阈值 {m['threshold']:.6f}，在训练和观测之前确定，未针对结果调参。PPL 使用 WikiText-2 固定 test_offset=256 的64篇文本，最多96词元，实际 {m['baseline']['tokens']} 个预测位置。表中“正确位”不是独立伯努利试验；位间及冗余前缀存在相关性。

{table(model_rows)}

交付格式为完整 ZIP：原始基座 safetensors、分词器、配置、已训练 HeadLoRA 适配器及会话登记记录。整个 ZIP 使用独立密钥 AES-GCM 加密，不仅加密哈希或测试字符串。六份明文包、密文及本地私密实验状态保留在项目 work/epoch_experiment/run1，未放入公共证据包。

## 密码学五组独立重复

每组包含“身份重抽”和“不重抽”两个分支；每个分支创建 A(1,2)、B(2,3,4) 在 E0/E1/E2 的六个独立密钥会话。E0 资产刷新两次，E1 资产刷新一次，E2 资产无需再刷新。仅第一组重抽分支的登记记录对应六次真实模型训练；其余密码学重复使用相同六个模型包作为固定加密载荷，重新采样密钥、多项式和证明随机性，不能当作额外模型训练或指纹定位重复。重抽分支逐周期重新采样身份秘密；不重抽分支保留身份秘密但重新绑定证明上下文。

检查总数：{len(c['checks'])}，失败：{c['failed']}。每行“检查总数”是该验证项在适用会话/周期/分支上的次数，不是独立模型训练次数。

{table(crypto_rows,['验证项','通过检查数','检查总数','通过独立重复'])}

跨周期直接插值共 {total} 个混合组合，恢复相同密钥 {same} 次，成功解密 {dec} 次。这是观测频数，不是无碰撞的数学断言。正常接口对所有这些混合组合按会话、周期及份额/证明检查拒绝。对完整旧周期份额的直接插值仍成功解密，刷新没有撤销旧密钥知识。

补充公开验证：只改证明上下文的周期字段（保留原公开承诺），以及将旧证明用于同系列的新交付会话，共 {len(replay)} 项，拒绝 {sum(r['rejected'] for r in replay)} 项。详见 public_replay_audit.json。

## 身份轮换之后的历史追踪

密码学轮换全部结束后，对每个真实交付密文重新解密，提取完整模型包，重新实例化模型并加载适配器，然后启动独立的一次性服务重测同一批768个前缀。此处是同一模型的历史追踪复测，不是重新训练。用保留的历史身份承诺重新计算种子，并复核原始登记证明。

{table([{'会话':r['sid'],'完整包哈希一致':r['hash_matches'],'历史种子一致':r['seed_matches_historical_registry'],'旧登记证明通过':r['saved_registration_proofs_valid'],'历史定位正确':r['correct_localization'],'正确位':r['correct_bits'],'读出位与原测一致':r['same_observed_bits'],'第二次HTTP':r['second_http']} for r in h['rows']])}

## 解释范围与限制

- 六个模型各一次训练，只能报告本次六个实例的定位结果，不能估计跨训练稳定成功率。密码学5组重复不提供模型训练成功率的额外样本。
- PPL 仅对应本次明确的 WikiText-2 留出片段，不与不同数据划分的历史 PPL 数字直接比较。
- 保留 E0 证明和 E0 公开承诺后，轮换身份不影响历史验证；新周期请求必须使用新上下文。错换新身份承诺的测试只替换身份承诺，保持旧份额承诺和旧上下文，以隔离身份变化的影响。
- 指纹归因得到登记会话及参与方名单，不能认定名单中哪一方泄漏；E0 模型在 E2 被发现仍只能追踪到 E0 会话，不能据此确定泄漏时间。
- 一次性约束按本地服务实例执行，第一次请求200，第二次409。结果依赖接口允许768个输入且如实返回分数，不代表严格单输入或纯生成文本接口。
- 历史记录完整性依赖本地可信快照；身份秘密“删除”仅在隔离字典中模拟删除引用。为测试完整旧份额被窃取的边界，测试工具仍保留私密状态；未声称磁盘/内存安全擦除或真实托管机已删除秘密。
- 刷新协议在本地集中验证全部贡献后才返回新不可变状态；没有实现真实私密认证通道、分布式一致性、恶意参与方公平性或故障恢复系统。Python 大整数运算非恒时，研究码本可公开复现，不是生产秘密配置。

## 复现

在项目目录中运行（模型及 WikiText-2 缓存须已准备好，输出目录须不存在）：

```bash
.venv/bin/python -m pip check
.venv/bin/python -m pytest -q
.venv/bin/python run_epoch_experiment.py --output work/epoch_experiment/new_run
.venv/bin/python run_epoch_crypto.py --run-dir work/epoch_experiment/new_run
.venv/bin/python verify_epoch_history.py --run-dir work/epoch_experiment/new_run
.venv/bin/python run_epoch_report.py --run-dir work/epoch_experiment/new_run --output outputs/epoch_report
```

所有表格来自原始 JSON；公开证据包包含指标、公开身份/承诺/证明、请求及响应分数、日志和本次新增/修改源码，不含身份秘密、密钥、私密份额及模型权重。随机秘密不固定，因此新运行的指纹码字与指标可能不同。源码与证据文件哈希见 SHA256SUMS.json。
'''
(out/'实验结果报告.md').write_text(text)
# Dependency-free OOXML workbook, with each original table on its own sheet.
from xml.sax.saxutils import escape
sheets=[('模型指纹',model_rows),('密码学汇总',crypto_rows),('历史模型复测',[dict(r,decoded_sessions=','.join(r['decoded_sessions'])) for r in h['rows']])]
with zipfile.ZipFile(out/'实验结果表.xlsx','w',zipfile.ZIP_DEFLATED) as z:
    ns='http://schemas.openxmlformats.org/spreadsheetml/2006/main'
    rel='http://schemas.openxmlformats.org/officeDocument/2006/relationships'
    z.writestr('[Content_Types].xml','<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'+''.join(f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>' for i in range(1,len(sheets)+1))+'</Types>')
    z.writestr('_rels/.rels','<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="'+rel+'/officeDocument" Target="xl/workbook.xml"/></Relationships>')
    z.writestr('xl/workbook.xml',f'<workbook xmlns="{ns}" xmlns:r="{rel}"><sheets>'+''.join(f'<sheet name="{escape(name)}" sheetId="{i}" r:id="rId{i}"/>' for i,(name,_) in enumerate(sheets,1))+'</sheets></workbook>')
    z.writestr('xl/_rels/workbook.xml.rels','<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'+''.join(f'<Relationship Id="rId{i}" Type="{rel}/worksheet" Target="worksheets/sheet{i}.xml"/>' for i in range(1,len(sheets)+1))+'</Relationships>')
    for i,(name,rows) in enumerate(sheets,1):
        fields=list(rows[0]);values=[fields]+[[r[f] for f in fields] for r in rows];xml=[]
        for ri,row in enumerate(values,1):
            cells=[]
            for ci,value in enumerate(row):
                ref=chr(65+ci)+str(ri)
                if type(value) in (int,float):cells.append(f'<c r="{ref}"><v>{value}</v></c>')
                else:cells.append(f'<c r="{ref}" t="inlineStr"><is><t>{escape(str(value))}</t></is></c>')
            xml.append(f'<row r="{ri}">'+''.join(cells)+'</row>')
        z.writestr(f'xl/worksheets/sheet{i}.xml',f'<worksheet xmlns="{ns}"><sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews><cols><col min="1" max="{len(fields)}" width="23" customWidth="1"/></cols><sheetData>'+''.join(xml)+f'</sheetData><autoFilter ref="A1:{chr(64+len(fields))}{len(values)}"/></worksheet>')

public=out/'evidence';public.mkdir(exist_ok=True)
for path in run.glob('*.json'):shutil.copy2(path,public/path.name)
for name in ('model.log','crypto.log','historical.log','all_tests.log','environment.txt'):
    if (work/name).exists():shutil.copy2(work/name,public/name)
source=public/'source';source.mkdir(exist_ok=True)
for name in ('run_epoch_experiment.py','run_epoch_crypto.py','verify_epoch_history.py','run_epoch_report.py','serve.py','tvmc/epochs.py','tests/test_epochs.py','requirements.txt','requirements-research.txt','docs/EPOCH_REFRESH.md'):
    dst=source/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/name,dst)
# Source context is needed to reproduce all imported project helpers.
for path in (root/'tvmc').glob('*.py'):shutil.copy2(path,source/'tvmc'/path.name)
manifest={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file() and p.suffix!='.zip' and p.name!='SHA256SUMS.json'}
(out/'SHA256SUMS.json').write_text(json.dumps(manifest,indent=2))
with zipfile.ZipFile(out/'实验结果与公开证据.zip','w',zipfile.ZIP_DEFLATED) as z:
    for path in out.rglob('*'):
        if path.is_file() and path.suffix!='.zip':z.write(path,path.relative_to(out))
print(out)
print('checks',len(c['checks']),'failures',c['failed'],'mixed',total,same,dec)
print(table(model_rows))
