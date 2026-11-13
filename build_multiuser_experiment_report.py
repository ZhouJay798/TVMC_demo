"""Export aggregate experiment tables without secret probes, keys or weights."""
import csv
import json
from pathlib import Path

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.shared import Cm, Pt
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

ROOT=Path(__file__).resolve().parent
OUT=Path('/workspace')
THRESHOLD=62.95931330472342

SUITE_LABELS={
 'baseline':'未处理基线',
 'row_int8_backbone':'逐行INT8动态量化（主干）',
 'row_int8_with_head':'逐行INT8动态量化（主干+输出头）',
 'prune10':'非结构化幅度剪枝10%',
 'prune30':'非结构化幅度剪枝30%',
 'structured_mlp10':'结构化MLP神经元置零10%',
 'full32':'全参普通文本微调32步',
 'lora_r4_32':'注意力LoRA r4微调32步',
 'lora_r8_32':'注意力LoRA r8微调32步',
 'known_probe_erase':'已知探针和码字的定向擦除',
 'full32_then_row_sim':'全参32步→逐行INT8浮点模拟',
 'prune10_then_lora_r8':'剪枝10%→LoRA r8微调32步',
}
GGUF_LABELS={
 'baseline_f32':'GGUF F32基线',
 'group_q8_head_f32':'真实Q8_0，输出头F32',
 'group_q8_head_q8':'真实Q8_0，输出头Q8_0',
 'group_q4_0':'真实Q4_0，输出头Q4_0',
 'group_q4_1':'真实Q4_1，输出头Q4_1',
}
PRUNE_LABELS={
 'normal':'不剪枝','mag10':'幅度剪枝10%（训练见过）',
 'mag20':'幅度剪枝20%（训练未见）','mag30':'幅度剪枝30%（训练见过）',
 'mag40':'幅度剪枝40%（训练未见）','mlp02':'结构化MLP置零2%（训练未见）',
 'mlp05':'结构化MLP置零5%（训练未见）',
 'target_legacy_rows':'定向删除原0/1输出行',
 'target_both_channels':'定向删除原通道和备用通道输出行',
}
STRUCT_LABELS={'normal':'不剪枝','mag20':'幅度剪枝20%（未见）','mag40':'幅度剪枝40%（未见）',
 'mlp02':'结构化MLP置零2%（见过）','mlp03':'结构化MLP置零3%（未见）',
 'mlp05':'结构化MLP置零5%（见过）','mlp07':'结构化MLP置零7%（未见）'}
MILD_LABELS={'all_0.1pct':'每层约0.1% MLP神经元','all_0.5pct':'每层约0.5% MLP神经元',
             'all_1pct':'每层约1% MLP神经元','layer15_10pct':'索引15（第16）层约10% MLP神经元',
             'activation_0.5pct':'激活感知每层约0.5%','activation_1pct':'激活感知每层约1%'}

def load(path):
    if not path.exists():return None
    return json.loads(path.read_text(encoding='utf-8'))

def hit_names(values):
    return '、'.join(v.rsplit('-',1)[-1] for v in values) or '无'

def score_text(scores):return ' / '.join(f'{float(v):.2f}' for v in scores)

def range_text(values,unit='%'):
    if not values:return '未完成'
    return f'{min(values):.2f}{unit}～{max(values):.2f}{unit}'

def row_table(doc,headers,rows):
    table=doc.add_table(rows=1,cols=len(headers))
    table.style='Table Grid'
    for c,s in zip(table.rows[0].cells,headers):c.text=str(s)
    for row in rows:
        cells=table.add_row().cells
        for cell,val in zip(cells,row):cell.text=str(val)
    return table

def add_sheet(wb,name,headers,rows):
    ws=wb.create_sheet(name)
    ws.append(headers)
    for row in rows:ws.append(row)
    ws.freeze_panes='B2';ws.auto_filter.ref=ws.dimensions
    for c in ws[1]:c.font=Font(bold=True,color='FFFFFF');c.fill=PatternFill('solid',fgColor='305A81')
    for col in ws.columns:
        letter=get_column_letter(col[0].column)
        ws.column_dimensions[letter].width=min(55,max(11,max(len(str(c.value or '')) for c in col)+2))
    for row in ws:
        for c in row:c.alignment=Alignment(vertical='center',wrap_text=True)
    return ws


def main():
    suite={u:load(ROOT/f'work/multiuser_suite/run1/U{u}/metrics.json') for u in (1,2,3)}
    gguf={u:load(ROOT/f'work/multiuser_suite/gguf_run1/U{u}/metrics.json') for u in (1,2,3)}
    prune={u:load(ROOT/f'work/prune_resilience/run1/U{u}/metrics.json') for u in (1,2,3)}
    structured={u:load(ROOT/f'work/prune_resilience/structured_run1/U{u}/metrics.json') for u in (1,2,3)}
    mild={u:load(ROOT/f'work/prune_resilience/mild_structured_run1/U{u}/metrics.json') for u in (1,2,3)}
    activation={u:load(ROOT/f'work/prune_resilience/activation_run1/U{u}/metrics.json') for u in (1,2,3)}
    difference={u:load(ROOT/f'work/prune_resilience/difference_attack/U{u}.json') for u in (1,2,3)}
    difference_structured={u:load(ROOT/f'work/prune_resilience/difference_structured/U{u}.json') for u in (1,2,3)}
    clean=load(ROOT/'work/prune_resilience/clean_control.json')
    mixed=load(ROOT/'work/multiuser_suite/collusion2/metrics.json')
    detail=[];qdetail=[];pdetail=[];mixdetail=[];diffdetail=[];structured_detail=[];mild_detail=[]
    for u,x in suite.items():
        if not x:continue
        rid=f'MP-E0-U{u}'
        for case,v in x['branches'].items():
            trace=v['actual_last'];q=v['quality'];hits=trace['hits']
            detail.append([f'U{u}',SUITE_LABELS.get(case,case),round(trace['scores'][u-1],3),
                score_text(trace['scores']),hit_names(hits), '是' if rid in hits else '否',
                hit_names([h for h in hits if h!=rid]),round(q['perplexity'],3),
                round(q.get('ppl_relative_pct',0),3),round(100*q['agreement_vs_original'],2),
                '原始相同语料留出片段；正常词元分数',round(v['elapsed_seconds'],2)])
    for u,x in gguf.items():
        if not x:continue
        rid=f'MP-E0-U{u}'
        for case,v in x['branches'].items():
            trace=v['trace'];q=v['quality'];hits=trace['hits']
            qdetail.append([f'U{u}',GGUF_LABELS.get(case,case),round(trace['scores'][u-1],3),
                score_text(trace['scores']),hit_names(hits),'是' if rid in hits else '否',
                hit_names([h for h in hits if h!=rid]),round(q['perplexity'],3),
                round(q['relative_ppl_pct'],3),round(100*q['agreement_vs_native_f32'],2),
                str(v['storage']['selected_tensor_types']),round(v['storage']['file_bytes']/1048576,2)])
    for u,x in prune.items():
        if not x:continue
        for case,v in x['variants'].items():
            for strategy,s in v['results'].items():
                t=s['trace'];q=s['quality'];rid=f'MP-E0-U{u}'
                primary=t['primary'];alt=t['alternate'];combined=t['combined_hits']
                pdetail.append([f'U{u}',PRUNE_LABELS.get(case,case),
                    {'original':'原始模型','normal_only':'仅无剪枝备用通道训练','prune_aware':'剪枝感知备用通道训练'}.get(strategy,strategy),
                    round(primary['scores'][u-1],3),round(alt['scores'][u-1],3),
                    round(t['alternate_last_prefix']['scores'][u-1],3),
                    score_text(alt['scores']),hit_names(combined),'是' if rid in combined else '否',
                    hit_names([h for h in combined if h!=rid]),round(q['perplexity'],3),
                    round(q.get('relative_ppl_pct_vs_original_unpruned',0),3),
                    round(100*q['agreement_vs_unpruned_original'],2)])
    if mixed:
        for case,v in mixed['cases'].items():
            t=v['actual_last']
            mixdetail.append([case,hit_names(v['contributors']),score_text(t['scores']),hit_names(t['hits']),
                '是' if v['contributor_hit'] else '否',hit_names(v['innocent_hits']),round(v['quality']['perplexity'],3)])
    for scheme,values in (('幅度感知',difference),('结构化感知',difference_structured)):
        for u,x in values.items():
            if not x:continue
            before=x['before'];after=x['after']
            diffdetail.append([scheme,f'U{u}',x['head_rows_selected'],x['total_head_rows'],
                 x['known_primary_rows_selected'],x['known_alternate_rows_selected'],
                 hit_names(before['trace']['combined_hits']),hit_names(after['trace']['combined_hits']),
                 score_text(after['trace']['primary']['scores']),score_text(after['trace']['alternate']['scores']),
                 round(before['quality']['perplexity'],3),round(after['quality']['perplexity'],3),
                 round(after['quality']['relative_ppl_pct_vs_before'],3),
                 round(100*after['quality']['agreement_vs_unpruned_original'],2)])
    for u,x in structured.items():
        if not x:continue
        for case,v in x['cases'].items():
            for strategy,item in v['strategies'].items():
                t=item['trace'];q=item['quality'];alt=t['alternate'];hold=t['alternate_last_prefix']
                rid=f'MP-E0-U{u}'
                structured_detail.append([f'U{u}',STRUCT_LABELS.get(case,case),
                  {'ordinary':'仅正常特征训练','magnitude_aware':'幅度剪枝感知','structured_aware':'加入结构化剪枝训练'}.get(strategy,strategy),
                  round(alt['scores'][u-1],3),round(hold['scores'][u-1],3),score_text(hold['scores']),
                  hit_names(hold['hits']),'是' if rid in hold['hits'] else '否',
                  hit_names([h for h in hold['hits'] if h!=rid]),round(q['perplexity'],3),
                  round(q['ppl_relative_pct'],3),round(100*q['agreement_vs_unpruned_original'],2)])
    for source in (mild,activation):
        for u,x in source.items():
            if not x:continue
            for case,v in x['cases'].items():
                p=v['pruning'];orig=v['original_trace']['primary'];t=v['protected_trace'];q=v['quality'];rid=f'MP-E0-U{u}'
                actual=p.get('actual_neuron_fraction',p.get('all_layer_fraction'))
                mild_detail.append([f'U{u}',MILD_LABELS.get(case,case),round(actual*100,4),
                     round(orig['scores'][u-1],3),round(t['alternate']['scores'][u-1],3),
                     round(t['alternate_last_prefix']['scores'][u-1],3),hit_names(t['alternate_last_prefix']['hits']),
                     hit_names(t['combined_hits']),'是' if rid in t['combined_hits'] else '否',
                     hit_names([h for h in t['combined_hits'] if h!=rid]),round(q['perplexity'],3),
                     round(q['ppl_relative_pct'],3),round(100*q['agreement_vs_unpruned_original'],2)])

    wb=Workbook();wb.remove(wb.active)
    add_sheet(wb,'综合实验', ['副本','测试','本用户得分','U1/U2/U3得分','命中','来源命中','无辜者命中','PPL','PPL变化%','预测一致率%','检测条件','耗时秒'], detail)
    add_sheet(wb,'真实GGUF', ['副本','格式','本用户得分','U1/U2/U3得分','命中','来源命中','无辜者命中','PPL','PPL变化%','预测一致率%','核对的权重类型','文件MiB'], qdetail)
    add_sheet(wb,'剪枝改进', ['副本','操作','方案','原通道得分','备用通道得分','备用留出前缀得分','备用U1/U2/U3','联合命中','来源命中','无辜者命中','PPL','相对原模型PPL变化%','预测一致率%'], pdetail)
    add_sheet(wb,'结构剪枝增强', ['副本','操作','方案','完整三前缀分数','留出前缀分数','留出U1/U2/U3','留出命中','留出来源命中','留出无辜者命中','PPL','PPL变化%','预测一致率%'], structured_detail)
    add_sheet(wb,'温和结构剪枝',['副本','操作','全网络实际神经元比例%','原通道分数','备用通道三前缀分数','备用留出前缀分数','备用留出命中','联合命中','来源命中','无辜者命中','PPL','PPL变化%','预测一致率%'],mild_detail)
    add_sheet(wb,'模型混合与差分', ['操作','参与副本','U1/U2/U3得分','命中','至少一名参与者命中','无辜者命中','PPL'], mixdetail)
    add_sheet(wb,'基座差分定向剪枝',['方案','副本','删去输出行数','输出总行数','发现原通道行数','发现备用通道行数','剪前命中','剪后命中','剪后原通道U1-U3','剪后备用通道U1-U3','剪前PPL','剪后PPL','PPL变化%','预测一致率%'],diffdetail)
    notes=[
      ['检测阈值',f'{THRESHOLD:.4f}；这是原演示阈值。增加备用通道后，联合判定需要独立重新校准，不能直接沿用单通道误报保证。'],
      ['探针与接口','使用原嵌入时的768条自然前缀探针，按实际最后非填充词元提取指定词元logits；远程只返回文本的接口未测试。'],
      ['位置偏差','原项目左填充批处理用attention_mask.sum-1作为位置，在本批768探针中519处不是最后词元；原交付指纹也按历史位置嵌入。本实验原型使用实际末词元，原交付流程未因此自动修复。'],
      ['零分数处理','原聚合器把均值刚好为零判成1。新增可选研究函数 aggregate_with_abstention 将零或弱分数视为擦除，但本次历史结果未启用；新规则和阈值需要版本化及重新校准。擦除规则不能恢复已被清除的信号。'],
      ['任务能力','PPL和下一词元一致率来自同一TinyShakespeare文本的16个留出窗口共1008预测词元；不是业务任务准确率。一致率低也不必然代表业务微调失败。'],
      ['模型来源','同一批已交付U1/U2/U3约135M参数副本；源模型曾经Q4_1量化再反量化并嵌入指纹，不是原始未量化权重。'],
      ['可信边界','发行和登记记录以同次运行保存的公钥核对；不是独立可信身份认证。实验调整后的副本没有重新签发或登记。'],
      ['剪枝感知训练','备用通道训练使用无剪枝、10%、30%非结构化幅度剪枝；20%、40%和结构化剪枝未参与剪枝增强训练。每位前三个相关前缀中，前2个用于本次调整，最后1个只对本次调整留出，但原始模型嵌入时也用过。'],
      ['结构化增强训练','额外将2%和5%结构化 MLP 置零特征加入训练；3%、7%结构化比例未参与训练。它们仅指按权重范数选择MLP神经元并置零，矩阵尺寸不变。'],
      ['温和结构化测试','每层取整后每层约0.065%、0.456%、0.977% MLP神经元，及只对索引15（第16）层取约10%；这几种情形均未用于结构化增强训练，不代表所有结构化剪枝器。'],
      ['激活感知剪枝','在普通文本窗口288–303上收集down_proj输入的平方激活及对应投影列范数，用其乘积排序选择神经元；与模型评估文本窗口分开，不使用指纹探针。属于一次性非迭代启发式。'],
      ['对照含义','仅无剪枝备用训练与剪枝感知备用训练均训练相同32个备用输出行；原始模型不含新通道训练。两个通道使用不同输出行，但共享探针与隐藏特征，不能假定统计独立。定向攻击两通道假设攻击者知道要清除的输出行。'],
      ['基座差分攻击','在本次模型中假设攻击者有精确的公开无指纹基座和实验性带指纹模型，仅根据输出头权重差异寻找应清除的行，不使用密钥、探针或码字；不代表所有模型都能用同一阈值攻击。'],
      ['量化实现','PyTorch逐行方案为真实per-channel qint8动态CPU算子及序列化整数权重，动态处理激活；GGUF为llama.cpp真实Q8_0/Q4_0/Q4_1文件和原生CPU推理；浮点模拟组合单独标示。'],
      ['统计限制','单一基座、每名托管方一份、一组随机种子；没有估计总体误报率或提供任意剪枝保证。'],
    ]
    if clean:
        notes.append(['干净基座阴性对照',f"单个干净基座，原通道与备用通道合并命中：{hit_names(clean['trace']['combined_hits'])}；PPL {clean['quality']['perplexity']:.3f}。仅一次阴性对照，不能估计误报率。"])
    add_sheet(wb,'说明',['项目','说明'],notes)
    workbook=OUT/'TVMC_多用户鲁棒性与剪枝改进实验.xlsx';wb.save(workbook)

    doc=Document();section=doc.sections[0];section.orientation=WD_ORIENT.LANDSCAPE
    section.page_width,section.page_height=section.page_height,section.page_width
    section.top_margin=section.bottom_margin=Cm(1.6);section.left_margin=section.right_margin=Cm(1.5)
    style=doc.styles['Normal'];style.font.name='Microsoft YaHei';style.font.size=Pt(9)
    doc.add_heading('多用户模型指纹鲁棒性与剪枝改进实验',0)
    doc.add_paragraph('实验对象：已交付 U1、U2、U3 的同批约 135M 参数指纹模型。下表报告实际最后非填充词元读法；实验为有限条件验证，不构成任意操作下的保证。')
    doc.add_paragraph('判定阈值 62.959；每行分别检查真实副本是否命中、其他副本是否误命中。PPL 和预测一致率只描述同一语料的留出文本，不能直接等同业务准确率。')
    doc.add_heading('一、三份原始副本：剪枝、微调和其他操作',level=1)
    summary=[]
    for case,label in SUITE_LABELS.items():
        cells=[];changes=[]
        for u in (1,2,3):
            x=suite.get(u);v=x and x['branches'].get(case)
            if not v:cells.append('未完成');continue
            t=v['actual_last'];q=v['quality'];cells.append(f"{t['scores'][u-1]:.2f} / {hit_names(t['hits'])}")
            changes.append(q.get('ppl_relative_pct',0))
        summary.append([label,*cells,range_text(changes)])
    row_table(doc,['操作','U1得分/命中','U2得分/命中','U3得分/命中','PPL变化范围'],summary)
    doc.add_paragraph('逐行 INT8 使用实际 PyTorch 逐通道 qint8 动态 CPU 算子，推理也会动态处理激活；“全参→逐行模拟”则仅模拟权重舍入，两者不可等同。结构化 MLP 操作将成组神经元连接置零，保留原矩阵形状，不宣称推理加速。')
    doc.add_heading('二、真实 GGUF 量化与原生 CPU 推理',level=1)
    qsummary=[]
    for case,label in GGUF_LABELS.items():
        cells=[];changes=[]
        for u in (1,2,3):
            x=gguf.get(u);v=x and x['branches'].get(case)
            if not v:cells.append('未完成');continue
            t=v['trace'];q=v['quality'];cells.append(f"{t['scores'][u-1]:.2f} / {hit_names(t['hits'])}")
            changes.append(q['relative_ppl_pct'])
        qsummary.append([label,*cells,range_text(changes)])
    row_table(doc,['实际格式','U1得分/命中','U2得分/命中','U3得分/命中','PPL变化范围'],qsummary)
    doc.add_paragraph('GGUF 文件由 llama.cpp 量化并使用 llama.cpp CPU 内核实际推理；PPL 与同一运行时 F32 基线比较。Q8_0、Q4_0、Q4_1 的具体权重类型和文件大小见工作簿。')
    doc.add_heading('三、真实模型参数混合与差分',level=1)
    if mixdetail:row_table(doc,['操作','参与者','U1/U2/U3得分','命中','无辜者命中','PPL'],[[r[0],r[1],r[2],r[3],r[5],r[6]] for r in mixdetail])
    else:doc.add_paragraph('未完成；不推断结果。')
    doc.add_paragraph('这是所列权重平均和公共基座差分的有限案例，不证明任意合谋策略或多人误报率。')
    doc.add_heading('四、剪枝感知嵌入与备用读出原型',level=1)
    doc.add_paragraph('使用原码字，增加按秘密键选择的 16 对备用输出词元；只调整对应的32行输出权重。对照模型只在未剪枝特征上训练备用行，增强模型同时使用未剪枝、10%及30%幅度剪枝特征。模型原有0/1读出仍被保留。')
    ps=[]
    for case,label in PRUNE_LABELS.items():
        cells=[]
        for u in (1,2,3):
            x=prune.get(u);v=x and x['variants'].get(case)
            if not v:cells.append('未完成');continue
            r=v['results'].get('prune_aware');t=r['trace']
            cells.append(f"原{t['primary']['scores'][u-1]:.1f} / 备{t['alternate']['scores'][u-1]:.1f} → {hit_names(t['combined_hits'])}")
        ps.append([label,*cells])
    row_table(doc,['剪枝/攻击','U1 原/备→命中','U2 原/备→命中','U3 原/备→命中'],ps)
    doc.add_paragraph('下表单独比较备用通道在本次适配训练中未使用的第三个前缀：每格为“普通训练得分 / 剪枝感知训练得分”。该前缀与前两个相关，且被原始嵌入用过，属于有限留出。')
    held=[]
    for case,label in list(PRUNE_LABELS.items())[:7]:
        cells=[]
        for u in (1,2,3):
            x=prune.get(u);v=x and x['variants'].get(case)
            if not v:cells.append('未完成');continue
            control=v['results']['normal_only']['trace']['alternate_last_prefix']['scores'][u-1]
            robust=v['results']['prune_aware']['trace']['alternate_last_prefix']['scores'][u-1]
            cells.append(f'{control:.2f} / {robust:.2f}')
        held.append([label,*cells])
    row_table(doc,['剪枝方式','U1 普通/增强','U2 普通/增强','U3 普通/增强'],held)
    doc.add_paragraph('备用通道对完整三前缀的得分会受到其中两个训练前缀影响。尤其应参考上表的单个留出前缀结果，并将剪枝后模型表现与得分一起判断。')
    quality_rows=[]
    for case,label in list(PRUNE_LABELS.items())[:7]:
        cells=[]
        for u in (1,2,3):
            x=prune.get(u);v=x and x['variants'].get(case)
            if not v:cells.append('未完成');continue
            original=v['results']['original']['quality']['perplexity']
            improved=v['results']['prune_aware']['quality']['perplexity']
            cells.append(f'{original:.2f} / {improved:.2f}')
        quality_rows.append([label,*cells])
    row_table(doc,['剪枝方式','U1 原/增强 PPL','U2 原/增强 PPL','U3 原/增强 PPL'],quality_rows)
    doc.add_paragraph('所测结构化 MLP 置零可能使 PPL 急剧变差；即使此时仍命中指纹，也不能说明被剪枝模型保持了可用的业务能力。')
    doc.add_paragraph('进一步将2%和5%结构化剪枝特征加入备用通道训练；下表分别是仅正常训练 / 幅度剪枝训练 / 结构化增强训练在第三个前缀上的得分。3%和7%结构化比例未用于训练。')
    structured_rows=[]
    for case,label in STRUCT_LABELS.items():
        cells=[]
        for u in (1,2,3):
            x=structured.get(u);v=x and x['cases'].get(case)
            if not v:cells.append('未完成');continue
            scores=[v['strategies'][name]['trace']['alternate_last_prefix']['scores'][u-1]
                    for name in ('ordinary','magnitude_aware','structured_aware')]
            cells.append(' / '.join(f'{s:.1f}' for s in scores))
        structured_rows.append([label,*cells])
    row_table(doc,['剪枝方式','U1 普通/幅度/结构','U2 普通/幅度/结构','U3 普通/幅度/结构'],structured_rows)
    doc.add_paragraph('另外检验更温和的结构化操作；表中是结构化增强备用通道的留出前缀得分和模型困惑度。')
    mild_rows=[]
    for case,label in MILD_LABELS.items():
        cells=[]
        for u in (1,2,3):
            x=(activation if case.startswith('activation_') else mild).get(u);v=x and x['cases'].get(case)
            if not v:cells.append('未完成');continue
            t=v['protected_trace']['alternate_last_prefix'];q=v['quality']
            cells.append(f"{t['scores'][u-1]:.1f} / PPL {q['perplexity']:.1f}")
        mild_rows.append([label,*cells])
    row_table(doc,['结构化方式','U1 留出分/PPL','U2 留出分/PPL','U3 留出分/PPL'],mild_rows)
    if diffdetail:
        doc.add_paragraph('白盒攻击补充：攻击者取得相同的无指纹公开基座，仅比较输出头权重，识别发生变化的行并置零；攻击选行不使用备用通道密钥或探针。')
        row_table(doc,['方案/副本','发现并清除行/总行','剪前/剪后命中','剪前/剪后PPL','PPL变化'],
          [[f'{r[0]} {r[1]}',f'{r[2]}/{r[3]}',f'{r[6]} / {r[7]}',f'{r[10]} / {r[11]}',f'{r[12]}%'] for r in diffdetail])
        doc.add_paragraph('因此，即便秘密词元通道可抵抗只删除原0/1行，也不能抵抗本例中已知精确基座的差分定位；不能将该原型宣称为抵抗任意剪枝。')
    doc.add_paragraph('原通道和备用通道取并集仅用于研究演示；两个通道沿用原单通道阈值会引入多重检验，正式部署前必须使用独立负样本重新校准。20%、40%及结构化剪枝不在增强训练中；最后一个前缀只对本次调整留出，原始指纹嵌入时已使用，不能称独立探针。')
    doc.add_paragraph('针对性删除原通道与备用通道输出行的测试假定攻击者知道这些位置。备用通道只能提高特定条件下的冗余，不能保证已知所有通道的白盒攻击下仍可追踪。')
    doc.add_heading('五、实验边界和后续工作',level=1)
    for _,s in notes:doc.add_paragraph(s,style='List Bullet')
    doc.add_paragraph('明细工作簿包含每个候选人的分数、命中、误命中、困惑度和预测一致率。实验性备用头权重及秘密键仅保存在本地受限目录，未重新签名、登记或发布。')
    report=OUT/'TVMC_多用户鲁棒性与剪枝改进实验报告.docx';doc.save(report)
    csv_path=OUT/'TVMC_多用户鲁棒性实验明细.csv'
    with csv_path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['副本','测试','本用户得分','U1/U2/U3得分','命中','来源命中','无辜者命中','PPL','PPL变化%','预测一致率%','检测条件','耗时秒']);w.writerows(detail)
    print(report);print(workbook);print(csv_path)


if __name__=='__main__':main()
