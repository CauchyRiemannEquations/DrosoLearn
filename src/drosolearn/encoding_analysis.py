"""Descriptive representation plots, pair tables, and a nonexpert report."""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .encoding import PAIRS,PRIMARY,ANCHORED
from .experiment import write_json

NAMES={"fixed":"실제 배선·출력만 학습","plastic":"실제 배선·내부도 학습",
       "rewired_fixed":"재배선·출력만 학습","rewired_plastic":"재배선·내부도 학습",
       "ann_fixed":"일반 신경망·출력만 학습","ann_plastic":"일반 신경망·내부도 학습"}

def analyze(output):
    output=Path(output); meta=json.loads((output/"metadata.json").read_text(encoding="utf-8")); cfg=meta["config"]
    summary=json.loads((output/"summary.json").read_text(encoding="utf-8")); outcomes=summary["outcomes"]
    with (output/"learning_curves.csv").open(encoding="utf-8",newline="") as stream: curves=list(csv.DictReader(stream))
    conditions=cfg["conditions"]; seeds=cfg["seeds"]; n=cfg["trials"]; every=cfg["checkpoint_every"]
    def selected(c,f):return [r for r in outcomes if r["condition"]==c and r["fit"]==f]
    fig,ax=plt.subplots(figsize=(10,4.5))
    for i,c in enumerate(conditions):
        for fit,style in (("quantity_present","-"),("quantity_removed","--")):
            data=np.array([[float(r["p_target"]) for r in curves if int(r["seed"])==s and r["condition"]==c and r["fit"]==fit and r["split"]=="iid"] for s in seeds])
            ax.plot(np.arange(data.shape[1])*every,data.mean(axis=0),style,color=plt.get_cmap("tab10")(i),label=f"{c}: {fit}")
    ax.axhline(.5,color="black",linestyle=":"); ax.set(xlabel="Rewarded training trials",ylabel="Correct-choice probability",ylim=(0,1.02),title="Only2-vs-4 training; permuted evaluations reuse the present-input fit")
    ax.legend(fontsize=6,bbox_to_anchor=(1.01,1));fig.tight_layout();fig.savefig(output/"acquisition.png",dpi=160);plt.close(fig)

    fig,ax=plt.subplots(figsize=(11,4.5)); x=np.arange(len(conditions)); width=.24
    keys=("ordered_unseen","permuted_unseen","removed_unseen")
    for j,(key,label) in enumerate(zip(keys,("Ordered scalar","720 anchored-code mean","Quantity removed"))):
        vals=[float(np.mean([r[key] for r in selected(c,"quantity_removed" if key=="removed_unseen" else "quantity_present")])) for c in conditions]
        ax.bar(x+(j-1)*width,vals,width,label=label)
    ax.axhline(.5,color="black",linestyle=":");ax.set(xticks=x,xticklabels=conditions,ylim=(0,1.02),ylabel="Macro mean of 15 fully untrained quantity pairs",title="Ordered/permuted share a policy; quantity-removed is trained separately")
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(output/"unseen_order.png",dpi=160);plt.close(fig)

    pair_rows=[]; matrices={}
    for c in conditions:
        ordered=[]; permuted=[]; removed=[]
        for seed in seeds:
            base=output/f"seed_{seed}"/c
            with np.load(base/"quantity_present/transfer_final.npz",allow_pickle=False) as a:
                ordered.append(a["map_p_target"][0].copy());permuted.append(a["map_p_target"].mean(axis=0))
            with np.load(base/"quantity_removed/transfer_final.npz",allow_pickle=False) as a:removed.append(a["p_target"].copy())
        matrices[c]={"ordered":np.array(ordered),"permuted":np.array(permuted),"removed":np.array(removed)}
        for j,(low,high) in enumerate(PAIRS):
            pair_rows.append({"condition":c,"low":low,"high":high,"ratio":low/high,"difference":high-low,
                              "stratum":"fully_unseen" if j in set(PRIMARY) else "one_anchor",
                              "ordered_mean":float(np.mean(matrices[c]["ordered"][:,j])),"permuted_mean":float(np.mean(matrices[c]["permuted"][:,j])),"removed_mean":float(np.mean(matrices[c]["removed"][:,j]))})
    with (output/"pair_summary.csv").open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(pair_rows[0]));writer.writeheader();writer.writerows(pair_rows)
    fig,axes=plt.subplots(2,1,figsize=(14,6.5),sharex=True,sharey=True)
    for ax,c in zip(axes,("fixed","plastic")):
        if c not in matrices: ax.set_visible(False);continue
        for key,style in (("ordered","o-"),("permuted","s-"),("removed",": ")):
            ax.plot(np.arange(len(PAIRS)),matrices[c][key].mean(axis=0),"--" if key=="removed" else style,label=key)
        ax.axhline(.5,color="black",linestyle=":");ax.set(title=c,ylabel="Correct-choice probability",ylim=(0,1.02));ax.legend(fontsize=8)
    axes[-1].set(xticks=np.arange(len(PAIRS)),xticklabels=[f"{a}:{b}" for a,b in PAIRS],xlabel="All27 new pairs, including12 with one trained quantity")
    fig.suptitle("Per-pair results; anchored pairs retain partial supplied order")
    fig.tight_layout();fig.savefig(output/"all_pairs.png",dpi=150);plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,key,title in zip(axes,("internal_change_l2","readout_change_l2","probe_mbon_change_l2"),("Internal weights","Readout weights","Fixed-probe MBON activity")):
        for c in ("fixed","plastic"):
            if c not in conditions:continue
            for f,style in (("quantity_present","-"),("quantity_removed","--")):
                d=np.array([[float(r[key]) for r in curves if int(r["seed"])==s and r["condition"]==c and r["fit"]==f and r["split"]=="iid"] for s in seeds])
                ax.plot(np.arange(d.shape[1])*every,d.mean(axis=0),style,label=f"{c}/{f}")
        ax.set(title=title,xlabel="Training trials",ylabel="L2 distance from initial state")
    axes[-1].legend(fontsize=6);fig.tight_layout();fig.savefig(output/"changes.png",dpi=150);plt.close(fig)

    aggregate=[];rows=[]
    for c in conditions:
        present,removed=selected(c,"quantity_present"),selected(c,"quantity_removed")
        r={"condition":c,"acquisition_passes":sum(t["acquired"] for t in present),"ordered_transfer_passes":sum(t["qualified_ordered_transfer"] for t in present),
           **{key:float(np.mean([t[key] for t in present])) for key in ("after_accuracy","ordered_unseen","ordered_unseen_before","ordered_unseen_gain","permuted_unseen","ordered_anchored","permuted_anchored","ordered_all","permuted_all")},
           "removed_unseen":float(np.mean([t["removed_unseen"] for t in removed]))}
        aggregate.append(r)
        rows.append(f"| {NAMES[c]} | {100*r['after_accuracy']:.1f}% ({r['acquisition_passes']}/{len(seeds)}) | {100*r['ordered_unseen_before']:.1f}%→{100*r['ordered_unseen']:.1f}% | {100*r['permuted_unseen']:.1f}% | {100*r['removed_unseen']:.1f}% | {r['ordered_transfer_passes']}/{len(seeds)} |")
    write_json(output/"analysis_summary.json",{"scope":cfg["scope"],"aggregate":aggregate,"uncertainty":"descriptive seed means; no biological independence or significance test",
               "primary_pairs":15,"anchored_pairs":12,"all_pairs":27,"permutations":720,"actual_updates":summary["actual_training_trials"]})
    table="\n".join(["| 모델 | 연습한2대4 성적·습득 | 새로운 수량: 순서 코드 학습 전→후 | 새로운 수량: 임의 코드 평균 | 수량 정보 제거 | 습득 후 순서 코드 전이 기준 |",
                     "|---|---:|---:|---:|---:|---:|",*rows])
    met=sum(r["criterion_met"] for r in summary["family_verdicts"])
    text=f"""# 수량 입력이 제공한 순서의 영향

## 먼저 쉬운 설명

지금까지 모델에는2보다4가 크다는 순서를 입력으로 제공했습니다. 이번에는 연습에 쓰는2와4는 그대로 두고, 처음 보는 수량의 코드만 임의 이름표처럼 바꿨습니다. 예를 들어 어떤 이름표의 숫자가 크다고 실제 물체 수가 많은 것은 아닐 수 있습니다. 배운 모델은 그대로이며 평가 때 추가로 가르치지 않습니다.

가능한 이름표 배치720가지를 모두 평가했습니다. **두 수량 모두 연습하지 않은15쌍에서 그 평균이50%가 되는 것은 이 실험의 대칭으로 미리 증명되는 성질**입니다. 모델의 새로운 실패나 실제 초파리의 한계를 발견했다는 뜻이 아닙니다. 핵심은 기존 순서 코드에서 나타나는 전이가 얼마나 입력으로 알려준 순서에 의존하는지 보는 것입니다.

## 결과

성적은 정답을 고를 확률의 평균이며 아래는{len(seeds)}개 계산 시드 평균입니다. 새 수량15쌍은1·3·5·6·7·8 사이 모든 조합입니다. 습득과 전이는 별도로 판정했습니다.

{table}

순서 코드와 임의 코드의 평균 차이≥{100*cfg['criteria']['representation_gap']:.0f}%포인트이고 양수 시드 비율≥{100*cfg['criteria']['positive_seed_fraction']:.0f}%인 사전 기준은{met}/{len(conditions)}모델이 통과했습니다. 이는 같은 함수에 주는 정보 방식의 효과이며, 학습 중 서로 다른 원리가 형성됐다는 결론이 아닙니다. 습득 후 절대 전이 기준≥{100*cfg['criteria']['transfer_accuracy']:.0f}%는 별도 열에 보고했습니다. 실패 시드를 제외하지 않았습니다.

## 무엇을 맞췄고 무엇은 다른가

- 2대4의 연습 입력은 순서 코드와 모든 임의 코드에서 정확히 같습니다. 따라서 한 번 학습한 모델을720회 고정 평가하며 추가 보상·업데이트는 없습니다.
- 수량 정보를 없앤 대조만 별도로 학습했습니다. 그 대조의 고정 평가50%도 단서 교차 균형에서 보장됩니다. 가중치가 바뀌어도 수량 정보가 생겼다는 뜻은 아닙니다.
- 실제 학습은{summary['actual_training_trials']:,}시행입니다. 720은 이름표 배치 수이며 독립 개체 수나 추가 학습 횟수가 아닙니다.
- 모두 같은7차원 입력·고정 encoder·뉴런·가중치 제한·학습률·{n}시행 예산을 사용했습니다. 임의 코드도 수량의 정체성과 인공1차원 거리는 남습니다.

## 새 수량쌍을 나누는 이유

전체 새27쌍 중12쌍은2나4 중 하나를 이미 연습에서 본 경우입니다. 두 고정점을 남겼으므로 이들 임의 코드에도 일부 순서 단서가 남습니다. 단순히 코드가 큰 쪽을 고르는 기준 모델은15개의 완전 새 쌍에서는50%, 한 수량을 본12쌍에서는63.89%, 전체27쌍에서는56.17%입니다. 따라서 전체27쌍 평균만 보고 임의 코드에서도 규칙을 배웠다고 해석하면 안 됩니다. [모든 쌍의 결과](pair_summary.csv)에 두 범주를 구분했습니다.

이름표와 진짜 수량의 대응표를 모두 아는 특권적 기준 모델은 임의 코드에서도100%를 낼 수 있습니다. 그러나 실제 모델은 연습한2와4 밖의 수량에 대해 그런 정보를 받지 않았습니다. 순서 코드의 성공도 물체를 세거나 수학적 순서를 스스로 발견했다는 증거가 아닙니다. 차이·비율에 따른 반응 역시 제공한 scalar 간격의 영향과 구분해야 합니다.

## 재현과 검산

[사전 프로토콜](../../docs/encoding-protocol.md) · [미리 증명한 설계 성질](../../docs/encoding-background.md) · [모든 시드 판정](summary.json) · [실행 정보](metadata.json) · [독립 검산](verification.json)

각 seed에는 입력·난수·배선·시행별 기록과 모든 checkpoint가 있습니다. 전이 cache는 모든 후보의 입력 코드별 출력 점수를 저장하며720배치를 동일한 고정 함수에서 계산합니다. 별도 검증기는 그 점수와 모든 map별 지표를 재계산합니다. 인간 데이터와 생물학적 회로 비교는 이번에 수행하지 않았습니다. Level4/5·시각 수량 인지·수학 규칙 발견은 주장하지 않습니다.

![습득](acquisition.png)

![완전히 새로운 수량쌍](unseen_order.png)

![모든 새 쌍](all_pairs.png)

![변화의 구분](changes.png)
"""
    (output/"report.md").write_text(text,encoding="utf-8")
    return aggregate
