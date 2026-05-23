#!/usr/bin/env python3
"""
Step 19c — Re-read cam_change_point_probe output as a binary
single-speaker vs multi-speaker VAD classifier.

For each probed VAD, label = 'single' if exactly 1 GT speaker has time
overlap with the VAD interval, 'multi' otherwise. The probe emits
adjacent-window cosine sequences; we summarize each VAD with:
  - min_cos:   the lowest adjacent cosine (the dip is the change-point signal)
  - mean_cos:  mean of adjacent cosines
  - dip:       1 - min_cos (higher = more change-point-like)

Then sweep min_cos as a threshold gate: VAD is flagged 'multi' when
min_cos < T. Report TPR/FPR per threshold and AUC.
"""
import json, argparse, sys
from collections import defaultdict

def load_gt(p):
    return [json.loads(l) for l in open(p) if l.strip()]

def load_probe(p):
    out=[]
    for l in open(p):
        if not l.strip(): continue
        o=json.loads(l)
        if not o.get('adj_cos'): continue
        o['min_cos']=min(o['adj_cos'])
        o['mean_cos']=sum(o['adj_cos'])/len(o['adj_cos'])
        out.append(o)
    return out

def gt_speakers_in_vad(gt, vs, ve):
    sp=set()
    for g in gt:
        gs=g['start_ms']/1000.0
        ge=g['end_ms']/1000.0
        if not (ge<=vs or gs>=ve):
            sp.add(g['speaker'])
    return sp

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--probe', required=True)
    ap.add_argument('--gt', default='tests/fixtures/test_ground_truth_v1.jsonl')
    args=ap.parse_args()

    gt=load_gt(args.gt)
    probe=load_probe(args.probe)
    print(f"[input] gt={len(gt)}  probed_vads={len(probe)}")

    # Label each VAD as single (1 speaker) / multi (2+) / nospk (0 GT overlap).
    rows=[]
    for p in probe:
        sp=gt_speakers_in_vad(gt, p['start_sec'], p['end_sec'])
        label='single' if len(sp)==1 else ('multi' if len(sp)>=2 else 'nospk')
        rows.append((p, label, sp))

    n_single=sum(1 for _,l,_ in rows if l=='single')
    n_multi =sum(1 for _,l,_ in rows if l=='multi')
    n_no    =sum(1 for _,l,_ in rows if l=='nospk')
    print(f"[labels] single={n_single}  multi={n_multi}  nospk={n_no}")

    # Show distribution
    print("\n=== min_cos distribution by label ===")
    for lab in ['single','multi','nospk']:
        vals=sorted([r[0]['min_cos'] for r in rows if r[1]==lab])
        if not vals: continue
        n=len(vals)
        med=vals[n//2]
        p25=vals[n//4]
        p75=vals[(3*n)//4]
        print(f"  {lab:8s} n={n:3d}  min={vals[0]:.3f}  p25={p25:.3f}  median={med:.3f}  p75={p75:.3f}  max={vals[-1]:.3f}")

    # ROC-style: for threshold T, predict 'multi' if min_cos < T. Compute
    # TP/FP/TN/FN against {multi vs single}. Sweep T from 0.30 to 0.95.
    pos=[r[0]['min_cos'] for r in rows if r[1]=='multi']   # true positives if T>x
    neg=[r[0]['min_cos'] for r in rows if r[1]=='single']  # false positives if T>x
    if not pos or not neg:
        print("[warn] no positives or no negatives; skipping ROC")
        return

    print("\n=== ROC sweep: min_cos < T => predicted multi ===")
    print("  T      TPR     FPR     precision  recall  F1   (TP/FP/FN/TN)")
    best_f1=0; best_T=0; best_stats=None
    Ts=[i/100 for i in range(30,96,2)]
    rocs=[]
    for T in Ts:
        tp=sum(1 for v in pos if v<T)
        fp=sum(1 for v in neg if v<T)
        fn=len(pos)-tp
        tn=len(neg)-fp
        tpr=tp/max(1,len(pos))
        fpr=fp/max(1,len(neg))
        prec=tp/max(1,tp+fp)
        rec=tpr
        f1=2*prec*rec/max(1e-9,prec+rec)
        rocs.append((T,tpr,fpr,prec,rec,f1,tp,fp,fn,tn))
        if f1>best_f1:
            best_f1=f1; best_T=T; best_stats=(tp,fp,fn,tn,prec,rec)

    # Print sparse table
    for T,tpr,fpr,prec,rec,f1,tp,fp,fn,tn in rocs:
        marker='  <--' if T==best_T else ''
        print(f"  {T:.2f}   {tpr:.3f}   {fpr:.3f}   {prec:.3f}      {rec:.3f}   {f1:.3f}   ({tp}/{fp}/{fn}/{tn}){marker}")

    # Trapezoid AUC
    rocs_sorted=sorted(rocs, key=lambda r:r[2])  # by FPR
    auc=0.0
    for i in range(1,len(rocs_sorted)):
        x0,y0=rocs_sorted[i-1][2],rocs_sorted[i-1][1]
        x1,y1=rocs_sorted[i][2],rocs_sorted[i][1]
        auc += (x1-x0)*(y0+y1)/2
    print(f"\n[AUC] approx={auc:.3f}  (chance=0.500)")
    print(f"[best F1] T={best_T:.2f}  F1={best_f1:.3f}  TP/FP/FN/TN={best_stats[:4]}  prec={best_stats[4]:.3f}  rec={best_stats[5]:.3f}")

    # Also test "max dip" magnitude: 1 - min_cos
    print("\n=== Sanity: single vs multi mean(min_cos) ===")
    import statistics as st
    sm=[r[0]['min_cos'] for r in rows if r[1]=='single']
    mm=[r[0]['min_cos'] for r in rows if r[1]=='multi']
    print(f"  single: mean={st.mean(sm):.3f}  stdev={st.pstdev(sm):.3f}")
    print(f"  multi:  mean={st.mean(mm):.3f}  stdev={st.pstdev(mm):.3f}")

if __name__=='__main__':
    main()
