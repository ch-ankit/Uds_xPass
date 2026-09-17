"""
build_dashboard.py
==================
Uds_xPass — builds a single dependency-free HTML dashboard.

Content: headline findings, SVG forest plot (weak foot), calibration /
ablation / recalibration / weighting charts from poster_outputs CSVs, plus
a calibration-geography map and player cards built from a sampled set of
held-out predictions with coordinates.

USAGE
-----
    conda run -n xpass python build_dashboard.py
    conda run -n xpass python build_dashboard.py --ids 90 42   # pitch seasons
    # open xpass_dashboard.html in any browser (no server needed)

OUTPUT
    xpass_dashboard.html  (~500 KB, data inlined as JSON)
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from all_seasons import (  # noqa: E402
    pull_season_passes, add_weak_foot_flag, build_features,
    BASE_FEATS, train_and_predict,
)

OUT = Path("xpass_dashboard.html")
POSTER = Path("poster_outputs")

TYPE_FLAGS = {"Cross": "pass_cross_f", "Switch": "pass_switch_f",
              "Through ball": "pass_through_ball_f"}


def ptype(row):
    for name, flag in TYPE_FLAGS.items():
        if row.get(flag) == 1:
            return name
    return "Ordinary"


def load_csv(name):
    p = POSTER / name
    return pd.read_csv(p).to_dict(orient="records") if p.exists() else []


def sample_pitch_passes(season_ids, per_season=1200, seed=42):
    """Retrain current model per season; stratified sample of TEST passes."""
    rng = np.random.RandomState(seed)
    out = []
    for sid in season_ids:
        passes = pull_season_passes(11, int(sid), f"id={sid}")
        if passes.empty:
            continue
        sname = str(passes["season_name"].iloc[0])
        res = train_and_predict(build_features(add_weak_foot_flag(passes)),
                                BASE_FEATS, seed=seed)
        if res is None:
            continue
        test, auc, _ = res
        print(f"[{sname}] test n={len(test):,} AUC={auc:.4f}")
        test = test.copy()
        test["ptype"] = test.apply(ptype, axis=1)
        keep = []
        for t in ["Cross", "Through ball", "Switch"]:
            sub = test[test.ptype == t]
            keep.append(sub if len(sub) <= 400 else sub.sample(400, random_state=seed))
        # Guarantee weak-foot representation (only ~13% naturally): the map
        # + player views slice by foot, so an unstratified sample starves them.
        if "weak_foot" in test.columns:
            have_idx = pd.concat(keep).index if keep else test.iloc[0:0].index
            extra = test[(test["weak_foot"] == 1).fillna(False)].drop(
                have_idx, errors="ignore")
            if len(extra):
                keep.append(extra.sample(min(len(extra), 400), random_state=seed))
        n_rare = sum(map(len, keep))
        ord_sub = test[test.ptype == "Ordinary"]
        keep.append(ord_sub.sample(min(len(ord_sub), max(0, per_season - n_rare)),
                                  random_state=seed))
        for _, r in pd.concat(keep).iterrows():
            loc, end = r.get("location"), r.get("pass_end_location")
            if not isinstance(loc, list) or not isinstance(end, list):
                continue
            wf = r.get("weak_foot")
            out.append({
                "s": sname, "p": str(r.get("player", "?")),
                "t": str(r.get("team", "?")),
                "sx": round(float(loc[0]), 1), "sy": round(float(loc[1]), 1),
                "ex": round(float(end[0]), 1), "ey": round(float(end[1]), 1),
                "x": round(float(r["pred"]), 3), "a": int(r["actual"]),
                "k": r["ptype"], "w": -1 if pd.isna(wf) else int(wf),
                "l": round(float(r.get("pass_length", 0)), 1),
                "u": int(bool(r.get("under_pressure", False))),
            })
    rng.shuffle(out)
    return out


HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Uds_xPass Dashboard — La Liga</title>
<style>
:root{--bg:#F4F5F1;--card:#fff;--ink:#16202B;--soft:#546270;--line:#D5D8CE;
--blue:#1D4ED8;--red:#DC2626;--green:#15803D;--grey:#9CA3AF;--amber:#D97706}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--ink);font-family:system-ui,-apple-system,sans-serif;line-height:1.5}
.wrap{max-width:1180px;margin:0 auto;padding:0 20px 80px}
.hero{padding:40px 0 24px;border-bottom:2px solid var(--line)}
.hero h1{font-size:30px;margin-bottom:6px}.hero p{color:var(--soft);max-width:760px}
.chips{display:flex;gap:10px;flex-wrap:wrap;margin-top:16px}
.chip{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:10px 16px;min-width:150px}
.chip b{display:block;font-family:monospace;font-size:20px}
.chip span{font-size:12px;color:var(--soft)}
.tabs{display:flex;gap:6px;flex-wrap:wrap;margin:20px 0}
.tabs button{border:1px solid var(--line);background:var(--card);border-radius:6px;padding:8px 14px;cursor:pointer;font-size:14px}
.tabs button.on{background:var(--ink);color:#fff;border-color:var(--ink)}
.tab{display:none}.tab.on{display:block}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:20px;margin-bottom:16px}
.card h2{font-size:18px;margin-bottom:4px}.card p.d{color:var(--soft);font-size:14px;margin-bottom:12px}
table{border-collapse:collapse;font-size:13px;margin-top:8px}
th,td{border-bottom:1px solid var(--line);padding:5px 10px;text-align:right}
th:first-child,td:first-child{text-align:left}
.tblwrap{overflow-x:auto}
.filters{display:flex;gap:12px;flex-wrap:wrap;align-items:end;margin-bottom:12px;font-size:13px}
.filters label{display:flex;flex-direction:column;gap:4px;color:var(--soft)}
.filters select,.filters input{background:#fff;border:1px solid var(--line);border-radius:6px;padding:6px 8px;font-size:13px;color:var(--ink)}
#tip{position:fixed;display:none;background:var(--ink);color:#fff;font-size:12px;border-radius:6px;padding:8px 10px;pointer-events:none;z-index:9;font-family:monospace}
.pitchwrap{overflow-x:auto}
svg text{font-family:monospace}
.note{font-size:13px;color:var(--soft);margin-top:10px}
</style></head><body><div class="wrap">
<div class="hero"><h1>Can We Predict Every Pass?</h1>
<p>Uds_xPass — event-only XGBoost xPass over 18 La Liga seasons. Weak foot hurts;
calibration fails on rare types from scarcity, not missing features.</p>
<div class="chips" id="chips"></div></div>
<div class="tabs" id="tabs"></div>
<div id="body"></div>
</div><div id="tip"></div>
<script>
const D = __DATA__;
let CUR = [];
const $ = s => document.querySelector(s);
const tip = $('#tip');
function showTip(e, h){ tip.innerHTML = h; tip.style.display='block';
  tip.style.left = (e.clientX+12)+'px'; tip.style.top = (e.clientY+12)+'px'; }
function hideTip(){ tip.style.display='none'; }
document.addEventListener('mousemove', e => { if(tip.style.display==='block'){
  tip.style.left=(e.clientX+12)+'px'; tip.style.top=(e.clientY+12)+'px'; } });

// Chips
$('#chips').innerHTML = [
  ['0.85–0.90','per-season AUC'],['−20–25%','weak-foot completion odds'],
  ['0.003 → 0.043','ECE ordinary → through ball'],['14×','calibration gap, rare vs ordinary']
].map(c => `<div class="chip"><b>${c[0]}</b><span>${c[1]}</span></div>`).join('');

// Generic grouped bars
function bars(el, groups, series, colors, fmt){
  const W=640,H=300,pad=44,bw=Math.min(46,(W-2*pad)/groups.length/series.length-8);
  let mx=0; groups.forEach((g,i)=>series.forEach(s=>{mx=Math.max(mx,s.v[i]);}));
  mx*=1.12;
  const X=i=>pad+i*(W-2*pad)/groups.length+( (W-2*pad)/groups.length-series.length*bw)/2;
  const Y=v=>H-40-(v/mx)*(H-80);
  let h=`<svg viewBox="0 0 ${W} ${H}" width="100%">`;
  for(let g=0;g<=4;g++){const v=mx*g/4,y=Y(v);
    h+=`<line x1="${pad}" y1="${y}" x2="${W-14}" y2="${y}" stroke="#eee"/>`+
       `<text x="${pad-6}" y="${y+4}" font-size="10" text-anchor="end">${fmt(v)}</text>`;}
  series.forEach((s,si)=>{ groups.forEach((g,i)=>{ const x=X(i)+si*bw,y=Y(s.v[i]);
    h+=`<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${bw}" height="${(H-40-y).toFixed(1)}" fill="${colors[si]}" rx="2">`+
       `<title>${s.n} · ${g}: ${fmt(s.v[i])}</title></rect>`; }); });
  groups.forEach((g,i)=>{ h+=`<text x="${(X(i)+series.length*bw/2).toFixed(1)}" y="${H-22}" font-size="11" text-anchor="middle">${g}</text>`;});
  h+=`<g font-size="12">${series.map((s,si)=>`<rect x="${pad+si*190}" y="${H-12}" width="12" height="12" fill="${colors[si]}"/><text x="${pad+14+si*190}" y="${H-2}">${s.n}</text>`).join('')}</g></svg>`;
  el.innerHTML = h;
}

// Forest plot
function forest(el, rows){
  rows=[...rows].sort((a,b)=>a.season<b.season?-1:1);
  const W=680,rh=24,pad=120,H=rows.length*rh+70;
  const xs=rows.flatMap(r=>[r.logit_ci_low,r.logit_ci_high]);
  const mn=Math.min(...xs,-0.1),mx=Math.max(...xs,0.1);
  const X=v=>pad+(v-mn)/(mx-mn)*(W-pad-30);
  let h=`<svg viewBox="0 0 ${W} ${H}" width="100%">`;
  h+=`<line x1="${X(0)}" y1="10" x2="${X(0)}" y2="${H-40}" stroke="#DC2626" stroke-dasharray="5,4"/>`;
  rows.forEach((r,i)=>{ const y=24+i*rh;
    const fb = r.fit_method==='sklearn_l2_fallback';
    h+=`<text x="${pad-8}" y="${y+4}" font-size="11" text-anchor="end">${r.season}</text>`+
      `<line x1="${X(r.logit_ci_low).toFixed(1)}" y1="${y}" x2="${X(r.logit_ci_high).toFixed(1)}" y2="${y}" stroke="#93C5FD" stroke-width="2"/>`+
      `<circle cx="${X(r.logit_weak_foot_coef).toFixed(1)}" cy="${y}" r="4.5" fill="${fb?'#D97706':'#1D4ED8'}">`+
      `<title>${r.season}: coef ${r.logit_weak_foot_coef.toFixed(3)} [${r.logit_ci_low.toFixed(3)}, ${r.logit_ci_high.toFixed(3)}] p=${Number(r.logit_pvalue).toExponential(1)}</title></circle>`;});
  h+=`<text x="${pad}" y="${H-12}" font-size="11">← weak foot hurts · helps →</text>
      <text x="${X(0)}" y="${H-26}" font-size="10" text-anchor="middle">0</text></svg>`;
  el.innerHTML = h;
}

// Tabs
const TABS = ['Weak foot','Calibration','Ablation','Recalibration','Weighting','Calibration map','Player cards'];
$('#tabs').innerHTML = TABS.map((t,i)=>`<button data-t="${i}" class="${i===0?'on':''}">${t}</button>`).join('');
const body = $('#body');
function table(rows, cols){
  return `<div class="tblwrap"><table><tr>${cols.map(c=>`<th>${c[1]}</th>`).join('')}</tr>`+
   rows.map(r=>`<tr>${cols.map(c=>`<td>${typeof c[2]==='function'?c[2](r):r[c[0]]}</td>`).join('')}</tr>`).join('')+`</table></div>`;
}
const f3 = v => Number(v).toFixed(3);

function render(i){
  document.querySelectorAll('#tabs button').forEach((b,j)=>b.classList.toggle('on',j===i));
  if(i===0){ const rows=D.weak;
    body.innerHTML=`<div class="card"><h2>Finding 1 — weak-foot passes complete less often</h2>
    <p class="d">Controlled logistic coefficient per season (95% CI). Negative = weak foot hurts at fixed difficulty. 1973/74 (n=469, incomplete tagging) excluded from headline.</p>
    <div id="f"></div><p class="note">Orange dot would mark sklearn-L2 fallback fits; all seasons here used statsmodels.</p></div>`;
    forest($('#f'), rows.filter(r=>r.season!=='1973/1974')); }
  if(i===1){ const rows=D.cal;
    body.innerHTML=`<div class="card"><h2>Finding 2 — calibration collapses on rare types</h2>
    <p class="d">Pooled held-out predictions. Error orders itself by sample size.</p><div id="f"></div>
    ${table(rows,[['pass_type','Type'],['n','n',r=>Number(r.n).toLocaleString()],['actual_rate','Actual',r=>f3(r.actual_rate)],['brier_score','Brier',r=>f3(r.brier_score)],['ece','ECE',r=>f3(r.ece)]])}</div>`;
    bars($('#f'), rows.map(r=>r.pass_type),
      [{n:'Brier',v:rows.map(r=>r.brier_score)},{n:'ECE',v:rows.map(r=>r.ece)}],
      ['#1D4ED8','#DC2626'], f3); }
  if(i===2){ const rows=D.abl, types=[...new Set(rows.map(r=>r.pass_type))];
    const V=[['no_footedness','No footedness','#9CA3AF'],['current','Current','#1D4ED8'],['full_interactions','Full interactions','#059669']];
    body.innerHTML=`<div class="card"><h2>Finding 3 — footedness helps marginally, more overfits</h2>
    <p class="d">ECE per type (lower better). Full interactions win only Switch.</p><div id="f"></div></div>`;
    bars($('#f'), types, V.map(v=>({n:v[1],v:types.map(t=>rows.find(r=>r.variant===v[0]&&r.pass_type===t).ece)})),
      V.map(v=>v[2]), f3); }
  if(i===3){ const rows=D.recal;
    body.innerHTML=`<div class="card"><h2>Finding 4 — isotonic recalibration diagnoses scarcity</h2>
    <p class="d">Fit on half the held-out preds, evaluated on the other. Succeeds only where n suffices.</p><div id="f"></div>
    ${table(rows,[['pass_type','Type'],['n_eval','n eval',r=>Number(r.n_eval).toLocaleString()],['ece_before','ECE before',r=>f3(r.ece_before)],['ece_after','ECE after',r=>f3(r.ece_after)]])}</div>`;
    bars($('#f'), rows.map(r=>r.pass_type),
      [{n:'Before',v:rows.map(r=>r.ece_before)},{n:'After',v:rows.map(r=>r.ece_after)}],
      ['#DC2626','#059669'], f3); }
  if(i===4){ const rows=D.weight, types=[...new Set(rows.map(r=>r.pass_type))];
    const V=[['unweighted','Unweighted','#1D4ED8'],['inv_type','Inverse-weighted','#DC2626']];
    body.innerHTML=`<div class="card"><h2>Finding 3b — reweighting rare types hurts</h2>
    <p class="d">Negative result: upweighting distorts base rates, working against calibration by construction. Only Switch improves, marginally.</p><div id="f"></div></div>`;
    bars($('#f'), types, V.map(v=>({n:v[1],v:types.map(t=>rows.find(r=>r.variant===v[0]&&r.pass_type===t).ece)})),
      V.map(v=>v[2]), f3); }
  if(i===5){ geoTab(); }
  if(i===6){ playerTab(); }
}
document.querySelector('#tabs').addEventListener('click', e=>{
  if(e.target.dataset.t!==undefined) render(+e.target.dataset.t); });

// Shared pitch markings, StatsBomb 120x80 coordinates (attack left-to-right)
function pitchBase(sx){
  const W=120*sx+20, H=80*sx+20, o=10;
  const X=x=>o+x*sx, Y=y=>o+(80-y)*sx;
  let h=`<svg viewBox="0 0 ${W} ${H}" width="100%">`;
  h+=`<rect x="${o}" y="${o}" width="${120*sx}" height="${80*sx}" fill="#1a6b3c" rx="4"/>`;
  h+=`<g stroke="#ffffff" opacity="0.85" fill="none" stroke-width="1.2">`+
    `<line x1="${X(60)}" y1="${Y(80)}" x2="${X(60)}" y2="${Y(0)}"/>`+
    `<circle cx="${X(60)}" cy="${Y(40)}" r="${10*sx}"/>`+
    `<rect x="${X(102)}" y="${Y(62)}" width="${18*sx}" height="${44*sx}"/>`+
    `<rect x="${X(0)}" y="${Y(62)}" width="${18*sx}" height="${44*sx}"/>`+
    `<rect x="${X(114)}" y="${Y(50)}" width="${6*sx}" height="${20*sx}"/>`+
    `<rect x="${X(0)}" y="${Y(50)}" width="${6*sx}" height="${20*sx}"/>`+
    `<circle cx="${X(108)}" cy="${Y(40)}" r="1.6" fill="#ffffff"/>`+
    `<circle cx="${X(12)}" cy="${Y(40)}" r="1.6" fill="#ffffff"/></g>`;
  return {h, X, Y};
}
function filtPasses(F){
  return D.passes.filter(p=>
    (F.season==='all'||p.s===F.season)&&(F.type==='all'||p.k===F.type)&&
    (F.foot==='all'||p.w===+F.foot));
}

// Calibration map: grid cells colored by actual − expected at pass origin
const NX=12, NY=8, MINCELL=5;
function geoTab(){
  const seasons=[...new Set(D.passes.map(p=>p.s))].sort();
  let F={season:'all',type:'all',foot:'all'};
  body.innerHTML=`<div class="card"><h2>Calibration map — where is the model wrong?</h2>
  <p class="d">Pass origins binned into a ${NX}×${NY} grid. Red = model overestimates completion here, blue = underestimates. Grey = too few passes (&lt;${MINCELL}). Hover a cell for numbers.</p>
  <div class="filters">
    <label>Season<select id="gs">${['all',...seasons].map(s=>`<option>${s}</option>`).join('')}</select></label>
    <label>Type<select id="gt">${['all','Ordinary','Cross','Switch','Through ball'].map(s=>`<option>${s}</option>`).join('')}</select></label>
    <label>Foot<select id="gf"><option value="all">all</option><option value="0">strong</option><option value="1">weak</option></select></label>
    <span id="gcnt"></span>
  </div><div class="pitchwrap" id="gw"></div>
  <p class="note">Ordinary passes should look flat and grey-green; rare types show red/blue patches — the scarcity pattern, drawn on grass.</p></div>`;
  const draw=()=>{
    const rows=filtPasses(F);
    const cells={};
    rows.forEach(p=>{
      const cx=Math.min(NX-1,Math.floor(p.sx/120*NX)), cy=Math.min(NY-1,Math.floor(p.sy/80*NY));
      const k=cx+'_'+cy;
      cells[k]=cells[k]||{n:0,a:0,x:0};
      cells[k].n++; cells[k].a+=p.a; cells[k].x+=p.x; });
    const sx=7.2, B=pitchBase(sx);
    let h=B.h;
    for(let cx=0;cx<NX;cx++) for(let cy=0;cy<NY;cy++){
      const c=cells[cx+'_'+cy];
      const x0=cx*120/NX, y0=cy*80/NY, w=120/NX, hh=80/NY;
      if(!c||c.n<MINCELL){
        h+=`<rect x="${B.X(x0)}" y="${B.Y(y0+hh)}" width="${w*sx}" height="${hh*sx}" fill="#3d7a52" opacity="0.45"/>`;
        continue; }
      const gap=c.a/c.n-c.x/c.n, t=Math.min(1,Math.abs(gap)/0.25);
      const col=gap<0?`rgba(220,38,38,${(0.25+0.65*t).toFixed(2)})`:`rgba(96,165,250,${(0.25+0.65*t).toFixed(2)})`;
      h+=`<rect data-c="${cx}_${cy}" x="${B.X(x0)}" y="${B.Y(y0+hh)}" width="${w*sx}" height="${hh*sx}" fill="${col}" stroke="#ffffff" stroke-opacity="0.4" stroke-width="0.8">`+
        `<title>cell: n=${c.n} actual=${(c.a/c.n).toFixed(3)} xPass=${(c.x/c.n).toFixed(3)} gap=${(gap*100).toFixed(1)}pp</title></rect>`+
        `<text x="${B.X(x0+w/2)}" y="${B.Y(y0+hh/2)+4}" font-size="9" text-anchor="middle" fill="#fff">${(gap*100).toFixed(0)}</text>`; }
    h+='</svg>';
    $('#gw').innerHTML=h;
    $('#gcnt').textContent=`${rows.length.toLocaleString()} passes in filter`;
    $('#gw').querySelectorAll('[data-c]').forEach(el=>{
      el.addEventListener('mouseenter', e=>{
        const c=cells[el.dataset.c], gap=c.a/c.n-c.x/c.n;
        showTip(e,`n=${c.n}<br>actual ${(c.a/c.n).toFixed(3)}<br>xPass ${(c.x/c.n).toFixed(3)}<br>gap ${(gap*100).toFixed(1)}pp`);});
      el.addEventListener('mouseleave', hideTip); });
  };
  $('#gs').onchange=e=>{F.season=e.target.value;draw();};
  $('#gt').onchange=e=>{F.type=e.target.value;draw();};
  $('#gf').onchange=e=>{F.foot=e.target.value;draw();};
  draw();
}

// Player cards: top passers in the sample, actual vs expected
function playerTab(){
  const byName={};
  D.passes.forEach(p=>{ (byName[p.p]=byName[p.p]||[]).push(p); });
  const top=Object.entries(byName).sort((a,b)=>b[1].length-a[1].length).slice(0,12);
  body.innerHTML=`<div class="card"><h2>Player cards — who beats their xPass?</h2>
  <p class="d">Sampled held-out passes per player. Trust the actual−xPass gap; raw rates are tilted because the sample oversamples rare types.</p>
  <div class="filters"><label>Player<select id="ps">${top.map(([n,ps])=>`<option value="${n}">${n} (${ps[0].t}, n=${ps.length})</option>`).join('')}</select></label>
  <span id="pcnt"></span></div>
  <div id="pcard"></div><div class="pitchwrap" id="ppw"></div></div>`;
  const draw=()=>{
    const name=$('#ps').value, ps=byName[name].slice(0,120);
    const agg=a=>({n:a.length, ar:a.reduce((s,p)=>s+p.a,0)/a.length, xr:a.reduce((s,p)=>s+p.x,0)/a.length});
    const o=agg(ps);
    const types=['Ordinary','Cross','Switch','Through ball'].map(t=>({t, ...agg(ps.filter(p=>p.k===t))})).filter(r=>r.n>0);
    const feet=[['Strong',ps.filter(p=>p.w===0)],['Weak',ps.filter(p=>p.w===1)]].map(([lab,a])=>({lab, ...agg(a)})).filter(r=>r.n>0);
    $('#pcard').innerHTML=
      `<p style="font-size:15px"><b>${name}</b> · ${ps[0].t} · n=${o.n} · actual ${(o.ar*100).toFixed(1)}% vs xPass ${(o.xr*100).toFixed(1)}% · <b>gap ${((o.ar-o.xr)*100).toFixed(1)}pp</b></p>`+
      table([{pass_type:'—',n:o.n,actual_rate:o.ar,brier_score:o.xr}].concat(
        types.map(r=>({pass_type:r.t,n:r.n,actual_rate:r.ar,brier_score:r.xr})),
        feet.map(r=>({pass_type:r.lab+' foot',n:r.n,actual_rate:r.ar,brier_score:r.xr}))),
        [['pass_type','Split'],['n','n'],['actual_rate','Actual',r=>r.actual_rate.toFixed(3)],['brier_score','xPass',r=>r.brier_score.toFixed(3)]]);
    const sx=6.4, B=pitchBase(sx);
    let h=B.h;
    ps.forEach((p,j)=>{
      const col=p.a?'#4ADE80':'#FCA5A5';
      h+=`<line data-j="${j}" x1="${B.X(p.sx)}" y1="${B.Y(p.sy)}" x2="${B.X(p.ex)}" y2="${B.Y(p.ey)}" stroke="${col}" stroke-width="1.5" opacity="0.75" stroke-linecap="round"/>`; });
    h+='</svg>';
    $('#ppw').innerHTML=h;
    CUR=ps;
    $('#ppw').querySelectorAll('[data-j]').forEach(el=>{
      el.addEventListener('mouseenter', e=>{ const p=CUR[+el.dataset.j];
        showTip(e,`${p.k} · ${p.w<0?'n/a':(p.w?'weak':'strong')} foot<br>xPass ${p.x} · ${p.a?'completed':'missed'} · ${p.l}m`);});
      el.addEventListener('mouseleave', hideTip); });
  };
  $('#ps').onchange=draw;
  draw();
}
render(0);
</script></body></html>
"""

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ids", nargs="*", default=["90", "42"],
                    help="Season cache ids for the pitch sample.")
    args = ap.parse_args()

    data = {
        "weak": load_csv("weak_foot_hypothesis_by_season.csv"),
        "cal": load_csv("calibration_summary_by_pass_type.csv"),
        "abl": load_csv("ablation_summary.csv"),
        "recal": load_csv("recalibration_summary.csv"),
        "weight": load_csv("weighted_summary.csv"),
        "passes": sample_pitch_passes(args.ids),
    }
    print(f"Pitch sample: {len(data['passes']):,} passes")
    html = HTML.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    OUT.write_text(html)
    print(f"Saved: {OUT.resolve()} ({OUT.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
