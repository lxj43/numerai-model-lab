'use strict';
const $=id=>document.getElementById(id);
const state={summary:null,mode:'daily',original:false,enabled:new Set(['current_only','three_weekly_snapshots']),rows:[],filtered:[],page:0,pageSize:12,sort:0,ascending:true,request:0};
const fmt=(n,d=5)=>Number(n).toFixed(d),num=n=>Number(n).toLocaleString('zh-CN');
const escapeHTML=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const stats=m=>({mean:m.series.reduce((s,x)=>s+x.ic,0)/m.series.length,positive:m.series.filter(x=>x.ic>0).length,n:m.series.length});
async function getJSON(path){const r=await fetch(path);if(!r.ok)throw Error(`无法读取 ${path}（HTTP ${r.status}）`);return r.json();}
function showError(e){$('error').hidden=false;$('error').textContent=`${e.message}。请通过本地 HTTP 服务打开页面，或重新运行 build_dashboard.py 更新数据。`;}
function visibleModels(){return state.summary.models.filter(m=>m.id!=='original'||state.original);}
function rankedModels(){return state.summary.models.filter(m=>m.id!=='original').slice().sort((a,b)=>stats(b).mean-stats(a).mean);}
function renderCards(){
 const best=rankedModels()[0],baseline=state.summary.models.find(m=>m.id==='current_only'),sequence=state.summary.models.find(m=>m.id==='three_weekly_snapshots');
 const cards=[{m:best,tag:'本次样本最高'},{m:baseline,tag:'单时点基准'},{m:sequence,tag:'时序对照'}];
 $('cards').classList.add('three');
 $('cards').innerHTML=cards.map(({m,tag})=>{const s=stats(m);return `<article class="card" style="--color:${m.color}"><div class="card-header"><span class="dot"></span>${m.name}<span class="tag">${tag}</span></div><div class="card-sub">${m.family} · ${m.features} 列输入</div><div class="metric-label">平均 Rank IC</div><div class="metric">${fmt(s.mean)}</div><div class="card-stats"><div>正相关日期<strong>${s.positive} / ${s.n}</strong></div><div>验证样本<strong>${num(m.validRows)}</strong></div><div>训练样本<strong>${num(m.trainRows)}</strong></div></div><div class="card-foot">验证 ${m.validDates.join(' — ')}</div></article>`}).join('');
 $('legend').innerHTML=visibleModels().map(m=>`<label><input type="checkbox" data-model="${m.id}" ${state.enabled.has(m.id)?'checked':''} style="accent-color:${m.color}">${m.name}</label>`).join('');
 document.querySelectorAll('[data-model]').forEach(el=>el.addEventListener('change',()=>{el.checked?state.enabled.add(el.dataset.model):state.enabled.delete(el.dataset.model);drawChart();}));
 renderLeaderboard();
}
function renderLeaderboard(){
 const models=rankedModels();if(state.original)models.push(state.summary.models.find(m=>m.id==='original'));
 $('leaderRows').innerHTML=models.map((m,i)=>{const s=stats(m),mean=s.mean,std=Math.sqrt(m.series.reduce((a,p)=>a+(p.ic-mean)**2,0)/(s.n-1));return `<tr class="${m.id==='original'?'reference-row':i===0?'best-row':''}"><td><button class="model-link" data-method="${m.id}"><span class="rank-number ${i===0?'first':''}">${m.id==='original'?'参':i+1}</span>${m.name}<span class="leader-sub">${m.id==='original'?'不同样本，不参与排名':m.description}</span></button></td><td>${m.family}</td><td>${m.features}</td><td><strong>${fmt(mean)}</strong></td><td>${fmt(std)}</td><td>${s.positive}/${s.n}</td><td>${m.seconds===null?'复用原结果':fmt(m.seconds,1)+' 秒'}</td></tr>`;}).join('');
 document.querySelectorAll('[data-method]').forEach(el=>el.addEventListener('click',()=>{$('methodSelect').value=el.dataset.method;renderMethod();$('methodPanel').scrollIntoView({behavior:'smooth',block:'start'});}));
}
function renderMethod(){
 const m=state.summary.models.find(m=>m.id===$('methodSelect').value);if(!m)return;
 const prep=['ridge','elastic_net','mlp'].includes(m.id)?'训练集拟合中位数填补与标准化，再应用于验证集。':['random_forest','extra_trees'].includes(m.id)?'缺失值填补的中位数仅由训练集计算。':'使用数值特征；不将股票标识或日期作为模型输入。';
 $('methodDetails').innerHTML=`<p class="method-intro">${escapeHTML(m.description)}</p><p>${m.features} 列输入 · ${escapeHTML(m.family)} · ${prep}</p><div class="param-list">${Object.entries(m.params||{}).map(([k,v])=>`<span class="param-item">${escapeHTML(k)} = ${escapeHTML(JSON.stringify(v))}</span>`).join('')}</div>${m.warnings?.length?`<p class="method-warning">运行提示：${m.warnings.map(w=>escapeHTML(w.includes("hasn't converged")?'已达到预设迭代上限，优化尚未收敛':w)).join('；')}。此处保留实际结果，没有按验证分数追加训练。</p>`:''}<p class="note">${m.id==='original'?'该实验样本不同，仅作参考。':'使用全部共同样本；预先固定参数，不根据本次验证分数调优。'}${m.actualIterations?' 实际迭代次数：'+m.actualIterations+'。':''} 这是预测实验，尚未构造买卖组合和资金净值。</p>`;
}
function renderPredictionHead(){
 const a=state.summary.models.find(m=>m.id===$('modelA').value),b=state.summary.models.find(m=>m.id===$('modelB').value);
 $('predictionHead').innerHTML=`<tr>${[[0,'股票标识'],[1,'真实标签'],[2,a.name+' 输出'],[4,a.name+' 排序'],[3,b.name+' 输出'],[5,b.name+' 排序']].map(([i,title])=>`<th><button data-sort="${i}">${title} ↕</button></th>`).join('')}</tr>`;
 document.querySelectorAll('[data-sort]').forEach(button=>button.addEventListener('click',()=>{const col=+button.dataset.sort;state.ascending=state.sort===col?!state.ascending:col===0;state.sort=col;filterRows();}));
}
function drawChart(){
 const models=visibleModels().filter(m=>state.enabled.has(m.id));
 const start=$('startDate').value;
 const dates=[...new Set(visibleModels().flatMap(m=>m.series.map(x=>x.date)))].filter(d=>d>=start).sort();
 if(!dates.length)return;
 const lines=models.map(m=>{let sum=0;return {...m,points:m.series.filter(p=>p.date>=start).map(p=>({...p,value:state.mode==='daily'?p.ic:(sum+=p.ic)}))};});
 const values=lines.flatMap(m=>m.points.map(p=>p.value));
 let lo=Math.min(0,...values),hi=Math.max(0,...values);if(hi-lo<.01){hi+=.01;lo-=.01;}const padding=(hi-lo)*.12;lo-=padding;hi+=padding;
 const W=1120,H=310,L=62,R=22,T=20,B=40;
 const x=d=>L+(dates.length===1?.5:dates.indexOf(d)/(dates.length-1))*(W-L-R);
 const y=v=>T+(hi-v)/(hi-lo)*(H-T-B);
 let markup=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${state.mode==='daily'?'逐期':'累计'}Rank IC对比曲线"><title>历史验证 Rank IC，不代表投资收益</title>`;
 for(let i=0;i<=4;i++){const v=lo+(hi-lo)*i/4;markup+=`<line x1="${L}" y1="${y(v)}" x2="${W-R}" y2="${y(v)}" stroke="#edf0f4"/><text x="${L-12}" y="${y(v)+4}" text-anchor="end">${fmt(v,3)}</text>`;}
 markup+=`<line x1="${L}" y1="${y(0)}" x2="${W-R}" y2="${y(0)}" stroke="#a5b0bf" stroke-dasharray="4 4"/>`;
 const tickCount=Math.min(6,dates.length);
 for(let i=0;i<tickCount;i++){const d=dates[Math.round(i*(dates.length-1)/Math.max(1,tickCount-1))];markup+=`<text x="${x(d)}" y="${H-9}" text-anchor="middle">${d.slice(5)}</text>`;}
 lines.forEach(m=>{markup+=`<path d="${m.points.map((p,i)=>`${i?'L':'M'}${x(p.date).toFixed(2)},${y(p.value).toFixed(2)}`).join(' ')}" stroke="${m.color}" fill="none" stroke-width="2.6" stroke-linejoin="round"/>`;m.points.forEach(p=>{markup+=`<circle cx="${x(p.date)}" cy="${y(p.value)}" r="3" fill="white" stroke="${m.color}" stroke-width="1.8"/>`;});});
 dates.forEach((date,i)=>{const step=(W-L-R)/Math.max(1,dates.length-1);markup+=`<rect data-date="${date}" x="${Math.max(L,x(date)-step/2)}" y="${T}" width="${dates.length===1?W-L-R:step}" height="${H-T-B}" fill="transparent" style="cursor:pointer"><title>${date}${lines.map(m=>{const p=m.points.find(p=>p.date===date);return p?` | ${m.name}: ${fmt(p.value)}`:''}).join('')}</title></rect>`;});
 markup+='</svg>';$('chart').innerHTML=markup;
 $('chartSubtitle').textContent=`${dates[0]} — ${dates.at(-1)} · ${state.mode==='daily'?'每个日期分别计算 Spearman 排名相关系数':'从所选起始日期累计各期 Rank IC'}${state.original?' · 灰色基线使用不同样本':''}`;
 $('chart').querySelectorAll('[data-date]').forEach(el=>{const inspect=()=>{$('chartTip').textContent=el.dataset.date+'　'+lines.map(m=>{const p=m.points.find(p=>p.date===el.dataset.date);return p?`${m.name} ${fmt(p.value)}`:null}).filter(Boolean).join('　 /　');};el.addEventListener('pointerenter',inspect);el.addEventListener('click',()=>{inspect();if(state.summary.dates.includes(el.dataset.date)){$('detailDate').value=el.dataset.date;loadRows();}});});
}
async function loadRows(){
 const request=++state.request;const date=$('detailDate').value;
 $('rows').innerHTML='<tr><td colspan="6">正在读取预测明细…</td></tr>';
 try{const rows=await getJSON(`data/${date}.json`);if(request!==state.request)return;state.rows=rows;state.page=0;filterRows();}catch(e){if(request!==state.request)return;$('rows').innerHTML='<tr><td colspan="6">读取失败，请检查本地服务和数据文件。</td></tr>';showError(e);}
}
function filterRows(){
 const ids=state.summary.predictionModels,n=ids.length,ai=ids.indexOf($('modelA').value),bi=ids.indexOf($('modelB').value);
 const q=$('search').value.trim().toUpperCase();
 state.filtered=state.rows.filter(r=>r[0].toUpperCase().includes(q)).map(r=>[r[0],r[1],r[2+ai],r[2+bi],r[2+n+ai],r[2+n+bi]]);
 const s=state.sort,sign=state.ascending?1:-1;state.filtered.sort((a,b)=>sign*(s===0?a[0].localeCompare(b[0]):a[s]-b[s]));state.page=0;renderRows();
}
function rank(v,color){return `<span class="rank"><span class="bar ${color}"><i style="width:${v*100}%"></i></span>${(v*100).toFixed(1)}%</span>`;}
function renderRows(){const total=state.filtered.length,pages=Math.max(1,Math.ceil(total/state.pageSize));state.page=Math.max(0,Math.min(pages-1,state.page));const rows=state.filtered.slice(state.page*state.pageSize,(state.page+1)*state.pageSize);$('rows').innerHTML=rows.length?rows.map(r=>`<tr><td>${escapeHTML(r[0])}</td><td>${fmt(r[1],2)}</td><td>${fmt(r[2],6)}</td><td>${rank(r[4],'')}</td><td>${fmt(r[3],6)}</td><td>${rank(r[5],'purple')}</td></tr>`).join(''):'<tr><td colspan="6">没有匹配的股票</td></tr>';$('rowCount').textContent=`${num(total)} 条匹配 / 当期 ${num(state.rows.length)} 只股票`;$('pageInfo').textContent=`${state.page+1} / ${pages}`;$('prev').disabled=state.page===0;$('next').disabled=state.page>=pages-1;document.querySelectorAll('[data-sort]').forEach(b=>{b.parentElement.setAttribute('aria-sort',+b.dataset.sort===state.sort?(state.ascending?'ascending':'descending'):'none');});}
function exportMetrics(){const rows=[['实验','特征数','训练样本','验证样本','日期数','平均RankIC','正相关日期数','验证开始','验证结束'],...state.summary.models.map(m=>{const s=stats(m);return[m.name,m.features,m.trainRows,m.validRows,s.n,s.mean,s.positive,...m.validDates]})];const csv='\uFEFF'+rows.map(r=>r.map(v=>'"'+String(v).replaceAll('"','""')+'"').join(',')).join('\r\n');const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download='quant-lab-experiments.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function renderAgent(){
 const a=state.summary.agent;if(!a)return;
 const selected=a.selection,test=a.test,dev=a.development;
 $('agentResearch').hidden=false;
 $('agentModel').textContent=`实际提案模型：${selected.llm} · 本地执行训练与评分 · 历史实验记录`;
 $('agentStatus').textContent=test.target_reached?'后续测试达到目标':'开发达到目标 · 后续测试未达标';
 const entries=[['开发集最佳',selected.development_metric,'2019年，参与方案选择'],['冻结方案的后续测试',test.selected_agent.mean_rank_ic,test.dates.join(' — ')],['同一测试期的固定基线',test.baseline.mean_rank_ic,'单时点 XGBoost，固定原参数']];
 $('agentMetrics').innerHTML=entries.map(([title,value,note])=>`<div class="agent-metric"><div class="agent-stage-label">${title}</div><div class="metric">${fmt(value)}</div><small>${note}</small></div>`).join('');
 $('agentProtocol').textContent=`目标为平均 Rank IC ≥ 0.02。后续测试共 ${num(test.rows)} 条记录、${test.selected_agent.n_dates} 个日期；方案在 ${selected.freeze_time.replace('T',' ')} 冻结后评估。测试结果未回传给提案模型，也未据此重新调参。这里的测试仍为历史数据，且60日标签重叠；不代表未来或实盘表现。`;
 const modes={current:'当前100列',history:'当前 + 两个历史快照（300列）',delta:'当前 + 7天/14天差分（300列）',smooth:'当前 + 三期均值（200列）'};
 if(state.summary.officialBenchmark){
  const official=state.summary.officialBenchmark.periods;
  const localBase=state.summary.models.find(m=>m.id==='current_only');
  const comparison=[['官方 lgbm_jupiter_60',official.development.mean_rank_ic,official.later_test.mean_rank_ic],['本地单时点 XGBoost',stats(localBase).mean,test.baseline.mean_rank_ic],['冻结的模型建议方案',selected.development_metric,test.selected_agent.mean_rank_ic]];
  $('officialComparison').innerHTML='<h2>与官方预测，在相同样本上比较</h2><div class="table-wrap"><table><thead><tr><th>方案</th><th>2019年开发期 Rank IC</th><th>2020年后续测试 Rank IC</th></tr></thead><tbody>'+comparison.map(r=>`<tr><td>${r[0]}</td><td>${fmt(r[1])}</td><td>${fmt(r[2])}</td></tr>`).join('')+'</tbody></table></div><p class="note">官方一行来自已发布的 v6 预测文件，按相同日期和股票对齐后在本地重算；不是官网 NCORR/NMMC。评估样本和指标相同，不代表训练历史、特征数与计算预算相同。</p>';
 }
 $('agentRows').innerHTML=dev.records.map(r=>`<tr><td>第${r.round}轮 · ${escapeHTML(r.recipe.name)}</td><td>${r.recipe.family==='ensemble'?'各成员当期排名加权':modes[r.recipe.feature_mode]}</td><td>${fmt(r.mean_rank_ic)}</td><td>${r.id===selected.selected_id?'已冻结并测试':'仅开发集评估'}</td></tr>`).join('');
 $('agentAudit').innerHTML='<p class="agent-warning">大模型解释仅是研究假设，具体配置决定实际执行。'+(a.audit.issues || []).map(x=>escapeHTML(x.id+'：'+x.explanation)).join('<br>')+'</p>';
 const names={lightgbm:'原始 LightGBM',ridge:'原始 Ridge'};dev.records.forEach(r=>names[r.id]=r.recipe.name);
 const recipe = selected.recipe;
 $('agentRecipe').innerHTML = recipe.members?.length
  ? '<p>方案成员及权重</p><ul>'+recipe.members.map(m=>`<li>${escapeHTML(names[m.id]||m.id)}：${(m.weight*100).toFixed(0)}%</li>`).join('')+'</ul>'
  : '<p>'+escapeHTML(recipe.family)+' · '+escapeHTML(JSON.stringify(recipe.params))+'</p>';
 $('agentSources').innerHTML = '<ul>'+(a.knowledge.protocol_rules || []).map(rule=>'<li>'+escapeHTML(rule)+'</li>').join('')+'</ul>';
}

async function main(){
 state.summary=await getJSON('data/summary.json');
 $('dataOrigin').textContent=state.summary.synthetic?'合成演示':'Numerai '+state.summary.version;
 if(state.summary.schema!==2)throw Error('页面数据尚未更新到十方案版本');
 const base=state.summary.models.find(m=>m.id==='current_only'),best=rankedModels()[0],difference=stats(best).mean-stats(base).mean;
 state.enabled=new Set([base.id,'three_weekly_snapshots',...rankedModels().slice(0,2).map(m=>m.id)]);
 $('finding').textContent=`本次样本中，${best.name} 的平均 Rank IC 最高：${fmt(stats(best).mean)}；比单时点 XGBoost 高 ${fmt(difference)}`;
 $('snapshot').textContent=`快照构建 ${state.summary.builtAt.slice(0,10)}`;
 const allDates=[...new Set(state.summary.models.flatMap(m=>m.series.map(p=>p.date)))].sort();
 $('startDate').innerHTML=allDates.map(d=>`<option>${d}</option>`).join('');$('startDate').value=state.summary.dates[0];
 $('detailDate').innerHTML=state.summary.dates.map(d=>`<option>${d}</option>`).join('');
 $('trainPeriod').textContent=base.trainDates.join(' — ');$('validPeriod').textContent=base.validDates.join(' — ');
 const options=state.summary.models.filter(m=>m.id!=='original').map(m=>`<option value="${m.id}">${m.name}</option>`).join('');
 $('modelA').innerHTML=options;$('modelB').innerHTML=options;$('modelA').value=base.id;$('modelB').value=best.id===base.id?'three_weekly_snapshots':best.id;
 $('methodSelect').innerHTML=state.summary.models.map(m=>`<option value="${m.id}">${m.name}</option>`).join('');
 $('methodSelect').addEventListener('change',renderMethod);
 for(const id of ['modelA','modelB'])$(id).addEventListener('change',()=>{renderPredictionHead();filterRows();});
 const live=state.summary.live;$('liveText').textContent=state.summary.synthetic?`合成演示快照 · ${live.date} · ${num(live.rows)} 个虚构标的。`:`保存快照：第 ${live.round} 轮，数据日期 ${live.date}，共 ${num(live.rows)} 只股票。`;
 $('showOriginal').addEventListener('change',e=>{state.original=e.target.checked;if(state.original)state.enabled.add('original');else state.enabled.delete('original');renderCards();drawChart();});
 document.querySelectorAll('[data-mode]').forEach(b=>b.addEventListener('click',()=>{state.mode=b.dataset.mode;document.querySelectorAll('[data-mode]').forEach(x=>x.classList.toggle('active',x===b));drawChart();}));
 $('startDate').addEventListener('change',drawChart);$('detailDate').addEventListener('change',loadRows);$('search').addEventListener('input',filterRows);$('prev').addEventListener('click',()=>{state.page--;renderRows();});$('next').addEventListener('click',()=>{state.page++;renderRows();});$('export').addEventListener('click',exportMetrics);
 renderPredictionHead();renderCards();renderMethod();renderAgent();drawChart();await loadRows();
}
main().catch(showError);
