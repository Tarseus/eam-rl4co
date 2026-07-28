from __future__ import annotations
import json,os
from pathlib import Path
import subprocess,sys,time
run=Path(__file__).resolve().parent
manifest=json.load(open(run/'manifest.json')); src=Path(manifest['source_worktree']); py='/data1/gushengda/anaconda3/envs/rlco1/bin/python'; tasks=json.load(open(run/'tasks.json')); cache=Path(manifest['baseline_cache'])
(run/'launcher.pid').write_text(str(os.getpid())+'\n',encoding='utf-8')
status={'status':'running','pid':os.getpid(),'started_at':time.time(),'stage':'baseline','current':'00_baseline_cache','completed':[]}
def save():
 p=run/'runtime_status.json.tmp'; p.write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8'); os.replace(p,run/'runtime_status.json')
def fail(reason,detail=None):
 status['status']='failed'; status['failure']={'reason':reason,'detail':detail}; save(); print('replay_failed',reason,detail,flush=True); sys.exit(1)
save(); print(f'replay_start run={run} pid={os.getpid()} source={src}',flush=True)
blog=run/'tasks/00_baseline_cache/log.txt'
cmd=[py,'-u',str(run/'generate_baseline.py'),str(src),str(run/'baseline_config.json'),str(run/'operator_whitelist.json')]
print('baseline_task_start',flush=True)
with blog.open('w',encoding='utf-8') as out: rc=subprocess.run(cmd,cwd=str(src),stdout=out,stderr=subprocess.STDOUT,env=os.environ.copy()).returncode
if rc!=0 or not cache.is_file(): fail('baseline_generation_failed',{'returncode':rc,'log':str(blog)})
b=json.load(open(cache)); bp=b.get('per_init',{}); required={'scratch','ckpt_135'}
if not required<=set(bp): fail('baseline_missing_init',sorted(bp))
baseline={k:float(bp[k]['aggregated_objective']) for k in sorted(required)}
status['completed'].append({'name':'00_baseline_cache','objectives':baseline,'log':str(blog)}); status['stage']='candidates'; save(); print('baseline_task_done',baseline,flush=True)
for task in tasks:
 name=task['name']; status['current']=name; status['task_started_at']=time.time(); save(); print(f'task_start name={name} f_id={task["f_id"]}',flush=True)
 sublog=run/'tasks'/name/'subprocess.log'; cmd=[py,'-u',str(src/'PTP/ptp_discovery/run_hf_pair_eval.py'),'--payload',task['payload'],'--result',task['result']]
 with sublog.open('w',encoding='utf-8') as out: rc=subprocess.run(cmd,cwd=str(src),stdout=out,stderr=subprocess.STDOUT,env=os.environ.copy()).returncode
 rp=Path(task['result']); result=json.load(open(rp)) if rp.is_file() else None
 if rc!=0 or not isinstance(result,dict) or not result.get('pair_ok'): fail('candidate_failed',{'name':name,'returncode':rc,'pair_reason':result.get('pair_reason') if isinstance(result,dict) else 'missing_result'})
 pi=(result.get('fitness') or {}).get('per_init') or {}; 
 if not required<=set(pi): fail('candidate_missing_init',{'name':name,'keys':sorted(pi)})
 objectives={k:float(pi[k]['obj_cand']) for k in sorted(required)}; deltas={k:objectives[k]-baseline[k] for k in sorted(required)}; mean=sum(deltas.values())/len(deltas)
 item={'name':name,'f_id':task['f_id'],'objectives':objectives,'deltas':deltas,'mean_delta':mean,'reported_score':result.get('score'),'result':str(rp),'subprocess_log':str(sublog)}
 status['completed'].append(item); save(); print('task_done',json.dumps(item,ensure_ascii=False),flush=True)
metrics={'baseline':baseline,'candidates':status['completed'][1:]}; (run/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8'); status.update({'status':'completed','stage':'done','current':None,'finished_at':time.time()}); save(); print('replay_completed',json.dumps(metrics,ensure_ascii=False),flush=True)
