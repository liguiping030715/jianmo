import pathlib,pickle,json,sys,collections
import numpy as np
import numpy.core,numpy.core.multiarray,numpy.core.numeric
sys.modules['numpy._core']=numpy.core;sys.modules['numpy._core.multiarray']=numpy.core.multiarray;sys.modules['numpy._core.numeric']=numpy.core.numeric
R=pathlib.Path('workspace/data/raw');O=pathlib.Path('workspace/results/data_audit');out={}
for ver in ('aligned','unaligned'):
 d=pickle.load(open(next(R.rglob(ver+'_50.pkl')),'rb'));out[ver]={}
 for split,b in d.items():
  q={};tm=b['text_bert'][:,1,:];q['text_valid']={'min':int(tm.sum(1).min()),'max':int(tm.sum(1).max()),'mean':float(tm.sum(1).mean()),'all_suffix_padding':bool(np.all(np.diff(tm,axis=1)<=0))}
  for m in ('audio','vision'):
   arr=b[m];non=np.any(arr!=0,axis=2);last=np.where(non,np.arange(non.shape[1])[None,:]+1,0).max(axis=1)
   if ver=='unaligned':
    ll=np.asarray(b[m+'_lengths']);q[m]={'length_eq_last':int(np.sum(ll==last)),'length_less_last':int(np.sum(ll<last)),'length_greater_last':int(np.sum(ll>last)),'allzero_samples':int(np.sum(last==0)),'length_mean':float(ll.mean()),'last_nonzero_mean':float(last.mean())}
   else:q[m]={'allzero_samples':int(np.sum(last==0)),'last_nonzero_mean':float(last.mean())}
   vals=arr[non]
   q[m]['observed_value_stats']={'count':int(vals.size),'mean':float(vals.mean()) if vals.size else None,'std':float(vals.std()) if vals.size else None,'min':float(vals.min()) if vals.size else None,'max':float(vals.max()) if vals.size else None}
  out[ver][split]=q
 del d

special=[]
for p in R.rglob('*.pkl'):
 if not any('附件3' in x for x in p.parts):continue
 b=pickle.load(open(p,'rb'))['test'];ver='unaligned' if b['audio'].shape[1]==500 else 'aligned';record={'file':p.name,'version':ver,'modality':{}}
 for m in ('audio','vision'):
  arr=b[m][0];z=np.all(arr==0,axis=1);nz=np.flatnonzero(~z);end=int(nz[-1])+1 if nz.size else 0; interior=[];inrun=False
  for j in range(end):
   if z[j] and not inrun:st=j;inrun=True
   if inrun and (not z[j] or j==end-1):
    en=j if not z[j] else j+1
    if st>0 and en<end:interior.append([st,en])
    inrun=False
  record['modality'][m]={'last_nonzero':end,'interior_zero_runs':interior,'prefix_zero':int(nz[0]) if nz.size else len(z),'suffix_zero':len(z)-end}
 special.append(record)
out['special_gaps']=special
(O/'deep_checks.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
print('wrote deep_checks')
