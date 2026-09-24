import json,pathlib,collections,statistics
R=pathlib.Path('workspace/results/data_audit'); g=lambda n:json.loads((R/n).read_text(encoding='utf8'))
inv=g('file_inventory.json');xlsx=g('xlsx_rows.json');ids=g('ids.json');s=g('pkl_schema.json');z=g('zero_statistics.json');sp=g('special_schema.json');sz=g('special_zeros.json');si=g('special_ids.json');vm=g('video_metadata.json')
out={}
P=lambda x:x.replace('\\','/').split('/')
v1=[x for x in inv if x['format']=='.mp4' and any('附件1' in a for a in P(x['path']))]
v4=[x for x in inv if x['format']=='.mp4' and any('附件4' in a for a in P(x['path']))]
x1=next(v for k,v in xlsx.items() if 'label-100' in k);x2=next(v for k,v in xlsx.items() if k.endswith('/label.xlsx'))
def id1(row):return str(row['video_id'])+'$_$'+str(row['clip_id'])
def vid(x):return x.split('$_$')[0] if '$_$' in x else None
out['q1']={'label_rows':len(x1),'unique_ids':len(set(map(id1,x1))),'video_files':len(v1),'readable':sum(x['readable'] for x in v1),'dur_min':min(x['duration_s'] for x in v1 if x['duration_s']),'dur_max':max(x['duration_s'] for x in v1 if x['duration_s']),'label_annotations':dict(collections.Counter(r['annotation'] for r in x1))}
v1keys=set((pathlib.Path(x['path']).parent.name,pathlib.Path(x['path']).stem) for x in v1)
x1keys=set((str(r['video_id']),str(r['clip_id'])) for r in x1)
out['q1'].update(missing_videos=sorted(x1keys-v1keys),unlabeled_videos=sorted(v1keys-x1keys))
out['q4']={'video_files':len(v4),'readable':sum(x['readable'] for x in v4),'duration_min':min(x['duration_s'] for x in v4 if x['duration_s']),'duration_max':max(x['duration_s'] for x in v4 if x['duration_s'])}
out['split']={}
for a in ('train','valid','test'):
 for b in ('train','valid','test'):
  if a>=b:continue
  aa=set(ids['aligned'][a]);bb=set(ids['aligned'][b]);va=set(map(vid,aa));vb=set(map(vid,bb))
  out['split'][a+'-'+b]={'sample_overlap':len(aa&bb),'video_overlap':len(va&vb),'video_examples':sorted(va&vb)[:8]}
for a in ('train','valid','test'):
 out['split'][a+'-q1']={'sample_overlap':len(set(ids['aligned'][a])&set(map(id1,x1))),'video_overlap':len(set(map(vid,ids['aligned'][a]))&set(r['video_id'] for r in x1))}
out['xlsx2']={'rows':len(x2),'modes':dict(collections.Counter(r.get('mode') for r in x2))}
out['special']={}
for at in sp:
 out['special'][at]={}
 for fmt in ('aligned','unaligned'):
  rr=[(k,v) for k,v in sp[at].items() if ('500' in str(v.get('audio',{}).get('shape')))==(fmt=='unaligned')]
  out['special'][at][fmt]={'n':len(rr),'keys':dict(collections.Counter(','.join(sorted(v)) for k,v in rr)),'ids':len(set(si[at].get(k,'') for k,v in rr))}
  for m in ('audio','vision','text'):
   stats=[sz[at][k][m] for k,v in rr if m in sz[at][k]]
   if stats:out['special'][at][fmt][m]={'interior_runs':sum(x['interior_runs'] for x in stats),'leading_runs':sum(x['leading_runs'] for x in stats),'trailing_runs':sum(x['trailing_runs'] for x in stats),'zero_total':sum(x['zero_valid'] for x in stats),'length_min':min(x['length_min'] for x in stats),'length_max':max(x['length_max'] for x in stats)}
(R/'summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(out,ensure_ascii=False,indent=2))
