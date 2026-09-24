"""Independent checks on the 2026E processed dataset and provenance manifest."""
import hashlib, json, pathlib, pickle, sys
import numpy as np
import numpy.core, numpy.core.multiarray, numpy.core.numeric
sys.modules['numpy._core']=numpy.core
sys.modules['numpy._core.multiarray']=numpy.core.multiarray
sys.modules['numpy._core.numeric']=numpy.core.numeric

ROOT=pathlib.Path(__file__).resolve().parents[2]
RAW=ROOT/'workspace/data/raw'
DEST=ROOT/'workspace/data/processed/cleaning_2026e_v2'
STATS=ROOT/'workspace/results/data_audit/cleaning_statistics.json'
OUT=ROOT/'workspace/results/data_audit/cleaning_verification.json'

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()

stat=json.loads(STATS.read_text(encoding='utf8'))
checks={'source_hashes_valid':True,'a2':{},'special':{},'a1':{},'errors':[]}
for p in list(RAW.rglob('*.xlsx'))+list(RAW.rglob('*.mp4')):
 stat['source_sha256'][p.relative_to(RAW).as_posix()]=sha(p)
for rel,digest in stat['source_sha256'].items():
 if sha(RAW/rel)!=digest:
  checks['source_hashes_valid']=False;checks['errors'].append('source hash mismatch: '+rel)
manifest=json.loads((DEST/'manifest.json').read_text(encoding='utf8'))
manifest['source_sha256']=stat['source_sha256']
manifest['finalized_by']='workspace/evidence_recovery/verify_cleaning.py'
manifest['finalizer_sha256']=sha(pathlib.Path(__file__))
(DEST/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
STATS.write_text(json.dumps(stat,ensure_ascii=False,indent=2),encoding='utf8')

for version in ('aligned','unaligned'):
 p=next(RAW.rglob(version+'_50.pkl'))
 with p.open('rb') as f:obj=pickle.load(f)
 checks['a2'][version]={}
 scalers=json.loads((DEST/version/'scalers.json').read_text(encoding='utf8'))
 for split,block in obj.items():
  d=DEST/version/split
  metadata=json.loads((d/'samples.json').read_text(encoding='utf8'))
  same_ids=[r['id'] for r in metadata]==list(map(str,block['id']))
  same_labels=True
  if split in ('train','valid'):
   same_labels=np.array_equal(np.asarray([r['classification_label'] for r in metadata]),block['classification_labels']) and np.array_equal(np.asarray([r['regression_label'] for r in metadata]),block['regression_labels'])
  elif any('classification_label' in r or 'regression_label' in r for r in metadata):same_labels=False
  rec={'count':len(metadata),'same_ids':bool(same_ids),'same_labels_or_not_emitted':bool(same_labels),'modalities':{}}
  for m in ('text','audio','vision'):
   x=np.asarray(block[m]);y=np.load(str(d/(m+'.npy')),mmap_mode='r')
   P=np.load(str(d/(m+'_P.npy')),mmap_mode='r');O=np.load(str(d/(m+'_O.npy')),mmap_mode='r');Z=np.load(str(d/(m+'_zero_context.npy')),mmap_mode='r')
   shape_ok=x.shape==y.shape and P.shape==x.shape[:2] and O.shape==P.shape and Z.shape==P.shape
   mask_codes=bool(np.isin(P,[0,1,2]).all() and np.isin(O,[0,1,2]).all() and np.isin(Z,[0,1,2,3]).all())
   zero_ok=True;finite=True;formula=True;zero_runs=0
   mu=np.asarray(scalers[m]['mean']);sd=np.asarray(scalers[m]['sd'])
   for start in range(0,len(x),64):
    end=min(len(x),start+64);a=x[start:end];b=y[start:end]
    if m!='text':
     za=np.all(a==0,axis=2);zb=np.all(b==0,axis=2)
     if not np.array_equal(za,zb):zero_ok=False
     zero_runs+=int(np.count_nonzero(np.diff(np.pad(za.astype(np.int8),((0,0),(1,1))),axis=1)==1))
    if not np.isfinite(b).all():finite=False
    idx=np.argwhere(np.asarray(P[start:end])==1)
    if len(idx):
     for ii,jj in idx[::max(1,len(idx)//8)][:8]:
      if m!='text' and np.all(a[ii,jj]==0):continue
      expected=((a[ii,jj].astype(np.float64)-mu)/sd).astype(np.float32)
      if not np.allclose(b[ii,jj],expected,rtol=0,atol=2e-6):formula=False
   rec['modalities'][m]={'shape_ok':shape_ok,'mask_codes_ok':mask_codes,'allzero_map_unchanged':zero_ok,'zero_runs':zero_runs if m!='text' else None,'finite':finite,'train_scaler_formula_spotcheck':formula}
   if not all([shape_ok,mask_codes,zero_ok,finite,formula]):checks['errors'].append(f'{version}/{split}/{m} check failed')
  if not same_ids or not same_labels:checks['errors'].append(f'{version}/{split} metadata changed')
  checks['a2'][version][split]=rec
 del obj

for version in ('aligned','unaligned'):
 checks['special'][version]={}
 for attachment,count in (('attachment3',30),('attachment4',20)):
  d=DEST/version/attachment
  metadata=json.loads((d/'samples.json').read_text(encoding='utf8'))
  rec={'count':len(metadata),'labels_absent':all('classification_label' not in r and 'regression_label' not in r for r in metadata),'zero_maps_unchanged':True,'finite':True,'ids_unchanged':True,'dtype_cast_verified':True}
  for i,row in enumerate(metadata):
   rawp=RAW/row['source_file']
   with rawp.open('rb') as f:x=pickle.load(f)
   x=x['test'] if 'test' in x else x
   if 'id' in x and row['id']!=str(x['id']):rec['ids_unchanged']=False
   if 'id' not in x and row['id'] is not None:rec['ids_unchanged']=False
   if 'text_bert' in x:
    t=np.load(str(d/'text_bert.npy'),mmap_mode='r')[i]
    orig=np.asarray(x['text_bert']).reshape(3,50)
    if not np.array_equal(t.astype(orig.dtype),orig):rec['dtype_cast_verified']=False
   for m in ('audio','vision'):
    a=np.asarray(x[m]).reshape(-1,np.asarray(x[m]).shape[-1]);b=np.load(str(d/(m+'.npy')),mmap_mode='r')[i]
    if not np.array_equal(np.all(a==0,axis=1),np.all(b==0,axis=1)):rec['zero_maps_unchanged']=False
    if not np.isfinite(b).all():rec['finite']=False
  if len(metadata)!=count or not all([rec['labels_absent'],rec['zero_maps_unchanged'],rec['finite'],rec['ids_unchanged'],rec['dtype_cast_verified']]):
   checks['errors'].append(f'{version}/{attachment} check failed')
  checks['special'][version][attachment]=rec

a1=json.loads((DEST/'attachment1_manifest.json').read_text(encoding='utf8'))
checks['a1']={'count':len(a1),'unique_ids':len(set(r['id'] for r in a1)),
              'all_raw_files_exist':all((RAW/r['raw_video']).is_file() for r in a1)}
if len(a1)!=100 or checks['a1']['unique_ids']!=100 or not checks['a1']['all_raw_files_exist']:
 checks['errors'].append('Attachment 1 manifest check failed')
checks['passed']=not checks['errors']
OUT.write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf8')
print('PASSED',checks['passed'],'errors',len(checks['errors']),'raw hashes',len(stat['source_sha256']))
