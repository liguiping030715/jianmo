"""Read-only Stage 2 evidence collector. Outputs are outside data/raw."""
import csv, hashlib, json, pathlib, pickle, sys, zipfile, xml.etree.ElementTree as ET
from collections import Counter, defaultdict
import numpy as np
import numpy.core
import numpy.core.multiarray
import numpy.core.numeric
import cv2

sys.modules['numpy._core'] = numpy.core  # permit NumPy-2-authored official PKL on NumPy 1.24
sys.modules['numpy._core.multiarray'] = numpy.core.multiarray
sys.modules['numpy._core.numeric'] = numpy.core.numeric
ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW = ROOT / 'workspace/data/raw'
OUT = ROOT / 'workspace/results/data_audit'
AUD = ROOT / 'workspace/data_audit'
OUT.mkdir(parents=True, exist_ok=True)
AUD.mkdir(parents=True, exist_ok=True)

def dump(name, x):
    (OUT/name).write_text(json.dumps(x, ensure_ascii=False, indent=2, default=str), encoding='utf-8')

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def xlsx_rows(p):
    ns={'x':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(p) as z:
        shared=[]
        if 'xl/sharedStrings.xml' in z.namelist():
            root=ET.fromstring(z.read('xl/sharedStrings.xml'))
            shared=[''.join(t.text or '' for t in si.findall('.//x:t',ns)) for si in root.findall('x:si',ns)]
        sheet=next(n for n in z.namelist() if n.startswith('xl/worksheets/sheet') and n.endswith('.xml'))
        root=ET.fromstring(z.read(sheet)); rows=[]
        for row in root.findall('.//x:sheetData/x:row',ns):
            d={}
            for c in row.findall('x:c',ns):
                ref=c.attrib.get('r',''); col=''.join(ch for ch in ref if ch.isalpha())
                v=c.find('x:v',ns)
                if c.attrib.get('t')=='inlineStr': val=''.join(t.text or '' for t in c.findall('.//x:t',ns))
                elif v is None: val=''
                elif c.attrib.get('t')=='s': val=shared[int(v.text)]
                else: val=v.text
                d[col]=val
            rows.append(d)
        if not rows:return []
        hdr=rows[0]
        return [{hdr.get(c,c):v for c,v in row.items()} for row in rows[1:]]

def load(p):
    with p.open('rb') as f:return pickle.load(f)

def video_info(p):
    cap=cv2.VideoCapture(str(p)); fps=cap.get(cv2.CAP_PROP_FPS); frames=cap.get(cv2.CAP_PROP_FRAME_COUNT)
    ok,frame=cap.read(); cap.release()
    return {'fps':fps,'frames':frames,'duration_s':frames/fps if fps else None,'readable':bool(ok),'frame_shape':list(frame.shape) if ok else None}

def field_summary(v):
    a=np.asarray(v); d={'type':type(v).__name__,'dtype':str(a.dtype),'shape':list(a.shape)}
    if a.dtype.kind in 'fiub':
        d.update(nan=int(np.isnan(a).sum()),inf=int(np.isinf(a).sum()),zero=int(np.count_nonzero(a==0)))
        finite=a[np.isfinite(a)]
        if finite.size:d.update(min=float(finite.min()),max=float(finite.max()))
    return d

def runs(mask):
    a=np.r_[False,np.asarray(mask,dtype=bool),False]
    st=np.where(np.diff(a.astype('int8'))==1)[0]; en=np.where(np.diff(a.astype('int8'))==-1)[0]
    return [(int(s),int(e)) for s,e in zip(st,en)]

def row_zero_stats(a,lengths=None):
    a=np.asarray(a); z=np.all(a==0,axis=-1); n,t=z.shape
    if lengths is None: lengths=np.full(n,t,dtype=int)
    lengths=np.atleast_1d(np.asarray(lengths,dtype=int))
    tail_nonzero=0; interior=[]; leading=[]; trailing=[]; zero_valid=0; zero_pad=0
    for i in range(n):
        L=int(lengths[i]); v=z[i,:L]; pad=z[i,L:]
        zero_valid+=int(v.sum());zero_pad+=int(pad.sum());tail_nonzero+=int((~pad).sum())
        rr=runs(v)
        interior += [(i,s,e) for s,e in rr if s>0 and e<L]
        leading += [(i,s,e) for s,e in rr if s==0]
        trailing += [(i,s,e) for s,e in rr if e==L]
    return {'n':n,'t':t,'length_min':int(lengths.min()),'length_max':int(lengths.max()),'length_mean':float(lengths.mean()),'zero_valid':zero_valid,'zero_padding':zero_pad,'tail_nonzero':tail_nonzero,'interior_runs':len(interior),'leading_runs':len(leading),'trailing_runs':len(trailing),'interior_examples':interior[:12],'leading_examples':leading[:8],'trailing_examples':trailing[:8]}

files=sorted(p for p in RAW.rglob('*') if p.is_file())
inventory=[]; xlsx={}; media={}
for p in files:
    rel=p.relative_to(RAW).as_posix(); ext=p.suffix.lower(); d={'path':rel,'filename':p.name,'size_bytes':p.stat().st_size,'format':ext}
    if ext in ('.pkl','.xlsx'):d['sha256']=sha(p)
    if ext=='.xlsx':
        try:
            rr=xlsx_rows(p)
            if p.name=='label.xlsx':
                for row in rr:
                    if row.get('mode')=='test':
                        row.pop('label',None);row.pop('annotation',None)
            xlsx[rel]=rr;d.update(sample_count=len(rr),fields=list(rr[0]) if rr else [])
        except Exception as e:d['read_error']=str(e)
    if ext=='.mp4':
        try:
            v=video_info(p);media[rel]=v;d.update(v)
        except Exception as e:d['read_error']=str(e)
    inventory.append(d)
with (AUD/'file_inventory.csv').open('w',newline='',encoding='utf-8-sig') as f:
    cols=sorted(set().union(*(r.keys() for r in inventory)));w=csv.DictWriter(f,cols);w.writeheader();w.writerows(inventory)
dump('file_inventory.json',inventory);dump('xlsx_rows.json',xlsx);dump('video_metadata.json',media)

schema={};zeros={};ids={};labels={}
for version in ('aligned','unaligned'):
    p=next(RAW.rglob(version+'_50.pkl')); obj=load(p); schema[version]={}; zeros[version]={};ids[version]={};labels[version]={}
    for split,block in obj.items():
        schema[version][split]={k:field_summary(v) for k,v in block.items()}
        ids[version][split]=[str(x) for x in block['id']]
        z={}
        for m in ('text','audio','vision'):
            ll=block.get(m+'_lengths')
            if ll is not None: z[m]=row_zero_stats(block[m],ll)
            else: z[m]=row_zero_stats(block[m])
        zeros[version][split]=z
        if split in ('train','valid'):
            labels[version][split]={'classification':Counter(map(str,block['classification_labels'])),'regression':field_summary(block['regression_labels']),'sign_cross':Counter(str(c)+'|'+str(int(np.sign(r))) for c,r in zip(block['classification_labels'],block['regression_labels'])),'neutral_exact':int(np.count_nonzero(block['regression_labels']==0)),'unique_regression':int(np.unique(block['regression_labels']).size)}
    del obj
dump('pkl_schema.json',schema);dump('zero_statistics.json',zeros);dump('ids.json',ids);dump('label_statistics.json',labels)

special={}; special_ids={};special_zeros={}
for attachment in ('附件3','附件4'):
    special[attachment]={}; special_ids[attachment]={};special_zeros[attachment]={}
    pp=[p for p in files if p.suffix.lower()=='.pkl' and attachment in p.as_posix()]
    for p in pp:
        rel=p.relative_to(RAW).as_posix(); x=load(p)
        if 'test' in x and isinstance(x['test'],dict):x=x['test']
        special[attachment][rel]={k:field_summary(v) for k,v in x.items()}
        special_ids[attachment][rel]=str(x.get('id',''))
        special_zeros[attachment][rel]={}
        for m in ('text','audio','vision'):
            if m not in x:continue
            a=np.asarray(x[m]);a=a[None] if a.ndim==2 else a
            if a.ndim==3:special_zeros[attachment][rel][m]=row_zero_stats(a,x.get(m+'_lengths'))
dump('special_schema.json',special);dump('special_ids.json',special_ids);dump('special_zeros.json',special_zeros)

print('files',len(files),'xlsx',len(xlsx),'videos',len(media),'special_pkl',sum(len(v) for v in special.values()))
print('split_n',{k:{s:len(v) for s,v in d.items()} for k,d in ids.items()})
