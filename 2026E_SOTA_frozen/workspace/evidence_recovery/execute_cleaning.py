"""Execute the frozen 2026E minimal cleaning contract, without altering raw files."""
from __future__ import annotations

import csv
import hashlib
import json
import pathlib
import pickle
import sys
from collections import Counter

import numpy as np
import numpy.core
import numpy.core.multiarray
import numpy.core.numeric

# Compatibility for official special PKLs written under NumPy 2.
sys.modules['numpy._core'] = numpy.core
sys.modules['numpy._core.multiarray'] = numpy.core.multiarray
sys.modules['numpy._core.numeric'] = numpy.core.numeric

ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW = ROOT / 'workspace/data/raw'
DEST = ROOT / 'workspace/data/processed/cleaning_2026e_v2'
STATS = ROOT / 'workspace/results/data_audit/cleaning_statistics.json'
CONTRACT = ROOT / 'workspace/evidence_recovery/cleaning_contract.md'

if DEST.exists():
    raise SystemExit(f'Processed output already exists; refusing overwrite: {DEST}')
if not RAW.is_dir():
    raise SystemExit(f'Missing raw directory: {RAW}')
DEST.mkdir(parents=True)

def sha(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()

def jwrite(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding='utf-8')

def load(p):
    with p.open('rb') as f:
        return pickle.load(f)

def normalize_numeric(a, name):
    a = np.asarray(a)
    if not np.isfinite(a).all():
        raise ValueError(f'NaN/Inf in {name}')
    if a.dtype == np.float64 and not np.array_equal(a, a.astype(np.float32).astype(np.float64)):
        raise ValueError(f'Float64 values are not exactly float32 representable: {name}')
    return a.astype(np.float32, copy=False)

def token_ids(a, name):
    a = np.asarray(a)
    if not np.isfinite(a).all() or not np.array_equal(a, np.rint(a)):
        raise ValueError(f'Non-integral token array: {name}')
    q = a.astype(np.int64)
    if not np.array_equal(q.astype(a.dtype), a):
        raise ValueError(f'Non-reversible token cast: {name}')
    if q.shape[-2:] != (3, 50):
        raise ValueError(f'Unexpected token shape: {name} {q.shape}')
    att = q[..., 1, :]
    if not np.isin(att, (0, 1)).all() or not (np.diff(att, axis=-1) <= 0).all():
        raise ValueError(f'Invalid attention mask: {name}')
    return q

def masks(modality, arr, version, source, tokens=None, lengths=None):
    """Codes: P/O 0=known padding, 1=known usable, 2=unknown."""
    if modality == 'text':
        if tokens is None:
            raise ValueError('Text array requires token attention mask')
        p = tokens[:, 1, :].astype(np.uint8)
        o = p.copy()
        z = np.where(p == 0, 1, 0).astype(np.uint8)
        return p, o, z
    n, t, _ = arr.shape
    nonzero = np.any(arr != 0, axis=2)
    p = np.full((n, t), 2, dtype=np.uint8)
    o = np.full((n, t), 2, dtype=np.uint8)
    z = np.full((n, t), 2, dtype=np.uint8)  # 2: native zero / ambiguous, not a missing label
    p[nonzero] = 1
    o[nonzero] = 1
    z[nonzero] = 0
    if version == 'unaligned' and modality == 'audio' and lengths is not None:
        ll = np.atleast_1d(np.asarray(lengths, dtype=int))
        if len(ll) != n or np.any(ll < 0) or np.any(ll > t):
            raise ValueError('Invalid unaligned audio lengths')
        idx = np.arange(t)[None, :]
        valid = idx < ll[:, None]
        if np.any(valid & ~nonzero) or np.any(~valid & nonzero):
            raise ValueError('Unaligned audio length/support disagreement')
        p = valid.astype(np.uint8)
        o = p.copy()
        z = np.where(valid, 0, 1).astype(np.uint8)
    if source == 'attachment3':
        # Internal zeros in special files are only candidates, not official O=0 labels.
        for i in range(n):
            idx = np.flatnonzero(nonzero[i])
            if idx.size:
                middle = np.arange(t)
                candidate = (~nonzero[i]) & (middle > idx[0]) & (middle < idx[-1])
                z[i, candidate] = 3  # suspected local missing; P/O stay unknown
    return p, o, z

def scalar_fit(arr, trusted, name):
    d = arr.shape[-1]
    count = 0
    sums = np.zeros(d, dtype=np.float64)
    squares = np.zeros(d, dtype=np.float64)
    for start in range(0, len(arr), 32):
        end = min(len(arr), start + 32)
        v = np.asarray(arr[start:end], dtype=np.float64)[trusted[start:end]]
        if v.size:
            count += len(v)
            sums += v.sum(axis=0)
            squares += np.square(v).sum(axis=0)
    if count == 0:
        raise ValueError(f'No trusted rows for scaler: {name}')
    mean = sums / count
    var = np.maximum(squares / count - np.square(mean), 0)
    sd = np.sqrt(var)
    constant = sd < 1e-12
    sd[constant] = 1.0
    return {'mean': mean, 'sd': sd, 'count': count, 'constant_channels': int(constant.sum())}

def write_scaled(arr, p, scaler, outpath, name):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    out = np.lib.format.open_memmap(str(outpath), mode='w+', dtype=np.float32, shape=arr.shape)
    before_zero = after_zero = 0
    output_nonfinite = 0
    sumv = np.zeros(arr.shape[-1], dtype=np.float64)
    sumsq = np.zeros(arr.shape[-1], dtype=np.float64)
    ntrusted = 0
    for start in range(0, len(arr), 32):
        end = min(len(arr), start + 32)
        x = normalize_numeric(arr[start:end], name)
        trusted = p[start:end] == 1
        if name != 'text':
            trusted &= np.any(x != 0, axis=2)
        y = x.copy()
        if trusted.any():
            y[trusted] = ((x[trusted].astype(np.float64) - scaler['mean']) / scaler['sd']).astype(np.float32)
            vals = y[trusted].astype(np.float64)
            ntrusted += len(vals)
            sumv += vals.sum(axis=0)
            sumsq += np.square(vals).sum(axis=0)
        if name != 'text':
            bz = np.all(x == 0, axis=2)
            az = np.all(y == 0, axis=2)
            if not np.array_equal(bz, az):
                raise ValueError(f'All-zero row map changed: {outpath}')
            before_zero += int(bz.sum())
            after_zero += int(az.sum())
        output_nonfinite += int(np.size(y) - np.isfinite(y).sum())
        out[start:end] = y
    out.flush()
    if output_nonfinite:
        raise ValueError(f'NaN/Inf after scaling: {outpath}')
    avg = sumv / ntrusted if ntrusted else np.full(arr.shape[-1], np.nan)
    std = np.sqrt(np.maximum(sumsq / ntrusted - np.square(avg), 0)) if ntrusted else avg
    return {'shape': list(arr.shape), 'dtype': 'float32', 'trusted_rows': int(ntrusted),
            'allzero_before': before_zero, 'allzero_after': after_zero,
            'scaled_mean_abs_max': float(np.max(np.abs(avg))) if ntrusted else None,
            'scaled_std_min': float(std.min()) if ntrusted else None,
            'scaled_std_max': float(std.max()) if ntrusted else None,
            'nan_inf': output_nonfinite}

def save_meta(d, ids, raw_texts, labels=None, extra=None):
    rows = []
    for i, sid in enumerate(ids):
        row = {'row': i, 'id': sid, 'raw_text': str(raw_texts[i]) if raw_texts is not None and raw_texts[i] is not None else None}
        if extra is not None:
            row.update(extra[i])
        if labels is not None:
            row.update(classification_label=float(labels[0][i]), regression_label=float(labels[1][i]))
        rows.append(row)
    jwrite(d / 'samples.json', rows)
    return rows

stats = {'contract_sha256': sha(CONTRACT), 'code_sha256': sha(pathlib.Path(__file__)),
         'mask_codes': {'P_O': {'0': 'known_padding', '1': 'known_usable', '2': 'unknown'},
                        'zero_context': {'0': 'nonzero', '1': 'known_padding',
                                         '2': 'native_zero_or_ambiguous', '3': 'suspected_local_gap_not_label'}},
         'scaler_fit': 'Attachment 2 train only, version-specific, per-channel trusted positions',
         'versions': {}, 'source_sha256': {}, 'label_policy': 'train/valid copied exactly; test/special labels not emitted'}

for version in ('aligned', 'unaligned'):
    source_path = next(RAW.rglob(version + '_50.pkl'))
    stats['source_sha256'][source_path.relative_to(RAW).as_posix()] = sha(source_path)
    obj = load(source_path)
    vdest = DEST / version
    vdest.mkdir()
    scalers = {}
    train = obj['train']
    train_tokens = token_ids(train['text_bert'], version + '/train')
    for m in ('text', 'audio', 'vision'):
        arr = np.asarray(train[m])
        p, _, _ = masks(m, arr, version, 'attachment2', tokens=train_tokens,
                        lengths=train.get(m + '_lengths'))
        trusted = p == 1
        if m != 'text':
            trusted &= np.any(arr != 0, axis=2)
        scalers[m] = scalar_fit(arr, trusted, version + '/' + m)
    jwrite(vdest / 'scalers.json', {m: {'mean': q['mean'].tolist(), 'sd': q['sd'].tolist(),
                                      'fit_split': 'train', 'count': q['count'],
                                      'constant_channels': q['constant_channels']} for m, q in scalers.items()})
    stats['versions'][version] = {'scalers': {m: {'fit_rows': q['count'],
                                                  'constant_channels': q['constant_channels']} for m, q in scalers.items()},
                                  'splits': {}, 'special': {}}
    for split in ('train', 'valid', 'test'):
        b = obj[split]
        d = vdest / split
        d.mkdir()
        ids = list(map(str, b['id']))
        if len(ids) != len(set(ids)):
            raise ValueError(f'Duplicate IDs: {version}/{split}')
        tokens = token_ids(b['text_bert'], version + '/' + split)
        np.save(str(d / 'text_bert.npy'), tokens, allow_pickle=False)
        lab = (b['classification_labels'], b['regression_labels']) if split in ('train', 'valid') else None
        rows = save_meta(d, ids, b['raw_text'], labels=lab)
        if [r['id'] for r in rows] != ids:
            raise ValueError('ID changed')
        if lab is not None:
            if not np.array_equal(np.asarray([r['classification_label'] for r in rows]), lab[0]) or not np.array_equal(np.asarray([r['regression_label'] for r in rows]), lab[1]):
                raise ValueError('Label changed')
        record = {'count': len(ids), 'ids_unchanged': True, 'labels_unchanged': bool(lab is not None), 'modalities': {}}
        for m in ('text', 'audio', 'vision'):
            arr = np.asarray(b[m]);p, o, z = masks(m, arr, version, 'attachment2', tokens=tokens,
                                                   lengths=b.get(m + '_lengths'))
            np.save(str(d / (m + '_P.npy')), p, allow_pickle=False)
            np.save(str(d / (m + '_O.npy')), o, allow_pickle=False)
            np.save(str(d / (m + '_zero_context.npy')), z, allow_pickle=False)
            record['modalities'][m] = write_scaled(arr, p, scalers[m], d / (m + '.npy'), m)
            record['modalities'][m]['P_counts'] = dict(Counter(map(str, p.ravel())))
            record['modalities'][m]['O_counts'] = dict(Counter(map(str, o.ravel())))
            record['modalities'][m]['zero_context_counts'] = dict(Counter(map(str, z.ravel())))
            if version == 'unaligned' and m in ('audio', 'vision'):
                np.save(str(d / (m + '_raw_lengths.npy')), np.asarray(b[m + '_lengths'], dtype=np.int32), allow_pickle=False)
        stats['versions'][version]['splits'][split] = record
    del obj

    for attachment, expected in (('附件3', 30), ('附件4', 20)):
        paths = sorted(p for p in RAW.rglob('*.pkl') if any(attachment in part for part in p.parts) and
                       (('unaligned' if np.asarray(load(p)['test']['audio'] if attachment == '附件3' else load(p)['audio']).shape[-2] == 500 else 'aligned') == version))
        # above classification reads file content, not translated directory names
        if len(paths) != expected:
            raise ValueError(f'{version}/{attachment}: expected {expected}, found {len(paths)}')
        blocks = []
        for pth in paths:
            x = load(pth);x = x['test'] if attachment == '附件3' else x
            blocks.append((pth, x))
            stats['source_sha256'][pth.relative_to(RAW).as_posix()] = sha(pth)
        d = vdest / ('attachment3' if attachment == '附件3' else 'attachment4')
        d.mkdir()
        toks = None
        if all('text_bert' in x for _, x in blocks):
            toks = np.stack([token_ids(np.asarray(x['text_bert']).reshape(1, 3, 50), str(p))[0]
                             for p, x in blocks])
            np.save(str(d / 'text_bert.npy'), toks, allow_pickle=False)
        ids = [str(x['id']) if 'id' in x else None for _, x in blocks]
        rawtexts = [np.asarray(x['raw_text']).reshape(-1)[0] if 'raw_text' in x else None for _, x in blocks]
        extra = [{'source_file': p.relative_to(RAW).as_posix(), 'local_key': p.stem,
                  'original_id_present': sid is not None} for (p, x), sid in zip(blocks, ids)]
        save_meta(d, ids, rawtexts, extra=extra)
        record = {'count': len(blocks), 'labels_emitted': False, 'modalities': {},
                  'numeric_text_available': all('text' in x for _, x in blocks),
                  'tokens_available': toks is not None,
                  'raw_text_available': all('raw_text' in x for _, x in blocks)}
        for m in ('text', 'audio', 'vision'):
            if not all(m in x for _, x in blocks):
                continue
            arr = np.stack([normalize_numeric(np.asarray(x[m]).reshape(-1, *np.asarray(x[m]).shape[-2:])[0],
                                              str(p) + '/' + m) for p, x in blocks])
            lengths = [int(x[m + '_lengths']) for _, x in blocks] if version == 'unaligned' and attachment == '附件4' and m in ('audio', 'vision') else None
            p, o, z = masks(m, arr, version, 'attachment3' if attachment == '附件3' else 'attachment4',
                             tokens=toks, lengths=lengths)
            np.save(str(d / (m + '_P.npy')), p, allow_pickle=False)
            np.save(str(d / (m + '_O.npy')), o, allow_pickle=False)
            np.save(str(d / (m + '_zero_context.npy')), z, allow_pickle=False)
            record['modalities'][m] = write_scaled(arr, p, scalers[m], d / (m + '.npy'), m)
            record['modalities'][m]['P_counts'] = dict(Counter(map(str, p.ravel())))
            record['modalities'][m]['O_counts'] = dict(Counter(map(str, o.ravel())))
            record['modalities'][m]['zero_context_counts'] = dict(Counter(map(str, z.ravel())))
            if lengths is not None:
                np.save(str(d / (m + '_raw_lengths.npy')), np.asarray(lengths, dtype=np.int32), allow_pickle=False)
        stats['versions'][version]['special'][attachment] = record

# Attachment 1 has no generated features at this stage: preserve the complete one-to-one manifest.
xlsx = json.loads((ROOT / 'workspace/results/data_audit/xlsx_rows.json').read_text(encoding='utf-8'))
rows = next(v for k, v in xlsx.items() if k.endswith('/label-100.xlsx'))
durations = json.loads((ROOT / 'workspace/evidence_recovery/mp4_mvhd_durations.json').read_text(encoding='utf-8'))
outrows = []
for row in rows:
    matches = [p for p in RAW.rglob(str(row['clip_id']) + '.mp4')
               if p.parent.name == str(row['video_id']) and '附件1' in p.as_posix()]
    if len(matches) != 1:
        raise ValueError(f'Attachment 1 file match failed: {row}')
    p = matches[0]
    outrows.append({'id': str(row['video_id']) + '$_$' + str(row['clip_id']),
                    'video_id': row['video_id'], 'clip_id': row['clip_id'],
                    'raw_video': p.relative_to(RAW).as_posix(), 'raw_size_bytes': p.stat().st_size,
                    'container_duration_s': durations[p.relative_to(RAW).as_posix()],
                    'text': row['text'], 'label': row['label'], 'annotation': row['annotation']})
jwrite(DEST / 'attachment1_manifest.json', outrows)
stats['attachment1'] = {'count': len(outrows), 'unique_ids': len(set(r['id'] for r in outrows)),
                        'videos_retained': len(outrows)}
jwrite(DEST / 'manifest.json', {'version': 1, 'contract_sha256': stats['contract_sha256'],
                                'code_sha256': stats['code_sha256'],
                                'raw_root': 'workspace/data/raw', 'source_sha256': stats['source_sha256'],
                                'mask_codes': stats['mask_codes'], 'notes':
                                ['No raw file modified', 'No samples removed', 'No imputation',
                                 'Scalers fit only on Attachment 2 train',
                                 'Special zero runs are candidates, not ground-truth missing labels']})
jwrite(STATS, stats)
print('CLEANING_OUTPUT', DEST.relative_to(ROOT).as_posix())
print('A2 split counts', {v: {s: q['count'] for s, q in d['splits'].items()}
                           for v, d in stats['versions'].items()})
print('special counts', {v: {s: q['count'] for s, q in d['special'].items()}
                          for v, d in stats['versions'].items()})
