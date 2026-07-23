# dataset/ — iSLEEPS raw recordings (not tracked in git)

This folder holds the raw polysomnography data. It is **git-ignored**: the recordings total
~7.3 GB and individual `.edf` files exceed GitHub's 100 MB per-file limit. Everything here is
re-downloadable from the public archive, so nothing is committed.

## What belongs here

Two files per subject, side by side in this directory:

| File | Contents |
|---|---|
| `SNxx.edf` | Raw PSG signals, European Data Format, vendor sampling rate (256 Hz) |
| `SNxx.xlsx` | Expert annotations; the hypnogram lives in the `Sleep profile` sheet |

`project/processing/build_npz_full.py` and `project/processing/iSLEEPS_preprocessing.ipynb`
both read from this directory and discover whatever subjects are present.

## Source

**iSLEEPS** — Maiti, S., Sharma, S. K., Mythirayee, S., Rajendran, S. & Bapi, R. S.
*Polysomnography Dataset for Sleep Analysis in Ischemic Stroke Patients.*
Scientific Data **13**, 421 (2026). <https://doi.org/10.1038/s41597-026-06747-w>

The release is split across two hosts:

- **Subjects SN1–SN40** — open on Zenodo, no registration:
  <https://doi.org/10.5281/zenodo.14873844>
- **Subjects SN41–SN100** — India Data Portal, requires a free account:
  <https://india-data.org/dataset-details/0b801dfa-4e42-4ec6-9c56-c6892b907ed2>

The paper reports N=99 (SN28 is a byte-identical duplicate of SN15 and is dropped).

## Fetching the open subset

Every file on the Zenodo record carries an md5 checksum in the record metadata, so a download
can be verified rather than assumed complete. A truncated `.edf` is the dangerous failure mode —
it raises no error and silently trains on half a night. Section 1 of the preprocessing notebook
checks each file against the size its own EDF header declares, and excludes any that fall short.

```python
import hashlib, json, os, urllib.request

REC = "https://zenodo.org/api/records/14873844"
OUT = os.path.dirname(os.path.abspath(__file__))

rec = json.load(urllib.request.urlopen(REC))
for f in rec["files"]:
    key = f["key"]
    if key.startswith("~$"):                       # Excel lock files in the record
        continue
    dst = os.path.join(OUT, key)
    if os.path.exists(dst) and os.path.getsize(dst) == f["size"]:
        continue
    print(f"{key} ({f['size'] / 1e6:.0f} MB)")
    urllib.request.urlretrieve(f["links"]["self"], dst)
    got = hashlib.md5(open(dst, "rb").read()).hexdigest()
    assert got == f["checksum"].split(":")[1], f"{key}: checksum mismatch"
```

Annotations alone are only ~17 MB (add `if not key.endswith(".xlsx"): continue`), which is
enough to run the notebook's annotation-decoding and cohort-audit sections without the signals.
