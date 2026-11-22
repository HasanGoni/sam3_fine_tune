"""Zero-shot UNVEIL vs SAM3 semantic path on a local COCO test split (per-category IoU of the union mask)."""
import json, sys, time, argparse, random
from pathlib import Path
import numpy as np, torch
from PIL import Image
from pycocotools import mask as mu
import unveil
sys.path.insert(0, ".")
from sam3.model_builder import build_sam3_image_model

p = argparse.ArgumentParser()
p.add_argument("--root", required=True); p.add_argument("--n", type=int, default=50)
p.add_argument("--sam3", required=True); p.add_argument("--bpe", required=True); p.add_argument("--out", required=True)
a = p.parse_args()
root = Path(a.root)
coco = json.load(open(root / "test" / "annotation_merged.json"))
cats = {c["id"]: c["name"] for c in coco["categories"]}
by_img = {}
for an in coco["annotations"]:
    by_img.setdefault(an["image_id"], []).append(an)
imgs = [i for i in coco["images"]]
random.Random(0).shuffle(imgs); imgs = imgs[: a.n]

sam3 = build_sam3_image_model(bpe_path=a.bpe, device="cpu", eval_mode=True, checkpoint_path=a.sam3, load_from_HF=False)
sam3 = sam3.to("cuda").eval().requires_grad_(False)
model = unveil.load(sam3_model=sam3)
names = list(cats.values()); prompts = [n.replace("_", " ") for n in names]

def gt_mask(anns, h, w):
    m = np.zeros((h, w), bool)
    for an in anns:
        s = an["segmentation"]
        if isinstance(s, list):
            r = mu.merge(mu.frPyObjects(s, h, w))
        elif isinstance(s["counts"], list):
            r = mu.frPyObjects(s, h, w)
        else:
            r = s
        m |= mu.decode(r).astype(bool)
    return m

acc = {n: {k: {"inter": 0, "union": 0, "n": 0, "pos": 0} for k in ("unveil", "sam3")} for n in names}
t0 = time.time()
for k, im in enumerate(imgs):
    path = root / "test" / "images" / im["file_name"]
    h, w = im["height"], im["width"]
    outs = {}
    outs["unveil"], pres = model.predict(str(path), prompts)
    outs["sam3"], _ = model.predict(str(path), prompts, path="sam3")
    for cid, n in cats.items():
        g = gt_mask([x for x in by_img.get(im["id"], []) if x["category_id"] == cid], h, w)
        for key in outs:
            pm = np.asarray(outs[key][names.index(n)]) > 0.5
            if pm.shape != g.shape:
                pm = np.asarray(Image.fromarray(pm).resize((w, h), Image.NEAREST))
            r = acc[n][key]
            if g.any():
                r["pos"] += 1
                r["inter"] += int((pm & g).sum()); r["union"] += int((pm | g).sum())
    if k % 10 == 0:
        print(k, f"{time.time()-t0:.0f}s", flush=True)
res = {n: {key: {"IoU(pooled)": (v["inter"] / v["union"]) if v["union"] else None, "images_with_gt": v["pos"]}
           for key, v in d.items()} for n, d in acc.items()}
json.dump(res, open(a.out, "w"), indent=1)
print(f"{'category':26s}{'#img':>6s}{'UNVEIL':>9s}{'SAM3 sem':>10s}")
for n, d in res.items():
    f = lambda x: "  n/a" if x is None else f"{x:.3f}"
    print(f"{n:26s}{d['unveil']['images_with_gt']:6d}{f(d['unveil']['IoU(pooled)']):>9s}{f(d['sam3']['IoU(pooled)']):>10s}")
