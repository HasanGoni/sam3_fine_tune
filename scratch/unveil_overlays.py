import json, sys, random
from pathlib import Path
import numpy as np, torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
from pycocotools import mask as mu
import unveil
sys.path.insert(0, ".")
from sam3.model_builder import build_sam3_image_model
root = Path(sys.argv[1]); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
coco = json.load(open(root/"test"/"annotation_merged.json"))
cats = {c["id"]: c["name"] for c in coco["categories"]}; names = list(cats.values())
by = {}
for a in coco["annotations"]: by.setdefault(a["image_id"], []).append(a)
random.seed(1); imgs = random.sample([i for i in coco["images"] if len({a["category_id"] for a in by.get(i["id"], [])}) >= 2], 6)
sam3 = build_sam3_image_model(bpe_path=sys.argv[4], device="cpu", eval_mode=True, checkpoint_path=sys.argv[3], load_from_HF=False).to("cuda").eval().requires_grad_(False)
model = unveil.load(sam3_model=sam3)
cm = plt.get_cmap("tab20"); col = {n: cm(i)[:3] for i, n in enumerate(names)}
def paint(base, masks):
    o = base.astype(float)/255
    for n, m in masks.items():
        o[m] = 0.45*o[m] + 0.55*np.array(col[n])
    return o
for im in imgs:
    pil = Image.open(root/"test"/"images"/im["file_name"]).convert("RGB"); h, w = pil.size[1], pil.size[0]; base = np.asarray(pil)
    gt = {}
    for a in by.get(im["id"], []):
        s = a["segmentation"]; r = mu.merge(mu.frPyObjects(s, h, w)) if isinstance(s, list) else (mu.frPyObjects(s, h, w) if isinstance(s["counts"], list) else s)
        n = cats[a["category_id"]]; gt[n] = gt.get(n, np.zeros((h, w), bool)) | mu.decode(r).astype(bool)
    prompts = [n.replace("_", " ") for n in names]
    mu_, _ = model.predict(pil, prompts); ms_, _ = model.predict(pil, prompts, path="sam3")
    f = lambda ms: {n: np.asarray(ms[i]) > 0.5 for i, n in enumerate(names) if (np.asarray(ms[i]) > 0.5).any()}
    fig, ax = plt.subplots(1, 3, figsize=(21, 7))
    for a_, t, m in zip(ax, ["Ground truth", "UNVEIL (zero-shot)", "SAM3 semantic path (base)"], [gt, f(mu_), f(ms_)]):
        a_.imshow(paint(base, m)); a_.set_title(t); a_.axis("off")
    hs = [plt.Line2D([], [], color=col[n], lw=6, label=n) for n in names]
    fig.legend(handles=hs, loc="lower center", ncol=6); plt.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out/f"{Path(im['file_name']).stem}.png", dpi=70); plt.close(fig)
print("done")
