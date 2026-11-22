import matplotlib.pyplot as plt
import json, random, re, numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
from pycocotools import mask as mu
import cv2
root = Path("/home/hasan-spark/workspace/projects/data/sam3_data/10cat_9cat_v3_unique_panda_matrix_substrate_v1_grayvoid_wholechips_v1/train")
out = Path.home()/"unveil_overlays"/"sheets"; out.mkdir(parents=True, exist_ok=True)
a = json.load(open(root/"annotation_merged.json"))
cn = {c["id"]: c["name"] for c in a["categories"]}; names = list(cn.values())
col = {n: tuple(int(255*v) for v in plt.get_cmap("tab20")(i)[:3]) for i, n in enumerate(names)}
anns = {}
for x in a["annotations"]: anns.setdefault(x["image_id"], []).append(x)
cats = {i["id"]: {cn[x["category_id"]] for x in anns.get(i["id"], [])} for i in a["images"]}
pre = lambda f: re.split(r"[_\d]", f)[0]
groups = {
 "pin_et_pin": lambda i, c: "et_pin" in c, "pin_eberhard_pin": lambda i, c: "eberhard_pin" in c, "pin_wuxi_pin": lambda i, c: "wuxi_pin" in c,
 "war_crack": lambda i, c: "war_crack" in c, "wire_panda_thick": lambda i, c: "thick_wire_panda" in c, "wire_panda_thin": lambda i, c: "thin_wire_panda" in c,
 "wire_HD": lambda i, c: "thick_wire_HD" in c, "chip_lifted_chip": lambda i, c: "lifted_chip" in c, "chip_gray_void": lambda i, c: "gray_void" in c,
 "chip_matrix_substrate": lambda i, c: "matrix_substrate" in c,
 "UNANNOTATED_f": lambda i, c: not c and pre(i["file_name"]) == "f", "UNANNOTATED_g": lambda i, c: not c and pre(i["file_name"]) == "g",
 "UNANNOTATED_pin_sources_a_c_e": lambda i, c: not c and pre(i["file_name"]) in ("a", "c", "e"),
}
T, COLS, ROWS = 420, 4, 3
def thumb(i):
    im = Image.open(root/"images"/i["file_name"]).convert("RGB"); w, h = im.size
    d = ImageDraw.Draw(im); lw = max(2, w // 300)
    for x in anns.get(i["id"], []):
        s = x["segmentation"]; r = mu.merge(mu.frPyObjects(s, h, w)) if isinstance(s, list) else (mu.frPyObjects(s, h, w) if isinstance(s["counts"], list) else s)
        m = mu.decode(r).astype(np.uint8)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cs:
            if len(c) > 1: d.line([tuple(p[0]) for p in c] + [tuple(c[0][0])], fill=col[cn[x["category_id"]]], width=lw)
    im.thumbnail((T, T)); canvas = Image.new("RGB", (T, T + 18), (20, 20, 20)); canvas.paste(im, ((T - im.width)//2, 0))
    ImageDraw.Draw(canvas).text((3, T + 3), (",".join(sorted(cats[i["id"]])) or "-")[:60] + " | " + i["file_name"][:20], fill=(255, 255, 255))
    return canvas
rng = random.Random(0); links = []
for g, f in groups.items():
    pool = [i for i in a["images"] if f(i, cats[i["id"]]) and (root/"images"/i["file_name"]).exists()]
    pick = rng.sample(pool, min(COLS*ROWS, len(pool)))
    sheet = Image.new("RGB", (COLS*T, ROWS*(T+18)))
    for k, i in enumerate(pick): sheet.paste(thumb(i), ((k % COLS)*T, (k // COLS)*(T+18)))
    sheet.save(out/f"{g}.jpg", quality=85); links.append(f'<h3>{g} ({len(pool)} images, {len(pick)} shown)</h3><img src="{g}.jpg" style="max-width:100%">')
legend = " ".join(f'<span style="background:rgb{col[n]};padding:2px 6px">{n}</span>' for n in names)
(out/"index.html").write_text(f"<html><body style='background:#222;color:#eee'><p>Outline colour = existing annotation. Look for objects of OTHER categories that have no outline.</p><p>{legend}</p>" + "".join(links) + "</body></html>")
print("ok", len(links))
