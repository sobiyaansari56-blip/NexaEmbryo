"""NexaEmbryo backend. Run:  uvicorn app:app --host 127.0.0.1 --port 8000"""
import io, json, os, sqlite3, time, uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import torch, torch.nn as nn, timm
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from torchvision import transforms as T

BASE = Path(__file__).parent
MODEL_DIR = BASE / "model_out"
UPLOADS = BASE / "uploads"; UPLOADS.mkdir(exist_ok=True)
FRONT = BASE.parent / "frontend"
DB = BASE / "nexaembryo.db"

# ---------------------------------------------------------------- model
cfg = json.load(open(MODEL_DIR / "labels.json"))
HEADS = cfg["heads"]
NCLS = dict(stage=len(cfg["stages"]), cells=len(cfg["cells"]), frag=len(cfg["frag"]),
            expn=len(cfg["expn"]), icm=3, te=3)
NAMES = dict(stage=cfg["stages"], cells=cfg["cells"], frag=cfg["frag"], expn=cfg["expn"],
             icm=cfg["grade"], te=cfg["grade"])


class NexaNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(cfg["backbone"], pretrained=False, num_classes=0)
        d = self.backbone.num_features
        self.drop = nn.Dropout(0.3)
        self.heads = nn.ModuleDict({h: nn.Linear(d, NCLS[h]) for h in HEADS})

    def forward(self, x):
        f = self.backbone(x)
        return {h: self.heads[h](self.drop(f)) for h in HEADS}, f


dev = "cuda" if torch.cuda.is_available() else "cpu"
model = NexaNet()
model.load_state_dict(torch.load(MODEL_DIR / "best.pt", map_location="cpu"))
model.to(dev).eval()
tf = T.Compose([T.Resize((cfg["size"], cfg["size"])), T.ToTensor(), T.Normalize(cfg["mean"], cfg["std"])])

ret = np.load(MODEL_DIR / "retrieval.npz")["feats"]
ret_meta = json.load(open(MODEL_DIR / "captions.json"))


def run_model(img: Image.Image):
    x = tf(img.convert("RGB")).unsqueeze(0).to(dev)
    with torch.no_grad():
        out, f = model(x)
    probs = {h: out[h].softmax(1)[0].cpu().numpy() for h in HEADS}
    f = torch.nn.functional.normalize(f, dim=1)[0].cpu().numpy()
    sims = ret @ f
    k = int(sims.argmax())
    return probs, dict(caption=ret_meta["captions"][k], filename=ret_meta["filenames"][k],
                       source=ret_meta["source"][k], similarity=float(sims[k]))


# ---------------------------------------------------------------- interpretation
ICM_TXT = {"A": "Well-defined", "B": "Moderately defined", "C": "Poorly defined"}
TE_TXT = {"A": "Good Quality", "B": "Fair Quality", "C": "Poor Quality"}
DAY = {"oocyte": 0, "zygote": 1, "morula": 4, "blastocyst": 5}


def interpret(probs):
    top = {h: int(probs[h].argmax()) for h in HEADS}
    conf = {h: float(probs[h].max()) for h in HEADS}
    stage = NAMES["stage"][top["stage"]]
    r = dict(stage_key=stage, details={})
    used = ["stage"]

    if stage == "invalid":
        r.update(stage="No analysable embryo", cellularity="N/A", icm="N/A", te="N/A",
                 fragmentation="N/A", exp_grade=None)
    else:
        if stage == "cleavage":
            c = top["cells"]; used.append("cells")
            cellularity = ["Low", "Low", "Medium", "Medium", "High"][c]
            cells_txt = NAMES["cells"][c]
            day = 2 if c <= 1 else 3
            r["stage"] = f"Cleavage stage, {cells_txt} (Day {day})"; r["cells_txt"] = cells_txt
        elif stage == "morula":
            cellularity = "High"; r["stage"] = "Morula (Day 4)"
        elif stage == "blastocyst":
            cellularity = "High"
        else:
            cellularity = "N/A"; r["stage"] = f"{stage.capitalize()} (Day {DAY[stage]})"
        r["cellularity"] = cellularity

        if stage in ("cleavage", "morula", "blastocyst"):
            used.append("frag")
            r["fragmentation"] = NAMES["frag"][top["frag"]].replace("none", "None (0%)").replace("mild", "Low").replace(
                "moderate", "Moderate").replace("severe", "High")
        else:
            r["fragmentation"] = "N/A"

        if stage == "blastocyst":
            used += ["expn", "icm", "te"]
            g = int(NAMES["expn"][top["expn"]])
            icm, te = NAMES["icm"][top["icm"]], NAMES["te"][top["te"]]
            r["exp_grade"] = g
            r["gardner"] = f"{g}{icm}{te}"
            r["stage"] = f"Blastocyst, Gardner {g}{icm}{te} (Day 5)"
            r["icm"] = f"{ICM_TXT[icm]} (grade {icm})"
            r["te"] = f"{TE_TXT[te]} (grade {te})"
        else:
            r["icm"] = r["te"] = "N/A"; r["exp_grade"] = None

    r["confidence"] = round(float(np.mean([conf[h] for h in used])), 3)
    r["confidence_label"] = "High Confidence" if r["confidence"] >= 0.8 else \
        "Medium Confidence" if r["confidence"] >= 0.6 else "Low Confidence"
    r["scores"] = {
        "cellularity": {"Low": .25, "Medium": .6, "High": .9}.get(r.get("cellularity"), 0),
        "icm": {"A": .9, "B": .6, "C": .3}.get(r.get("gardner", "  ")[1:2], 0),
        "te": {"A": .9, "B": .6, "C": .3}.get(r.get("gardner", "   ")[2:3], 0),
        "frag": {"None (0%)": .02, "Low (<10%)": .1, "Moderate (10-25%)": .4, "High (>25%)": .8}.get(r["fragmentation"], 0),
    }
    r["description"], r["summary"] = describe(r)
    return r


def describe(r):
    s = r["stage_key"]
    if s == "invalid":
        return ("No embryo or oocyte could be identified in this image (it may be empty, too dark or out of focus).",
                "Image not analysable. Please re-capture before assessment.")
    frag = r["fragmentation"]
    fr_txt = {"None (0%)": "no visible fragmentation", "Low (<10%)": "minimal fragmentation",
              "Moderate (10-25%)": "moderate fragmentation", "High (>25%)": "heavy fragmentation"}.get(frag, "")
    if s == "blastocyst":
        d = (f"The image shows a blastocyst (Gardner {r['gardner']}). The inner cell mass is {r['icm'].lower()} and the "
             f"trophectoderm is of {r['te'].lower()}; {fr_txt}.")
        sm = (f"The embryo shows a {r['icm'].split(' (')[0].lower()} inner cell mass and {r['te'].split(' (')[0].lower()} "
              f"trophectoderm with {fr_txt}, consistent with a Gardner {r['gardner']} blastocyst.")
    elif s == "cleavage":
        d = f"The image shows a cleavage-stage embryo with approximately {r['cells_txt']} and {fr_txt}."
        sm = f"Cleavage-stage embryo, {r['cellularity'].lower()} cellularity, {fr_txt}."
    elif s == "morula":
        d = f"The image shows a compacted/compacting morula with {fr_txt}."
        sm = f"Morula stage embryo with {fr_txt}."
    elif s == "zygote":
        d = "The image shows a fertilised oocyte (zygote), indicated by the presence of two polar bodies and/or pronuclei."
        sm = "Zygote (Day 1). No further morphological grading applies at this stage."
    else:
        d = "The image shows an oocyte. Maturity (MI/MII) and fertilisation status should be confirmed by the embryologist."
        sm = "Oocyte image. No embryo morphology grading applies."
    return d, sm


# ---------------------------------------------------------------- database
def db():
    c = sqlite3.connect(DB); c.row_factory = sqlite3.Row; return c


with db() as c:
    c.executescript("""
    CREATE TABLE IF NOT EXISTS analyses(
      id TEXT PRIMARY KEY, created REAL, patient_id TEXT, embryo_id TEXT, clinician TEXT, comments TEXT,
      selected INTEGER DEFAULT 0, image TEXT, result TEXT, report_id TEXT, report_created REAL);
    CREATE TABLE IF NOT EXISTS settings(k TEXT PRIMARY KEY, v TEXT);
    """)
DEFAULT_SETTINGS = dict(clinic="NexaEmbryo", clinician="", confidence_threshold="0.6")


def get_settings():
    s = dict(DEFAULT_SETTINGS)
    with db() as c:
        for r in c.execute("SELECT * FROM settings"): s[r["k"]] = r["v"]
    return s


def row_out(r):
    d = dict(r); d["result"] = json.loads(d["result"]); d["image_url"] = f"/uploads/{d['image']}"
    d["created_iso"] = datetime.fromtimestamp(d["created"]).isoformat(timespec="minutes"); return d


def next_patient_id():
    with db() as c:
        n = c.execute("SELECT COUNT(DISTINCT patient_id) FROM analyses").fetchone()[0]
    return f"NX-{datetime.now().year}-{n + 1:03d}"


# ---------------------------------------------------------------- api
app = FastAPI(title="NexaEmbryo")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/uploads", StaticFiles(directory=UPLOADS), name="uploads")


@app.get("/api/health")
def health(): return dict(ok=True, device=dev)


@app.get("/api/next-patient-id")
def npid(): return dict(patient_id=next_patient_id())


@app.post("/api/analyze")
async def analyze(image: UploadFile = File(...), patient_id: str = Form(""), embryo_id: str = Form("")):
    data = await image.read()
    try:
        img = Image.open(io.BytesIO(data)); img.load()
    except Exception:
        raise HTTPException(400, "Could not read image file")
    probs, similar = run_model(img)
    result = interpret(probs); result["similar_case"] = similar
    aid = uuid.uuid4().hex[:10]
    fname = f"{aid}.jpg"; img.convert("RGB").save(UPLOADS / fname, quality=92)
    s = get_settings()
    patient_id = patient_id.strip() or next_patient_id()
    if not embryo_id.strip():
        with db() as c:
            n = c.execute("SELECT COUNT(*) FROM analyses WHERE patient_id=?", (patient_id,)).fetchone()[0]
        embryo_id = f"EMB-{n + 1:03d}"
    with db() as c:
        c.execute("INSERT INTO analyses(id,created,patient_id,embryo_id,clinician,comments,image,result) VALUES(?,?,?,?,?,?,?,?)",
                  (aid, time.time(), patient_id, embryo_id, s["clinician"], "", fname, json.dumps(result)))
        r = c.execute("SELECT * FROM analyses WHERE id=?", (aid,)).fetchone()
    return row_out(r)


@app.get("/api/analyses")
def list_analyses(q: str = "", limit: int = 500):
    with db() as c:
        rows = c.execute("SELECT * FROM analyses WHERE patient_id LIKE ? OR embryo_id LIKE ? ORDER BY created DESC LIMIT ?",
                         (f"%{q}%", f"%{q}%", limit)).fetchall()
    return [row_out(r) for r in rows]


@app.get("/api/analyses/{aid}")
def get_analysis(aid: str):
    with db() as c:
        r = c.execute("SELECT * FROM analyses WHERE id=?", (aid,)).fetchone()
    if not r: raise HTTPException(404)
    return row_out(r)


class Patch(BaseModel):
    patient_id: str | None = None
    embryo_id: str | None = None
    clinician: str | None = None
    comments: str | None = None
    selected: bool | None = None


@app.patch("/api/analyses/{aid}")
def patch_analysis(aid: str, p: Patch):
    fields = {k: v for k, v in p.model_dump().items() if v is not None}
    if "selected" in fields: fields["selected"] = int(fields["selected"])
    if fields:
        with db() as c:
            c.execute(f"UPDATE analyses SET {','.join(k + '=?' for k in fields)} WHERE id=?", (*fields.values(), aid))
    return get_analysis(aid)


@app.delete("/api/analyses/{aid}")
def delete_analysis(aid: str):
    with db() as c:
        c.execute("DELETE FROM analyses WHERE id=?", (aid,))
    return dict(ok=True)


@app.get("/api/stats")
def stats():
    with db() as c:
        rows = [row_out(r) for r in c.execute("SELECT * FROM analyses ORDER BY created DESC")]
    stages = {}
    for r in rows:
        k = r["result"]["stage_key"]; stages[k] = stages.get(k, 0) + 1
    return dict(total=len(rows), patients=len({r["patient_id"] for r in rows}),
                selected=sum(1 for r in rows if r["selected"]),
                reports=sum(1 for r in rows if r["report_id"]),
                avg_conf=round(float(np.mean([r["result"]["confidence"] for r in rows])), 3) if rows else 0,
                stages=stages, recent=rows[:6])


@app.get("/api/settings")
def read_settings(): return get_settings()


@app.put("/api/settings")
def write_settings(s: dict):
    with db() as c:
        for k, v in s.items():
            if k in DEFAULT_SETTINGS:
                c.execute("INSERT INTO settings(k,v) VALUES(?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v", (k, str(v)))
    return get_settings()


# ---------------------------------------------------------------- PDF report
def build_pdf(a: dict) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Image as RLImage, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    navy = colors.HexColor("#0B2A6B"); light = colors.HexColor("#EAF2FF")
    ss = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=ss["Normal"], fontSize=9.5, leading=13)
    small = ParagraphStyle("s", parent=body, fontSize=8, textColor=colors.HexColor("#4A5568"))
    h = ParagraphStyle("h", parent=body, fontName="Helvetica-Bold", textColor=colors.white, fontSize=10)
    title = ParagraphStyle("t", parent=ss["Title"], textColor=navy, alignment=0, fontSize=20, spaceAfter=0)
    r = a["result"]; settings = get_settings()
    rd = datetime.fromtimestamp(a["report_created"] or time.time()).strftime("%d %b %Y, %H:%M")

    def band(txt):
        t = Table([[Paragraph(txt, h)]], colWidths=[180 * mm])
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), navy), ("TOPPADDING", (0, 0), (-1, -1), 3),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
        return t

    def grid(rows, widths):
        t = Table(rows, colWidths=widths)
        t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
                               ("FONTSIZE", (0, 0), (-1, -1), 9), ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                               ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F1F5F9")),
                               ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
        return t

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=14 * mm, bottomMargin=14 * mm)
    el = [Paragraph(f"{settings['clinic']}", title),
          Paragraph("EMBRYO ANALYSIS &amp; SELECTION REPORT", small), Spacer(1, 6),
          band("1. PATIENT AND REPORT INFORMATION"),
          grid([["Report ID", a["report_id"] or "-"], ["Patient ID", a["patient_id"]], ["Embryo ID", a["embryo_id"]],
                ["Report Date", rd], ["Reviewed By", a["clinician"] or "-"]], [45 * mm, 135 * mm]), Spacer(1, 8),
          band("2. EMBRYO IMAGE OVERVIEW AND 3. AI-GENERATED DESCRIPTION")]
    img = RLImage(str(UPLOADS / a["image"]), width=62 * mm, height=62 * mm)
    desc = Paragraph(r["description"] + "<br/><br/><font size=8 color='#4A5568'>" + r["summary"] + "</font>", body)
    t = Table([[img, desc]], colWidths=[66 * mm, 114 * mm])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("BACKGROUND", (1, 0), (1, 0), light),
                           ("LEFTPADDING", (1, 0), (1, 0), 8), ("TOPPADDING", (1, 0), (1, 0), 8)]))
    el += [t, Spacer(1, 8), band("4. MORPHOLOGICAL OBSERVATIONS"),
           grid([["Developmental Stage", r["stage"]], ["Cellularity", r["cellularity"]],
                 ["Inner Cell Mass (ICM)", r["icm"]], ["Trophectoderm (TE)", r["te"]],
                 ["Fragmentation", r["fragmentation"]], ["AI Confidence", f"{r['confidence']*100:.0f}% ({r['confidence_label']})"]],
                [60 * mm, 120 * mm]), Spacer(1, 8), band("5. FINAL SELECTION"),
           Paragraph("<b>Selected by Embryologist</b>" if a["selected"] else "Not selected", body),
           Spacer(1, 4), Paragraph(f"<b>Comments:</b> {a['comments'] or '-'}", body), Spacer(1, 14),
           Paragraph("<i>This report supports laboratory documentation and expert review. "
                     "AI-generated observations are not a clinical recommendation.</i>", small)]
    doc.build(el)
    return buf.getvalue()


@app.get("/api/report/{aid}")
def report(aid: str):
    with db() as c:
        r = c.execute("SELECT * FROM analyses WHERE id=?", (aid,)).fetchone()
        if not r: raise HTTPException(404)
        if not r["report_id"]:
            c.execute("UPDATE analyses SET report_id=?, report_created=? WHERE id=?",
                      (f"NX-RPT-{uuid.uuid4().hex[:4].upper()}", time.time(), aid))
        r = c.execute("SELECT * FROM analyses WHERE id=?", (aid,)).fetchone()
    pdf = build_pdf(row_out(r))
    return StreamingResponse(io.BytesIO(pdf), media_type="application/pdf",
                             headers={"Content-Disposition": f'attachment; filename="{r["report_id"]}.pdf"'})


# ---------------------------------------------------------------- front end
if FRONT.exists():
    app.mount("/", StaticFiles(directory=FRONT, html=True), name="front")