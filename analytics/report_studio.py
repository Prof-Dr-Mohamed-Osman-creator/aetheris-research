# ==========================================
# Aetheris Research - Report Studio engine (v1.12, Catalogue §8)
# Pure builders: assemble captured results into one academic report (title page, abstract, numbered sections, APA /
# journal tables and figures with captions, merged reference list) and export to Word, PDF (Arabic shaped with
# HarfBuzz), LaTeX (zip with figures) and self-contained HTML. Also: a deterministic results writer that only uses the
# computed numbers, and a number checker that flags any figure in a text that is not in the computed results.
# ==========================================

from __future__ import annotations

import datetime as _dt
import io
import json
import os
import re
import zipfile

import numpy as np
import pandas as pd

REF_RX = re.compile(r"\((1[89]|20)\d\d[a-z]?\)|\b(1[89]|20)\d\d\.\s")
NUM_RX = re.compile(r"(?<![\w.])[-−]?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\w])|(?<![\w.])[-−]?\d*\.?\d+(?:[eE][-+]?\d+)?(?![\w])")
AR_RX = re.compile(r"[؀-ۿ]")


def strip_html(s) -> str:
    s = re.sub(r"<br\s*/?>", "\n", str(s))
    s = re.sub(r"<[^>]+>", "", s)
    return s.replace("**", "").replace("&nbsp;", " ").strip()


def is_reference(s: str) -> bool:
    return bool(REF_RX.search(str(s))) and len(str(s)) > 30


def _fmt_df(df: pd.DataFrame, d: int = 4) -> pd.DataFrame:
    out = df.copy()
    if out.index.name or not isinstance(out.index, pd.RangeIndex):
        out = out.reset_index()
    out.columns = [str(c) for c in out.columns]
    for c in out.columns:                      # numbers that arrive as text (tables pre-formatted for Word) become numbers again
        if out[c].dtype == object:
            out[c] = out[c].map(lambda v: np.nan if isinstance(v, str) and v.strip().lower() in ("nan", "none", "nat", "") else v)
            conv = pd.to_numeric(out[c], errors="coerce")
            if conv.notna().sum() == out[c].replace("", np.nan).notna().sum() and conv.notna().any():
                out[c] = conv.astype(float)
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]) and (c.lower().startswith("p-value") or c.lower() in ("p", "p>|t|", "p>|z|", "robust p") or c.lower().startswith("p (")):
            out[c] = out[c].map(lambda v: "" if pd.isna(v) else ("< 0.001" if v < 0.001 else f"{v:.3f}"))
        elif pd.api.types.is_float_dtype(out[c]) and out[c].dropna().map(float.is_integer).all() and out[c].abs().max() < 1e9:
            out[c] = out[c].map(lambda v: "" if pd.isna(v) else f"{int(v):d}")
        elif pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].map(lambda v: "" if pd.isna(v) else (f"{v:.{d}g}" if abs(v) >= 1e5 or (0 < abs(v) < 1e-4) else f"{v:.{d}f}"))
        else:
            out[c] = out[c].map(lambda v: "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v))
    return out


# ======================================================================================================================
# Figures: plotly JSON → PNG (kaleido if available, else a matplotlib re-rendering of scatter / line / bar traces)
# ======================================================================================================================
_PNG_CACHE: dict = {}


def fig_png(fig_json: str, width: int = 900, height: int = 520) -> bytes | None:
    import hashlib
    key = hashlib.md5((fig_json + f"{width}x{height}").encode()).hexdigest()
    if key not in _PNG_CACHE:
        if len(_PNG_CACHE) > 200:
            _PNG_CACHE.clear()
        _PNG_CACHE[key] = _fig_png(fig_json, width, height)
    return _PNG_CACHE[key]


def _fig_png(fig_json: str, width: int = 900, height: int = 520) -> bytes | None:
    import plotly.io as pio
    try:
        fig = pio.from_json(fig_json)
    except Exception:  # noqa: BLE001
        return None
    if os.environ.get("AETHERIS_NO_KALEIDO") != "1":
        try:
            return fig.to_image(format="png", width=width, height=height, scale=2)
        except Exception:  # noqa: BLE001
            pass
    return _mpl_png(fig, width, height)


def _mpl_png(fig, width, height) -> bytes | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        f, ax = plt.subplots(figsize=(width / 110, height / 110), dpi=150)
        drawn = 0
        for tr in fig.data:
            typ = tr.type
            x = np.asarray(tr.x) if getattr(tr, "x", None) is not None else None
            y = np.asarray(tr.y) if getattr(tr, "y", None) is not None else None
            if x is None or y is None:
                continue
            col = None
            if getattr(tr, "marker", None) is not None and isinstance(tr.marker.color, str):
                col = tr.marker.color
            if getattr(tr, "line", None) is not None and isinstance(getattr(tr.line, "color", None), str):
                col = tr.line.color
            lab = tr.name if tr.showlegend is not False and tr.name else None
            if typ == "bar":
                try:
                    ax.bar(x, y.astype(float), color=col or "#1f4e79", label=lab, alpha=0.85)
                except Exception:  # noqa: BLE001
                    ax.bar([str(v) for v in x], y.astype(float), color=col or "#1f4e79", label=lab, alpha=0.85)
                drawn += 1
            elif typ in ("scatter", "scattergl"):
                mode = tr.mode or "lines"
                if tr.fill == "toself":
                    ax.fill(x, y.astype(float), color=col or "#1f4e79", alpha=0.18, lw=0)
                    drawn += 1
                    continue
                ls = "--" if getattr(tr.line, "dash", None) in ("dash", "dot", "dashdot") else "-"
                if "lines" in mode:
                    ax.plot(x, y.astype(float), ls, color=col, label=lab, lw=1.8)
                if "markers" in mode:
                    ax.scatter(x, y.astype(float), s=14, color=col, label=None if "lines" in mode else lab)
                drawn += 1
        if not drawn:
            plt.close(f)
            return None
        xt = fig.layout.xaxis.title.text if fig.layout.xaxis and fig.layout.xaxis.title else None
        yt = fig.layout.yaxis.title.text if fig.layout.yaxis and fig.layout.yaxis.title else None
        if xt and not AR_RX.search(xt):
            ax.set_xlabel(xt)
        if yt and not AR_RX.search(yt):
            ax.set_ylabel(yt)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        h, lb = ax.get_legend_handles_labels()
        if h and not any(AR_RX.search(str(v)) for v in lb):
            ax.legend(frameon=False, fontsize=8)
        f.tight_layout()
        buf = io.BytesIO()
        f.savefig(buf, format="png")
        plt.close(f)
        return buf.getvalue()
    except Exception:  # noqa: BLE001
        return None


# ======================================================================================================================
# Deterministic results writer and number checker
# ======================================================================================================================
_EST = ("Estimate", "Coef.", "Coef", "coef", "ATT", "Effect", "Elasticity", "b")
_SE = ("Std. Err.", "SE", "Std. Err. (simulated)", "Std. Err. (bootstrap)", "std err")
_P = ("p-value", "P>|t|", "P>|z|", "p", "Robust p")
_NAME = ("Term", "Variable", "Parameter", "Aggregation", "Estimator", "Method", "Input", "Good", "index")


def _col(df, cands):
    for c in cands:
        if c in df.columns:
            return c
    return None


def coef_tables(result) -> list:
    out = []
    for kind, content in result.get("sections", []):
        if kind == "table" and isinstance(content, pd.DataFrame):
            df = content.reset_index() if not isinstance(content.index, pd.RangeIndex) else content
            if _col(df, _EST) and (_col(df, _P) or _col(df, _SE)):
                out.append(df)
    return out


def results_paragraph(result: dict, lang: str = "en", alpha: float = 0.05) -> str:
    """Write a results paragraph strictly from the computed numbers: the analysis' own interpretations plus one
    sentence per estimate (sign, size, standard error, significance)."""
    parts = [strip_html(t) for t in result.get("insights", [])]
    for df in coef_tables(result)[:2]:
        e, s, p, nm = _col(df, _EST), _col(df, _SE), _col(df, _P), _col(df, _NAME)
        rows = []
        for _, r in df.iterrows():
            name = str(r[nm]) if nm else str(r.iloc[0])
            if name.lower() in ("const", "_cons", "intercept", "(intercept)"):
                continue
            try:
                b = float(r[e])
            except (TypeError, ValueError):
                continue
            if not np.isfinite(b):
                continue
            pv = None
            if p:
                try:
                    pv = float(str(r[p]).replace("< ", "").replace("<", ""))
                except ValueError:
                    pv = None
            se = None
            if s:
                try:
                    se = float(r[s])
                except (TypeError, ValueError):
                    se = None
            rows.append((name, b, se, pv))
        if not rows:
            continue
        sig = [r_ for r_ in rows if r_[3] is not None and r_[3] < alpha]
        ns = [r_ for r_ in rows if r_[3] is not None and r_[3] >= alpha]
        if lang == "ar":
            if sig:
                parts.append("وتُظهر التقديرات أن " + "؛ و".join(
                    f"أثر {n} {'موجب' if b > 0 else 'سالب'} ({b:.4g}" + (f"، خ.م. {se:.3g}" if se else "") + f"، p {'< 0.001' if pv < 0.001 else f'= {pv:.3f}'})" for n, b, se, pv in sig[:8])
                    + f"، وجميعها معنوية عند مستوى {int(alpha * 100)}%.")
            if ns:
                parts.append("في حين لم تكن آثار " + "، ".join(n for n, *_ in ns[:8]) + f" معنوية إحصائياً عند مستوى {int(alpha * 100)}%.")
        else:
            if sig:
                parts.append("The estimates show " + "; ".join(
                    f"a {'positive' if b > 0 else 'negative'} effect of {n} ({b:.4g}" + (f", SE {se:.3g}" if se else "") + f", p {'< 0.001' if pv < 0.001 else f'= {pv:.3f}'})" for n, b, se, pv in sig[:8])
                    + f", all significant at the {int(alpha * 100)}% level.")
            if ns:
                parts.append("The effects of " + ", ".join(n for n, *_ in ns[:8]) + f" are not statistically significant at the {int(alpha * 100)}% level.")
    return " ".join(x for x in parts if x)


def _numbers(text: str) -> list:
    out = []
    for m in NUM_RX.finditer(str(text).replace("−", "-").replace("٫", ".")):
        tok = m.group(0).replace(",", "")
        try:
            out.append((m.group(0), float(tok)))
        except ValueError:
            pass
    return out


def source_numbers(results: list) -> np.ndarray:
    vals = []
    for r in results:
        for kind, content in r.get("sections", []):
            if isinstance(content, pd.DataFrame):
                for v in content.to_numpy().ravel():
                    if isinstance(v, (int, float, np.integer, np.floating)) and np.isfinite(v):
                        vals.append(float(v))
                    elif isinstance(v, str):
                        vals += [x for _, x in _numbers(v)]
            elif isinstance(content, str):
                vals += [x for _, x in _numbers(strip_html(content))]
        for t in r.get("insights", []):
            vals += [x for _, x in _numbers(strip_html(t))]
    return np.array(vals, float)


def check_numbers(text: str, results: list, ignore_small_ints: bool = True) -> list:
    """Numbers in `text` that cannot be traced (to the precision written, incl. % ↔ fraction) to the computed
    results. Small integers (≤ 12: counts, lags, list numbering) and years are ignored by default."""
    src = source_numbers(results)
    bad = []
    for raw, v in _numbers(text):
        if ignore_small_ints and float(v).is_integer() and (abs(v) <= 12 or 1800 <= abs(v) <= 2100):
            continue
        if abs(v) in (0.001, 0.01, 0.05, 0.1, 0.10, 90, 95, 99, 0.95, 0.9, 0.99):   # conventional significance / CI levels
            continue
        dec = len(raw.split(".")[1]) if "." in raw and "e" not in raw.lower() else 0
        tol = 0.5 * 10 ** (-dec) + 1e-12
        cands = np.r_[src, src * 100, src / 100, np.abs(src), np.abs(src) * 100]
        if src.size == 0 or not np.any(np.abs(cands - v) <= tol * 1.0001 + 1e-9 * abs(v)):
            bad.append(raw)
    return bad


# ======================================================================================================================
# Assembling the document model
# ======================================================================================================================
def assemble(items: list, meta: dict, lang: str = "en") -> dict:
    """Turn report items into a flat document model: blocks (heading / paragraph / table / figure / note / bullets)
    with numbered tables and figures, and a merged, alphabetised reference list."""
    blocks, refs = [], []
    tno = fno = 0
    sec = 0
    T = {"table": "Table" if lang != "ar" else "جدول", "figure": "Figure" if lang != "ar" else "شكل"}
    for it in items:
        kind = it["kind"]
        if kind == "heading":
            sec += 1
            blocks.append(("h1", f"{sec}. {it['title']}"))
            continue
        if kind == "text":
            if it.get("title"):
                blocks.append(("h2", it["title"]))
            blocks.append(("p", it.get("text", "")))
            continue
        r = it["result"]
        sec += 1
        blocks.append(("h1", f"{sec}. {it.get('title') or r['title']}"))
        if it.get("text"):
            blocks.append(("p", it["text"]))
        if it.get("include_insights", True):
            for ins in r.get("insights", []):
                blocks.append(("p", strip_html(ins)))
        cur_h = None
        for k, content in r.get("sections", []):
            if k == "h":
                cur_h = strip_html(content)
                blocks.append(("h2", cur_h))
            elif k == "p":
                blocks.append(("p", strip_html(content)))
            elif k == "note":
                blocks.append(("note", strip_html(content)))
            elif k == "table" and it.get("include_tables", True) and isinstance(content, pd.DataFrame) and len(content):
                tno += 1
                cap = getattr(content, "attrs", {}).get("caption") or (cur_h if cur_h and cur_h.lower() not in ("methodology", "المنهجية") else r["title"])
                tb = _fmt_df(content)
                if lang == "ar":
                    tb = _arabic_table(tb)
                blocks.append(("table", f"{T['table']} {tno}. {cap}", tb))
            elif k == "bullets":
                lst = [strip_html(x) for x in content]
                rf = [x for x in lst if is_reference(x)]
                refs += rf
                rest = [x for x in lst if x not in rf]
                if rest:
                    blocks.append(("bullets", rest))
        if it.get("include_figures", True):
            for fg in r.get("figures", []):
                fno += 1
                blocks.append(("figure", f"{T['figure']} {fno}. {fg.get('caption') or fg['name'].replace('_', ' ')}", fg["json"]))
    # drop empty headings left behind (e.g. a "Methodology" heading whose text was not included)
    clean = []
    for i, b in enumerate(blocks):
        if b[0] == "h2" and (i + 1 == len(blocks) or blocks[i + 1][0] in ("h1", "h2")):
            continue
        clean.append(b)
    refs = sorted(dict.fromkeys(refs), key=lambda s: s.lower())
    return {"meta": meta, "blocks": clean, "references": refs, "lang": lang, "n_tables": tno, "n_figures": fno}


def _arabic_table(df: pd.DataFrame) -> pd.DataFrame:
    """Arabic headers and category labels for an Arabic report (the platform's on-screen glossary)."""
    try:
        from analytics import ts_ui as tu
        cols, vals = dict(tu._AR_COLS), dict(tu._AR_VALS)
    except Exception:  # noqa: BLE001
        return df
    extra = {"Estimate": "التقدير", "Std. Err.": "الخطأ المعياري", "p-value": "القيمة الاحتمالية", "CI low": "الحد الأدنى", "CI high": "الحد الأعلى",
             "Term": "الحد", "Variable": "المتغير", "Method": "الطريقة", "z": "z", "t": "t", "N": "N"}
    cols = {**extra, **cols}
    out = df.rename(columns={c: cols.get(c, c) for c in df.columns})
    lrm = "\u200e"                              # keep "< 0.001" and negative numbers left-to-right inside RTL cells
    for c in out.columns:
        out[c] = out[c].map(lambda v: vals.get(v, v)).map(lambda v: lrm + v if isinstance(v, str) and v[:1] in "-<−0123456789" else v)
    return out


def _meta_lines(meta, lang):
    date = meta.get("date") or _dt.date.today().isoformat()
    return [x for x in (meta.get("author"), meta.get("affiliation"), date) if x]


# ======================================================================================================================
# Word
# ======================================================================================================================
def to_docx(doc_model: dict, apa: bool = True) -> bytes:
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Inches, Pt

    from analytics.reg_report import _booktabs
    meta, lang = doc_model["meta"], doc_model["lang"]
    rtl = lang == "ar"
    doc = Document()
    stl = doc.styles["Normal"]
    stl.font.name = meta.get("font", "Times New Roman")
    stl.font.size = Pt(12)
    stl.element.rPr.rFonts.set(qn("w:cs"), meta.get("font", "Times New Roman"))

    def bidi(p):
        if rtl:
            pPr = p._p.get_or_add_pPr()
            b = OxmlElement("w:bidi")
            b.set(qn("w:val"), "1")
            pPr.append(b)
            for r in p.runs:
                rPr = r._r.get_or_add_rPr()
                rt = OxmlElement("w:rtl")
                rt.set(qn("w:val"), "1")
                rPr.append(rt)
        return p
    t = doc.add_heading(meta.get("title") or "Report", level=0)
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    bidi(t)
    for line in _meta_lines(meta, lang):
        p = doc.add_paragraph(line)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        bidi(p)
    if meta.get("abstract"):
        bidi(doc.add_heading("الملخص" if rtl else "Abstract", level=1))
        p = doc.add_paragraph(meta["abstract"])
        bidi(p)
    if meta.get("keywords"):
        p = doc.add_paragraph(("الكلمات المفتاحية: " if rtl else "Keywords: ") + meta["keywords"])
        p.runs[0].italic = True
        bidi(p)
    for b in doc_model["blocks"]:
        if b[0] == "h1":
            bidi(doc.add_heading(b[1], level=1))
        elif b[0] == "h2":
            bidi(doc.add_heading(b[1], level=2))
        elif b[0] == "p":
            p = doc.add_paragraph(b[1])
            p.paragraph_format.first_line_indent = Inches(0.4) if apa else None
            bidi(p)
        elif b[0] == "note":
            p = doc.add_paragraph(b[1])
            for r in p.runs:
                r.font.size = Pt(9.5)
                r.italic = True
            bidi(p)
        elif b[0] == "bullets":
            for x in b[1]:
                bidi(doc.add_paragraph(x, style="List Bullet"))
        elif b[0] == "table":
            cap = doc.add_paragraph()
            run = cap.add_run(b[1])
            run.bold = not apa
            run.italic = apa
            bidi(cap)
            df = b[2]
            tb = doc.add_table(rows=1, cols=len(df.columns))
            for i, c in enumerate(df.columns):
                cell = tb.rows[0].cells[i]
                cell.text = str(c)
                for r in cell.paragraphs[0].runs:
                    r.bold = True
            for _, row in df.iterrows():
                cells = tb.add_row().cells
                for i, v in enumerate(row.values):
                    cells[i].text = str(v)
            for row in tb.rows:
                for cell in row.cells:
                    for p in cell.paragraphs:
                        for r in p.runs:
                            r.font.size = Pt(9)
            _booktabs(tb)
            if rtl:
                tblPr = tb._tbl.tblPr
                bv = OxmlElement("w:bidiVisual")
                bv.set(qn("w:val"), "1")
                tblPr.append(bv)
            doc.add_paragraph("")
        elif b[0] == "figure":
            png = fig_png(b[2])
            if png:
                doc.add_picture(io.BytesIO(png), width=Inches(6.0))
                doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            cap = doc.add_paragraph()
            run = cap.add_run(b[1] + ("" if png else (" — (الشكل متاح في نسخة HTML)" if rtl else " — (figure available in the HTML version)")))
            run.italic = True
            cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            bidi(cap)
    if doc_model["references"]:
        bidi(doc.add_heading("المراجع" if rtl else "References", level=1))
        for r in doc_model["references"]:
            p = doc.add_paragraph(r)
            p.paragraph_format.left_indent = Inches(0.5)
            p.paragraph_format.first_line_indent = Inches(-0.5)
    if meta.get("note"):
        p = doc.add_paragraph(meta["note"])
        for r in p.runs:
            r.font.size = Pt(9)
            r.italic = True
        bidi(p)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


# ======================================================================================================================
# PDF (fpdf2 + HarfBuzz shaping for Arabic; DejaVu Sans from matplotlib's bundled fonts)
# ======================================================================================================================
def _font_paths():
    import matplotlib
    base = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
    return os.path.join(base, "DejaVuSans.ttf"), os.path.join(base, "DejaVuSans-Bold.ttf")


def to_pdf(doc_model: dict) -> bytes:
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos
    meta, lang = doc_model["meta"], doc_model["lang"]
    rtl = lang == "ar"
    reg, bold = _font_paths()
    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(True, margin=18)
    pdf.add_font("dv", "", reg)
    pdf.add_font("dv", "B", bold)
    try:
        pdf.set_text_shaping(use_shaping_engine=True, direction="rtl" if rtl else "ltr")
        shaped = True
    except Exception:  # noqa: BLE001  (uharfbuzz missing)
        shaped = False
    pdf.add_page()
    W = pdf.w - pdf.l_margin - pdf.r_margin
    al = "R" if rtl else "L"

    def direction(txt):
        if shaped:
            pdf.set_text_shaping(use_shaping_engine=True, direction="rtl" if AR_RX.search(str(txt)) else "ltr")

    def para(txt, size=10.5, style="", align=None, h=5.6):
        pdf.set_font("dv", style, size)
        direction(txt)
        ar = bool(AR_RX.search(txt))
        if align is None:
            align = "R" if ar else ("L" if rtl else "J")
        elif align in ("R", "L") and not ar and rtl:
            align = "L"                                  # Latin-only lines in an Arabic report stay left-to-right
        pdf.multi_cell(W, h, txt, align=align, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    para(meta.get("title") or "Report", 17, "B", "C", 9)
    for line in _meta_lines(meta, lang):
        para(line, 10, "", "C")
    pdf.ln(3)
    if meta.get("abstract"):
        para("الملخص" if rtl else "Abstract", 12, "B", al)
        para(meta["abstract"], 10)
    if meta.get("keywords"):
        para(("الكلمات المفتاحية: " if rtl else "Keywords: ") + meta["keywords"], 9.5)
    for b in doc_model["blocks"]:
        if b[0] == "h1":
            pdf.ln(3)
            para(b[1], 13.5, "B", al, 7)
        elif b[0] == "h2":
            pdf.ln(1)
            para(b[1], 11.5, "B", al, 6.5)
        elif b[0] == "p":
            para(b[1])
            pdf.ln(1.5)
        elif b[0] == "note":
            para(b[1], 8.5)
        elif b[0] == "bullets":
            for x in b[1]:
                para("• " + x, 10)
        elif b[0] == "table":
            pdf.ln(1)
            para(b[1], 9.5, "B", al)
            df = b[2]
            cols = list(df.columns)
            if rtl:
                cols = cols[::-1]
            pdf.set_font("dv", "", 7.5)
            widths = np.array([max(pdf.get_string_width(str(c)), *(pdf.get_string_width(str(v)) for v in df[c].head(60))) + 3 for c in cols])
            widths = widths * min(1.0, W / widths.sum())
            pdf.set_draw_color(40, 40, 40)
            x0 = pdf.l_margin + (W - widths.sum() if rtl else 0)
            x1 = x0 + widths.sum()
            y0 = pdf.get_y()
            pdf.line(x0, y0, x1, y0)
            pdf.set_font("dv", "B", 7.5)
            pdf.set_x(x0)
            for c, w in zip(cols, widths):
                direction(c)
                pdf.cell(w, 5.5, str(c)[:40], align="C")
            pdf.ln(5.5)
            pdf.line(x0, pdf.get_y(), x1, pdf.get_y())
            pdf.set_font("dv", "", 7.5)
            for _, row in df.head(200).iterrows():
                if pdf.get_y() > pdf.h - 25:
                    pdf.add_page()
                pdf.set_x(x0)
                for c, w in zip(cols, widths):
                    direction(row[c])
                    pdf.cell(w, 4.8, str(row[c])[:45], align="C")
                pdf.ln(4.8)
            pdf.line(x0, pdf.get_y(), x1, pdf.get_y())
            pdf.ln(3)
        elif b[0] == "figure":
            png = fig_png(b[2])
            if png:
                if pdf.get_y() > pdf.h - 100:
                    pdf.add_page()
                pdf.image(io.BytesIO(png), x=pdf.l_margin + W * 0.075, w=W * 0.85)
            para(b[1], 8.5, "", "C")
            pdf.ln(2)
    if doc_model["references"]:
        pdf.ln(3)
        para("المراجع" if rtl else "References", 13.5, "B", al, 7)
        for r in doc_model["references"]:
            para(r, 9, "", "L", 4.8)
    if meta.get("note"):
        para(meta["note"], 8)
    return bytes(pdf.output())


# ======================================================================================================================
# LaTeX (zip: main.tex + figures) and HTML
# ======================================================================================================================
_TEX_ESC = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\^{}",
            "\\": r"\textbackslash{}"}


def tex_escape(s: str) -> str:
    return "".join(_TEX_ESC.get(ch, ch) for ch in str(s))


def to_latex_zip(doc_model: dict) -> bytes:
    meta, lang = doc_model["meta"], doc_model["lang"]
    rtl = lang == "ar"
    L = [r"\documentclass[11pt,a4paper]{article}", r"\usepackage{booktabs,graphicx,longtable,geometry,caption}", r"\geometry{margin=2.5cm}"]
    if rtl:
        L += [r"% Compile with XeLaTeX (Arabic via polyglossia; Amiri font: https://github.com/aliftype/amiri)", r"\usepackage{fontspec}",
              r"\usepackage{polyglossia}", r"\setmainlanguage{arabic}", r"\setotherlanguage{english}",
              r"\newfontfamily\arabicfont[Script=Arabic]{Amiri}"]
    else:
        L += [r"\usepackage[utf8]{inputenc}", r"\usepackage[T1]{fontenc}", r"\usepackage{lmodern}"]
    L += [rf"\title{{{tex_escape(meta.get('title') or 'Report')}}}", rf"\author{{{tex_escape(meta.get('author', ''))}" +
          (rf"\\ \small {tex_escape(meta['affiliation'])}" if meta.get("affiliation") else "") + "}", rf"\date{{{tex_escape(meta.get('date') or _dt.date.today().isoformat())}}}",
          r"\begin{document}", r"\maketitle"]
    if meta.get("abstract"):
        L += [r"\begin{abstract}", tex_escape(meta["abstract"]), r"\end{abstract}"]
    files = {}
    for b in doc_model["blocks"]:
        if b[0] == "h1":
            L.append(rf"\section*{{{tex_escape(b[1])}}}")
        elif b[0] == "h2":
            L.append(rf"\subsection*{{{tex_escape(b[1])}}}")
        elif b[0] == "p":
            L += [tex_escape(b[1]), ""]
        elif b[0] == "note":
            L += [r"{\footnotesize\itshape " + tex_escape(b[1]) + "}", ""]
        elif b[0] == "bullets":
            L += [r"\begin{itemize}"] + [r"\item " + tex_escape(x) for x in b[1]] + [r"\end{itemize}"]
        elif b[0] == "table":
            df = b[2]
            cap = re.sub(r"^(Table|جدول) \d+\. ", "", b[1])
            L += [r"\begin{table}[htbp]\centering\small", rf"\caption{{{tex_escape(cap)}}}", r"\begin{tabular}{" + "l" * len(df.columns) + "}", r"\toprule",
                  " & ".join(tex_escape(c) for c in df.columns) + r" \\", r"\midrule"]
            L += [" & ".join(tex_escape(v) for v in row) + r" \\" for row in df.astype(str).to_numpy()]
            L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        elif b[0] == "figure":
            png = fig_png(b[2])
            cap = re.sub(r"^(Figure|شكل) \d+\. ", "", b[1])
            if png:
                fn = f"figures/fig{len(files) + 1}.png"
                files[fn] = png
                L += [r"\begin{figure}[htbp]\centering", rf"\includegraphics[width=0.85\linewidth]{{{fn}}}", rf"\caption{{{tex_escape(cap)}}}", r"\end{figure}"]
    if doc_model["references"]:
        L += [r"\section*{" + ("المراجع" if rtl else "References") + "}", r"\begin{itemize}\setlength\itemsep{0.2em}"]
        L += [r"\item[] " + tex_escape(r) for r in doc_model["references"]] + [r"\end{itemize}"]
    L.append(r"\end{document}")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("main.tex", "\n".join(L) + "\n")
        for fn, data in files.items():
            z.writestr(fn, data)
    return buf.getvalue()


def to_html(doc_model: dict) -> str:
    import html as _h
    meta, lang = doc_model["meta"], doc_model["lang"]
    rtl = lang == "ar"
    out = [f"<!DOCTYPE html><html lang='{lang}' dir='{'rtl' if rtl else 'ltr'}'><head><meta charset='utf-8'><title>{_h.escape(meta.get('title') or 'Report')}</title>",
           "<script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script>",
           "<style>body{font-family:'Times New Roman',Amiri,serif;max-width:880px;margin:40px auto;line-height:1.6;color:#222;padding:0 16px}"
           "h1.t{text-align:center}.m{text-align:center;color:#555}table{border-collapse:collapse;margin:8px auto;font-size:13px}"
           "th,td{padding:3px 9px;text-align:center}thead tr{border-top:2px solid #000;border-bottom:1px solid #000}tbody tr:last-child{border-bottom:2px solid #000}"
           ".cap{font-style:italic;margin-top:14px}.note{font-size:12px;font-style:italic;color:#555}.ref{padding-inline-start:2em;text-indent:-2em}</style></head><body>",
           f"<h1 class='t'>{_h.escape(meta.get('title') or 'Report')}</h1>"]
    out += [f"<div class='m'>{_h.escape(x)}</div>" for x in _meta_lines(meta, lang)]
    if meta.get("abstract"):
        out += [f"<h2>{'الملخص' if rtl else 'Abstract'}</h2><p>{_h.escape(meta['abstract'])}</p>"]
    k = 0
    for b in doc_model["blocks"]:
        if b[0] in ("h1", "h2"):
            out.append(f"<{'h2' if b[0] == 'h1' else 'h3'}>{_h.escape(b[1])}</{'h2' if b[0] == 'h1' else 'h3'}>")
        elif b[0] == "p":
            out.append(f"<p>{_h.escape(b[1])}</p>")
        elif b[0] == "note":
            out.append(f"<p class='note'>{_h.escape(b[1])}</p>")
        elif b[0] == "bullets":
            out.append("<ul>" + "".join(f"<li>{_h.escape(x)}</li>" for x in b[1]) + "</ul>")
        elif b[0] == "table":
            out.append(f"<div class='cap'>{_h.escape(b[1])}</div>" + b[2].to_html(index=False, border=0, escape=True))
        elif b[0] == "figure":
            k += 1
            out.append(f"<div id='fig{k}'></div><script>(function(){{var f={b[2]};Plotly.newPlot('fig{k}',f.data,f.layout,{{displaylogo:false}});}})();</script>"
                       f"<div class='cap' style='text-align:center'>{_h.escape(b[1])}</div>")
    if doc_model["references"]:
        out.append(f"<h2>{'المراجع' if rtl else 'References'}</h2>" + "".join(f"<p class='ref'>{_h.escape(r)}</p>" for r in doc_model["references"]))
    if meta.get("note"):
        out.append(f"<p class='note'>{_h.escape(meta['note'])}</p>")
    out.append("</body></html>")
    return "\n".join(out)


def to_markdown(doc_model: dict) -> str:
    meta = doc_model["meta"]
    out = [f"# {meta.get('title') or 'Report'}", ""] + [f"*{x}*  " for x in _meta_lines(meta, doc_model['lang'])] + [""]
    if meta.get("abstract"):
        out += ["## " + ("الملخص" if doc_model["lang"] == "ar" else "Abstract"), meta["abstract"], ""]
    for b in doc_model["blocks"]:
        if b[0] == "h1":
            out += [f"## {b[1]}", ""]
        elif b[0] == "h2":
            out += [f"### {b[1]}", ""]
        elif b[0] in ("p", "note"):
            out += [b[1], ""]
        elif b[0] == "bullets":
            out += [f"- {x}" for x in b[1]] + [""]
        elif b[0] == "table":
            df = b[2]
            out += [f"**{b[1]}**", "", "| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
            out += ["| " + " | ".join(map(str, r)) + " |" for r in df.to_numpy()] + [""]
        elif b[0] == "figure":
            out += [f"*{b[1]}*", ""]
    if doc_model["references"]:
        out += ["## " + ("المراجع" if doc_model["lang"] == "ar" else "References"), ""] + [f"- {r}" for r in doc_model["references"]]
    return "\n".join(out) + "\n"


def summary_json(doc_model: dict) -> str:
    """Compact machine-readable view (used as grounded context for the AI assistant)."""
    out = {"title": doc_model["meta"].get("title"), "blocks": []}
    for b in doc_model["blocks"]:
        if b[0] == "table":
            out["blocks"].append({"table": b[1], "rows": b[2].head(40).to_dict("records")})
        elif b[0] in ("p", "h1", "h2", "note"):
            out["blocks"].append({b[0]: b[1][:1500]})
    return json.dumps(out, ensure_ascii=False)[:60000]
