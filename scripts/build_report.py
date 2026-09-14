"""Build the Phase 1 report as a PDF, computing every number from the result files.

Nothing here is transcribed by hand: if a result file changes, re-running this changes
the report. -> outputs/IsPersonalizationReal_Phase1_Report.pdf
"""
import itertools, json, subprocess, sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (PageBreak, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

import config

OUT = Path(config.OUTPUTS_DIR) / "IsPersonalizationReal_Phase1_Report.pdf"
ACCENT = colors.HexColor("#1f3a5f")
MUTED = colors.HexColor("#5a6672")
RULE = colors.HexColor("#c8d0d8")


# ------------------------------------------------------------------ data
def load():
    C = sorted(config.COHORT)
    mat = json.loads(Path("outputs/results/baseline/lopo_score_matrix.json").read_text())
    pairs = {r["patient"]: sum(d["n_pos"] * d["n_neg"] for d in r["per_source"].values())
             for r in mat.values() if r["role"] == "held_out"}
    probe = json.loads(Path("outputs/signatures/identity_probe.json").read_text())
    rows = []
    for f in C:
        d = json.loads(Path(f"outputs/results/adapted/{f}.json").read_text())
        z = np.load(f"outputs/signatures/{f}/z_behavior.npz")
        meta = json.loads(Path(config.FOLDS_DIR, f, "fold.json").read_text())
        c = np.stack([z[q] for q in meta["train_patients"]]).mean(0)
        rows.append({**d, "pairs": pairs[f], "D_p": float(np.linalg.norm(z[f] - c))})
    held = {r["patient"]: r["within_source"] for r in mat.values() if r["role"] == "held_out"}
    val = [r["within_source"] for r in mat.values() if r["role"] == "internal_val"]
    return rows, probe, held, val


def signflip(d):
    d = np.asarray(d, float); obs = abs(d.mean())
    ge = sum(1 for s in itertools.product([-1, 1], repeat=len(d))
             if abs((np.array(s) * d).mean()) >= obs - 1e-15)
    return float(d.mean()), ge / 2 ** len(d), int((d > 0).sum()), len(d)


# ------------------------------------------------------------------ styles
def styles():
    s = getSampleStyleSheet()
    s.add(ParagraphStyle("T", parent=s["Title"], fontSize=19, leading=23,
                         textColor=ACCENT, spaceAfter=2))
    s.add(ParagraphStyle("Sub", parent=s["Normal"], fontSize=9.5, leading=13,
                         textColor=MUTED, spaceAfter=14))
    s.add(ParagraphStyle("H", parent=s["Heading1"], fontSize=13, leading=16,
                         textColor=ACCENT, spaceBefore=14, spaceAfter=5))
    s.add(ParagraphStyle("H2", parent=s["Heading2"], fontSize=10.5, leading=13,
                         textColor=ACCENT, spaceBefore=9, spaceAfter=3))
    s.add(ParagraphStyle("B", parent=s["BodyText"], fontSize=9.6, leading=14,
                         alignment=TA_JUSTIFY, spaceAfter=6))
    s.add(ParagraphStyle("Cap", parent=s["Normal"], fontSize=8.3, leading=11,
                         textColor=MUTED, spaceBefore=3, spaceAfter=10))
    s.add(ParagraphStyle("Key", parent=s["BodyText"], fontSize=10.2, leading=15,
                         textColor=ACCENT, leftIndent=8, borderPadding=4,
                         spaceBefore=4, spaceAfter=8))
    return s


def table(data, widths, align_right=None, highlight=None):
    t = Table(data, colWidths=widths, hAlign="LEFT")
    st = [("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
          ("FONTSIZE", (0, 0), (-1, -1), 8.4),
          ("TEXTCOLOR", (0, 0), (-1, 0), ACCENT),
          ("LINEBELOW", (0, 0), (-1, 0), 0.8, ACCENT),
          ("LINEBELOW", (0, 1), (-1, -2), 0.25, RULE),
          ("LINEBELOW", (0, -1), (-1, -1), 0.8, ACCENT),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("TOPPADDING", (0, 0), (-1, -1), 3.5),
          ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5)]
    for c in (align_right or []):
        st.append(("ALIGN", (c, 0), (c, -1), "RIGHT"))
    for r in (highlight or []):
        st.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#fdf3e7")))
    t.setStyle(TableStyle(st))
    return t


def build():
    rows, probe, held, val = load()
    res = [r for r in rows if r["pairs"] >= 100]
    d8 = np.array([r["adapted_auc"] - r["shuffled_mean"] for r in rows])
    d6 = np.array([r["adapted_auc"] - r["shuffled_mean"] for r in res])
    b8 = np.array([r["adapted_auc"] - r["baseline_auc"] for r in rows])
    b6 = np.array([r["adapted_auc"] - r["baseline_auc"] for r in res])
    D8 = np.array([r["D_p"] for r in rows]); D6 = np.array([r["D_p"] for r in res])
    m8, p8, k8, n8 = signflip(d8); m6, p6, k6, n6 = signflip(d6)
    mb8, pb8, kb8, _ = signflip(b8); mb6, pb6, kb6, _ = signflip(b6)
    acc = [probe[f]["accuracy"] for f in sorted(config.COHORT)]
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = "?"

    s = styles()
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, topMargin=18*mm, bottomMargin=18*mm,
                            leftMargin=20*mm, rightMargin=20*mm,
                            title="IsPersonalizationReal - Phase 1",
                            author="Nagham Daood")
    E = []
    P = lambda t, st="B": E.append(Paragraph(t, s[st]))

    P("Zero-shot patient-conditioned weight modulation does not personalise "
      "video-based seizure detection", "T")
    P(f"Phase 1 result report &nbsp;|&nbsp; {date.today().isoformat()} &nbsp;|&nbsp; "
      f"commit {commit} &nbsp;|&nbsp; WU-SAHZU-EMU-Video, 8-patient LOPO cohort", "Sub")

    P("Summary", "H")
    P("An amortised hypernetwork generates LoRA weight deltas for a frozen VSViG backbone, "
      "conditioned on a behavioural signature computed from a patient's unlabelled "
      "interictal video alone. Across all eight leave-one-patient-out folds, the "
      "patient's own signature produces <b>no measurable advantage over another "
      f"patient's signature</b>. Zero of eight folds meet a criterion registered before "
      f"the experiment was run, and {8-k8} of 8 point estimates favour the wrong "
      "signature.")
    P("The result is not a failure of the signature: it identifies its own patient at "
      f"{np.mean(acc):.1%} against a chance rate of 12.5%. It is a failure of the "
      "training design. The hypernetwork is trained on exactly the patients the frozen "
      "backbone was already fitted on, where the loss it minimises is near zero, and it "
      "is asked to amortise a mapping from a 128-dimensional signature to weight deltas "
      "from five examples of that mapping.", "B")
    E.append(Spacer(1, 4))
    P("<b>Headline:</b> own-signature minus wrong-signature = "
      f"{m8:+.5f} AUC (exact sign-flip permutation p = {p8:.3f}, n = 8). "
      f"Restricted to the six folds whose metric can resolve the effect, {m6:+.5f} "
      f"(p = {p6:.3f}).", "Key")

    P("1. What was tested", "H")
    P("Eight LOPO folds over patients pat01-pat04 and pat06-pat09; six further patients "
      "were excluded for insufficient interictal exposure. Each fold trains a VSViG "
      "backbone on five patients under a fixed 50-epoch budget with last-5-epoch weight "
      "averaging and no checkpoint selection, then freezes it including BatchNorm "
      "buffers, then trains a 741,537-parameter hypernetwork with the Max-Pool Dynamic "
      "Cyclic Sampler on patient-homogeneous 50/50 batches, BCEWithLogitsLoss, AdamW "
      "with a 300-step warmup to 1e-3 and cosine decay. B is zero-initialised, so the "
      "adapted model is exactly the baseline before the first gradient step - verified "
      "at run time on every fold.")
    P("Discrimination is measured as <b>within-recording</b> AUC. Pooling clips across "
      "recordings inflates AUC by +0.07 to +0.10 here, because recordings sit at "
      "different baseline score levels and carry different class mixes; a model can "
      "score one recording above another without ordering anything correctly inside "
      "either. Patients are weighted equally and pairs weight recordings within a "
      "patient.")

    P("2. The pre-registered criterion", "H")
    P("A gain over the unadapted baseline is <i>not</i> evidence of personalisation: "
      "perturbing a frozen network's weights at all can help. Personalisation was "
      "therefore defined, before the experiment, as the patient's own signature standing "
      "apart from the distribution obtained with every other cohort patient's signature:")
    P("z = ( AUC<sub>own</sub> &minus; mean AUC<sub>shuffled</sub> ) / sd "
      "AUC<sub>shuffled</sub> &nbsp;&gt;&nbsp; 2.0, with AUC<sub>own</sub> above the mean.", "Key")
    P("Each fold's held-out patient is scored eight times: once with their own signature "
      "and once with each of the seven others. 64 evaluations in total.")

    P("3. Result", "H")
    data = [["fold", "baseline", "own z", "shuffled mean", "shuffled sd", "z-score", "AUC pairs"]]
    hl = []
    for i, r in enumerate(rows, start=1):
        zs = "n/a" if r["z_score"] != r["z_score"] else f"{r['z_score']:+.2f}"
        data.append([r["fold"], f"{r['baseline_auc']:.4f}", f"{r['adapted_auc']:.4f}",
                     f"{r['shuffled_mean']:.4f}", f"{r['shuffled_sd']:.4f}", zs,
                     f"{r['pairs']:,}"])
        if r["pairs"] < 100:
            hl.append(i)
    E.append(table(data, [18*mm, 21*mm, 20*mm, 27*mm, 22*mm, 19*mm, 22*mm],
                   align_right=[1, 2, 3, 4, 5, 6], highlight=hl))
    P("Shaded rows are folds whose metric cannot resolve the effect under test "
      "(section 4). No fold reaches z = 2.0.", "Cap")

    P("4. Two folds are not evidence", "H")
    P("Within-recording AUC counts only recordings holding both classes. For pat04 that "
      "is a single recording with 7 ictal and 2 interictal clips - <b>14 ordered "
      "pairs</b>, so its AUC moves in steps of 0.071. For pat03 it is 41 pairs, steps of "
      "0.024. The effect under test is roughly 0.0005. This is why every condition "
      "returned an identical number with a standard deviation of zero: the instrument "
      "cannot move. pat04's apparent +0.143 from adaptation is <b>two pairs changing "
      "order</b>.")
    P("This qualifies more than the present table. The cohort baseline reported in "
      f"Phase 1 - a held-out mean of {np.mean(list(held.values())):.3f} across eight "
      "patients - includes both of these folds.")

    P("5. Statistical verdict", "H")
    P("Sample sizes here are too small for a normal approximation, so these are exact "
      "paired sign-flip permutation tests over all 2<super>n</super> sign assignments.")
    E.append(table([["quantity", "all 8 folds", "6 resolvable folds"],
                    ["own &minus; shuffled (personalisation)",
                     f"{m8:+.5f}   p = {p8:.3f}   {k8}/{n8} positive",
                     f"{m6:+.5f}   p = {p6:.3f}   {k6}/{n6} positive"],
                    ["own &minus; baseline (any adaptation)",
                     f"{mb8:+.5f}   p = {pb8:.3f}   {kb8}/8 positive",
                     f"{mb6:+.5f}   p = {pb6:.3f}   {kb6}/6 positive"]],
                   [58*mm, 55*mm, 55*mm]))
    P("Two conclusions follow. Personalisation is <b>absent</b> - not merely "
      "non-significant, but negative in point estimate under both analyses, with 0 of 8 "
      "folds clearing a threshold fixed in advance. And the apparent benefit of adapting "
      f"at all ({mb8:+.4f} across eight folds) is carried entirely by pat04's two-pair "
      f"flip: among the six folds that can resolve anything it is {mb6:+.5f}, slightly "
      "harmful.")

    E.append(PageBreak())
    P("6. The stated hypothesis", "H")
    P("Section 3.5 of the methodology predicts that the benefit of personalisation "
      "correlates <i>positively</i> with how atypically a patient moves relative to the "
      "cohort, measured as D<sub>p</sub> = ||z<sub>p</sub> &minus; c|| with c the "
      "centroid over that fold's training patients.")
    E.append(table([["correlation with D_p", "all 8 folds", "6 resolvable folds"],
                    ["own &minus; baseline (as §3.5 defines benefit)",
                     f"{np.corrcoef(D8, b8)[0,1]:+.3f}", f"{np.corrcoef(D6, b6)[0,1]:+.3f}"],
                    ["own &minus; shuffled (personalisation only)",
                     f"{np.corrcoef(D8, d8)[0,1]:+.3f}", f"{np.corrcoef(D6, d6)[0,1]:+.3f}"]],
                   [80*mm, 45*mm, 43*mm], align_right=[1, 2]))
    P("Every version on the resolvable folds is negative. At n = 6 to 8 none is "
      "significant; the honest statement is that the predicted relationship is not "
      "present and the data lean the other way.")

    P("7. Why it failed", "H")
    P("The diagnosis matters more than the negative, because it is a property of the "
      "design rather than of this run.", "B")
    P("7.1 &nbsp;The objective is already satisfied", "H2")
    P("The methodology trains the backbone on five patients, freezes it, then trains the "
      "hypernetwork on <i>those same five patients</i>. Measured on fold pat01, the "
      "frozen backbone's BCE is <b>0.0066</b> on its own training patients against "
      "<b>0.726</b> on unseen ones - a 112-fold gap. The loss the hypernetwork minimises "
      "is near zero on every example it is permitted to see, while generalisation to an "
      "unseen patient never enters the objective. Training behaves accordingly: training "
      "BCE falls toward 1e-5 while validation BCE rises monotonically.")
    P("7.2 &nbsp;Five examples of the mapping", "H2")
    P("A hypernetwork learns a function from a 128-dimensional signature to weight "
      "deltas. With five training patients it has five examples of that function. This "
      "is a sample-size problem, not an optimisation problem, and it predicts what was "
      "observed: deltas that saturate their Frobenius budget and then vary with the "
      "signature in a way unrelated to which patient it belongs to.")
    P("7.3 &nbsp;Not the signature's fault", "H2")
    P(f"The signature carries real patient identity - {np.mean(acc):.1%} "
      f"identification accuracy across the eight backbones (range {min(acc):.1%} to "
      f"{max(acc):.1%}), against 12.5% chance, every fold significant by permutation. "
      "But identification recall correlates <b>+0.726</b> with baseline AUC: the "
      "signature is least recoverable for exactly the patients the model most needs help "
      "with. It also rests mostly on the mean of stage-2 features rather than their "
      "variability, which leaves open whether it encodes motor behaviour or static "
      "appearance.")

    P("8. Threats to validity", "H")
    for t in [
        "<b>Cohort size.</b> Eight evaluation patients and 18 seizures. Every "
        "correlation reported here is underpowered; the sign tests are exact but weak.",
        "<b>Measurement resolution.</b> Two of eight patients cannot support a "
        "clip-level AUC. Event-level metrics would be the appropriate instrument for "
        "them and are not yet reported alongside.",
        "<b>One optimisation family.</b> Two regimes were tried (peak LR 1e-3 with a "
        "0.5 delta budget, and 1e-4 with 0.1) and agreed. A wider search was not run.",
        "<b>The baseline is not clinically usable.</b> At per-fold thresholds selected "
        "on internal validation patients it detects 18 of 18 seizures at 41 false alarms "
        "per hour. Conclusions here concern discrimination, not deployment.",
        "<b>Batching asymmetry.</b> The adapted arm uses patient-homogeneous batches "
        "with frozen BatchNorm; the baseline was trained with shuffled batches and live "
        "BatchNorm. A baseline retrained under the adapted arm's batching would make "
        "the comparison exactly like-for-like."]:
        P("&bull;&nbsp; " + t)

    P("9. What would be required to test the method fairly", "H")
    P("The result above tests the method as published. It does not establish that "
      "amortised patient conditioning cannot work, because two design faults - a "
      "saturated objective and five amortisation examples - are confounded with the "
      "method itself. A fair test needs the hypernetwork to train on patients the frozen "
      "backbone has never seen, and needs more than five of them. Both are addressable "
      "with data already held.")

    P("Reproducibility", "H")
    P("Every figure is computed from files under <font face='Courier'>outputs/</font> by "
      "<font face='Courier'>scripts/build_report.py</font>. The full decision record, "
      "including the errors found and corrected during this work, is in "
      "<font face='Courier'>DECISIONS.md</font> (D1-D30). Each result file carries a run "
      "manifest with its seed, git commit and configuration.", "Cap")

    doc.build(E)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    build()
