"""Builds Handwritten_Digit_Recognition.pptx from the real results in results/."""
import json, numpy as np
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

R = "results/"
h = json.load(open(R + "cnn_history.json"))
cl = json.load(open(R + "classical_results.json"))
rep = json.load(open(R + "cnn_report.json"))
cm = np.array(rep["confusion"])
acc = h["test_acc"] * 100
errors = rep["errors"]
epochs = len(h["loss"])
best_val = max(h["val_accuracy"]) * 100
per_class = {d: rep["report"][str(d)]["f1-score"] for d in range(10)}
worst = min(per_class, key=per_class.get); best = max(per_class, key=per_class.get)
off = cm.copy(); np.fill_diagonal(off, 0)
(ca, cb) = np.unravel_index(off.argmax(), off.shape)
macro_f1 = rep["report"]["macro avg"]["f1-score"]
svm = cl["SVM RBF (PCA-50)"]; best_classical = max(cl, key=lambda k: cl[k]["accuracy"])
al = json.load(open(R + "alnum_report.json")); ah = json.load(open(R + "alnum_history.json"))
a_acc, a_dig, a_let = al["accuracy"] * 100, al["digit_accuracy"] * 100, al["letter_accuracy"] * 100
pairs = al["top_pairs"]
pair_txt = ", ".join(f"{p['true']}\u2192{p['pred']}" for p in pairs[:4])

def C(hex_): return RGBColor.from_string(hex_)
BG, INK, BLUE, ORANGE, MUTED, CARD, LINE, PALE = "F7F4EE", "1F2A37", "2F5D7C", "C0623B", "5F6B7A", "FFFFFF", "D9D4C7", "E4ECF1"
HEAD, BODY = "Georgia", "Calibri"

prs = Presentation(); prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
blank = prs.slide_layouts[6]
TOTAL = 10


def bg(s, color=BG):
    f = s.background.fill; f.solid(); f.fore_color.rgb = C(color)


def rect(s, x, y, w, h_, fill=CARD, line=LINE, shape=MSO_SHAPE.RECTANGLE, lw=0.75):
    sh = s.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h_))
    sh.fill.solid(); sh.fill.fore_color.rgb = C(fill)
    if line: sh.line.color.rgb = C(line); sh.line.width = Pt(lw)
    else: sh.line.fill.background()
    sh.shadow.inherit = False
    return sh


def text(s, x, y, w, h_, paras, size=16, color=INK, bold=False, font=BODY, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, space=6, shape=None):
    """paras: str or list of str / (str, dict) / list-of-runs. dict keys: size,color,bold,font,bullet"""
    tb = shape or s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h_))
    tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.08); tf.margin_top = tf.margin_bottom = Inches(0.04)
    if isinstance(paras, str): paras = [paras]
    for i, p in enumerate(paras):
        o = {}
        if isinstance(p, tuple): p, o = p
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align; para.space_after = Pt(o.get("space", space))
        runs = p if isinstance(p, list) else [p]
        if o.get("bullet"):
            runs = [("▪  ", {"color": ORANGE, "bold": True})] + list(runs)
        for r in runs:
            ro = {}
            if isinstance(r, tuple): r, ro = r
            run = para.add_run(); run.text = r
            f = run.font; f.name = ro.get("font", o.get("font", font)); f.size = Pt(ro.get("size", o.get("size", size)))
            f.bold = ro.get("bold", o.get("bold", bold)); f.color.rgb = C(ro.get("color", o.get("color", color)))
    return tb


def bullets(s, x, y, w, h_, items, size=16, space=8):
    return text(s, x, y, w, h_, [(i, {"bullet": True}) if not isinstance(i, tuple) else i for i in items], size=size, space=space)


def header(s, title, n, kicker=None):
    bg(s)
    rect(s, 0, 0, 0.28, 7.5, fill=BLUE, line=None)
    if kicker: text(s, 0.7, 0.32, 8, 0.35, kicker.upper(), size=12, color=ORANGE, bold=True)
    text(s, 0.7, 0.6, 11.5, 0.8, title, size=32, font=HEAD, bold=True)
    rect(s, 0.78, 1.38, 0.9, 0.05, fill=ORANGE, line=None)
    text(s, 0.7, 7.02, 9, 0.3, "Handwritten Digit Recognition Using Machine Learning", size=10.5, color=MUTED)
    text(s, 11.6, 7.02, 1.2, 0.3, f"{n} / {TOTAL}", size=10.5, color=MUTED, align=PP_ALIGN.RIGHT)


def card(s, x, y, w, h_, title=None, fill=CARD):
    rect(s, x, y, w, h_, fill=fill)
    if title:
        text(s, x + 0.15, y + 0.1, w - 0.3, 0.4, title, size=15, bold=True, color=BLUE)


def stat(s, x, y, w, big, small, color=BLUE, size=30):
    rect(s, x, y, w, 1.35, fill=CARD)
    rect(s, x, y, 0.08, 1.35, fill=color, line=None)
    text(s, x + 0.2, y + 0.08, w - 0.3, 0.7, big, size=size, font=HEAD, bold=True, color=color)
    text(s, x + 0.2, y + 0.78, w - 0.3, 0.5, small, size=13, color=MUTED)


def mini(s, x, y, w, big, small, color=BLUE):
    rect(s, x, y, w, 0.95, fill=PALE, line=None)
    text(s, x, y + 0.04, w, 0.5, big, size=22, font=HEAD, bold=True, color=color, align=PP_ALIGN.CENTER)
    text(s, x, y + 0.56, w, 0.35, small, size=11, color=MUTED, align=PP_ALIGN.CENTER)


def pic(s, f, x, y, w=None, h_=None):
    return s.shapes.add_picture(R + f, Inches(x), Inches(y), Inches(w) if w else None, Inches(h_) if h_ else None)


def notes(s, t): s.notes_slide.notes_text_frame.text = t


def flow_box(s, x, y, w, h_, title, sub, fill=CARD, tcolor=INK, line=LINE):
    sh = rect(s, x, y, w, h_, fill=fill, line=line, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    sh.adjustments[0] = 0.08
    text(s, x, y + 0.08, w, h_ - 0.1, [(title, {"bold": True, "size": 14, "color": tcolor, "space": 3}),
                                         (sub, {"size": 11.5, "color": tcolor if fill != CARD else MUTED})],
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)


def arrow(s, x, y, w=0.35, color=ORANGE):
    a = rect(s, x, y, w, 0.3, fill=color, line=None, shape=MSO_SHAPE.RIGHT_ARROW)
    return a


# ---------------------------------------------------------------- 1 Title
s = prs.slides.add_slide(blank); bg(s, BLUE)
rect(s, 0, 0, 13.333, 7.5, fill=BLUE, line=None)
rect(s, 8.55, 0, 4.783, 7.5, fill="264C66", line=None)
# decorative digit grid (right panel)
digits = "3 7 1 9 4 0 8 2 5 6".split()
for i, d in enumerate(digits):
    cx, cy = 8.95 + (i % 2) * 2.05, 0.7 + (i // 2) * 1.3
    rect(s, cx, cy, 1.75, 1.1, fill="2F5D7C" if i % 3 else "3A6B8C", line=None)
    text(s, cx, cy, 1.75, 1.1, d, size=44, font=HEAD, color="F7F4EE", align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
text(s, 0.8, 0.8, 7, 0.4, "MINI PROJECT  |  MACHINE LEARNING", size=13, bold=True, color="F0B79C")
text(s, 0.8, 1.45, 7.5, 2.2, "Handwritten Digit Recognition Using Machine Learning", size=40, font=HEAD, bold=True, color="FFFFFF")
text(s, 0.8, 3.65, 7.4, 0.9, "A CNN benchmarked against four classical ML models on MNIST digits, then extended to recognise letters A\u2013Z (36 classes) with EMNIST",
     size=16, color="D6E2EA")
rect(s, 0.88, 4.7, 0.9, 0.05, fill=ORANGE, line=None)
text(s, 0.8, 4.85, 4, 0.35, "PRESENTED BY", size=11.5, bold=True, color="F0B79C")
team = [("Yogisha Rani", "2430256"), ("Upasna Dey", ""), ("Shreyansh Ray", "2430257"), ("Suyash Gupta", "")]
for i, (n, r) in enumerate(team):
    x, y = 0.8 + (i % 2) * 3.7, 5.3 + (i // 2) * 0.7
    text(s, x, y, 3.6, 0.65, [(n, {"size": 18, "bold": True, "color": "FFFFFF", "space": 0}),
                              (r if r else " ", {"size": 12, "color": "D6E2EA"})])
notes(s, "Introduce the topic and team members.")

# ---------------------------------------------------------------- 2 Overview
s = prs.slides.add_slide(blank); header(s, "Project Overview", 2, "Introduction")
card(s, 0.7, 1.75, 5.9, 2.3, "Problem Statement")
text(s, 0.85, 2.2, 5.6, 1.8, "Automatically recognise a handwritten digit (0-9) or letter (A-Z) from an image. Handwriting varies in slant, thickness, "
     "size and style, so fixed rules fail – the system must learn the patterns from examples.", size=15.5, color=INK)
card(s, 0.7, 4.25, 5.9, 2.55, "Objectives")
bullets(s, 0.85, 4.65, 5.6, 2.2, [
    "Build an end-to-end digit recognition pipeline",
    "Compare 4 classical ML models with a CNN",
    "Reach > 99% accuracy on unseen test data",
    "Extend to letters: recognise 0\u20139 and A\u2013Z live"], size=15.5, space=5)
stat(s, 6.95, 1.75, 2.8, f"{acc:.2f}%", "digit accuracy (MNIST)", ORANGE)
stat(s, 9.95, 1.75, 2.7, "250,000+", "images (MNIST + EMNIST)")
stat(s, 6.95, 3.3, 2.8, f"{a_acc:.1f}%", "digits + letters (36 classes)")
stat(s, 9.95, 3.3, 2.7, f"{errors}", "digit errors out of 10,000", ORANGE)
card(s, 6.95, 4.85, 5.7, 1.95, "Why it matters")
text(s, 7.1, 5.25, 5.4, 1.5, "Digit recognition powers cheque processing, postal PIN-code sorting, form digitisation and exam-sheet scanning. "
     "It is also the standard benchmark for testing image-classification ideas.", size=14.5)
notes(s, "Explain the problem, objectives, and headline results.")

# ---------------------------------------------------------------- 3 Dataset
s = prs.slides.add_slide(blank); header(s, "Dataset & Data Preprocessing", 3, "Data")
card(s, 0.7, 1.75, 6.2, 2.05, "Datasets: MNIST (digits) + EMNIST (letters)")
bullets(s, 0.85, 2.15, 5.95, 1.65, [
    "MNIST: 70,000 digit images, 28 \u00D7 28 grayscale, 10 classes",
    "Split: 55,000 train  |  5,000 validation  |  10,000 test",
    f"EMNIST: {ah['n_train']+ah['n_val']:,} train / {ah['n_test']:,} test images, 36 classes (0\u20139, A\u2013Z; a = A)",
    "Test sets are never used during training or tuning"], size=13, space=2)
card(s, 0.7, 3.95, 6.2, 2.9, "Samples: MNIST (top), EMNIST (bottom)")
pic(s, "samples.png", 1.35, 4.4, w=4.9)
pic(s, "emnist_samples.png", 1.35, 5.58, w=4.9)
card(s, 7.15, 1.75, 5.5, 5.1, "Preprocessing Steps")
steps = [("1", "Normalise", "pixels 0–255 scaled to 0–1"),
         ("2", "Reshape", "28×28×1 tensor for CNN; flattened 784 vector for classical models"),
         ("3", "Dimensionality reduction", "PCA to 50 components for k-NN and SVM"),
         ("4", "Augmentation", "random rotation ±22°, shift 8%, zoom 10% (training only)"),
         ("5", "Real-input conversion", "invert, Otsu threshold, crop, resize to 20×20, centre in 28×28 by centre of mass")]
for i, (n, t, d) in enumerate(steps):
    y = 2.3 + i * 0.88
    c = rect(s, 7.3, y + 0.05, 0.5, 0.5, fill=ORANGE, line=None, shape=MSO_SHAPE.OVAL)
    text(s, 7.3, y + 0.05, 0.5, 0.5, n, size=15, bold=True, color="FFFFFF", align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    text(s, 7.9, y - 0.02, 4.65, 0.85, [(t, {"bold": True, "size": 14.5, "space": 0}), (d, {"size": 12.5, "color": MUTED})])
notes(s, "MNIST details and the preprocessing chain, including conversion of user drawings to MNIST format.")

# ---------------------------------------------------------------- 4 Architecture
s = prs.slides.add_slide(blank); header(s, "Proposed System Architecture", 4, "Architecture")
# Training pipeline
rect(s, 0.7, 1.7, 11.95, 2.2, fill=PALE, line=None)
text(s, 0.85, 1.75, 6, 0.35, "A.  TRAINING PHASE (offline)", size=12.5, bold=True, color=BLUE)
tr = [("MNIST + EMNIST", "digits and letters"), ("Preprocessing", "normalise, 55k/5k/10k split"), ("Augmentation", "rotate, shift, zoom"),
      ("CNN Training", "Adam, early stopping"), ("Trained Models", "digit CNN + 36-class CNN")]
bw, gap = 1.95, 0.46
for i, (t, d) in enumerate(tr):
    x = 0.9 + i * (bw + gap)
    flow_box(s, x, 2.2, bw, 1.45, t, d, fill=BLUE if i == 3 else CARD, tcolor="FFFFFF" if i == 3 else INK, line=None if i == 3 else LINE)
    if i < 4: arrow(s, x + bw + 0.04, 2.78, 0.35)
# Inference pipeline
rect(s, 0.7, 4.05, 11.95, 2.2, fill="F3E5DD", line=None)
text(s, 0.85, 4.1, 6, 0.35, "B.  RECOGNITION PHASE (live)", size=12.5, bold=True, color=ORANGE)
inf = [("User Input", "digit or letter, drawn / image"), ("Image Preparation", "invert, threshold, crop, centre"), ("28×28 Image", "MNIST-style input"),
       ("Trained CNN", "classifies 0\u20139 and A\u2013Z"), ("Predicted Character", "digit or letter + confidence")]
for i, (t, d) in enumerate(inf):
    x = 0.9 + i * (bw + gap)
    flow_box(s, x, 4.55, bw, 1.45, t, d, fill=ORANGE if i == 3 else CARD, tcolor="FFFFFF" if i == 3 else INK, line=None if i == 3 else LINE)
    if i < 4: arrow(s, x + bw + 0.04, 5.13, 0.35, color=BLUE)
# link: trained model used by recognition CNN
a = rect(s, 8.94, 3.62, 0.32, 0.5, fill=MUTED, line=None, shape=MSO_SHAPE.DOWN_ARROW)
text(s, 9.3, 3.68, 3, 0.35, "same weights reused", size=11.5, color=MUTED)
text(s, 0.7, 6.4, 11.95, 0.55, [[("Evaluation: ", {"bold": True, "color": BLUE}),
     "held-out test sets (10,000 MNIST and 35,526 EMNIST images) measure accuracy, F1 and confusion matrices; the digit CNN is also compared with 4 classical baselines."]], size=14)
notes(s, "Top row = how the model is built. Bottom row = how a new digit is recognised. The CNN weights from training are reused.")

# ---------------------------------------------------------------- 5 Methodology
s = prs.slides.add_slide(blank); header(s, "Methodology / Working Procedure", 5, "Method")
steps = [("Collect", "Load MNIST and\ncreate train /\nvalidation / test"),
         ("Prepare", "Scale pixels, reshape,\naugment training\nimages"),
         ("Baselines", "Train LR, RF,\nk-NN, SVM on\nflattened pixels"),
         ("Build CNN", "Stacked Conv +\nBatchNorm + Pool\n+ Dropout"),
         ("Train", "Adam optimiser,\nLR decay, early\nstopping"),
         ("Evaluate", "Accuracy, F1,\nconfusion matrix,\nerror analysis"),
         ("Deploy", "Drawing-pad app\nwith live\nprediction")]
w = 1.62
for i, (t, d) in enumerate(steps):
    x = 0.7 + i * 1.72
    rect(s, x, 1.85, w, 0.62, fill=BLUE if i != 3 else ORANGE, line=None, shape=MSO_SHAPE.PENTAGON)
    text(s, x, 1.85, w - 0.2, 0.62, f"{i+1}. {t}", size=14, bold=True, color="FFFFFF", align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    rect(s, x, 2.6, w, 1.45, fill=CARD)
    text(s, x, 2.65, w, 1.4, d, size=12.5, color=INK, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
card(s, 0.7, 4.3, 5.85, 2.55, "How the CNN recognises a digit")
bullets(s, 0.85, 4.72, 5.6, 2.1, [
    "Early convolution layers detect edges and strokes",
    "Deeper layers combine them into loops, curves and corners",
    "Pooling keeps the strongest features and tolerates small shifts",
    "Softmax gives a probability for each of the 10 digits"], size=14, space=4)
card(s, 6.8, 4.3, 5.85, 2.55, "Training & validation strategy")
bullets(s, 6.95, 4.72, 5.6, 2.1, [
    "Loss: sparse categorical cross-entropy",
    "Validation set monitored every epoch; best weights restored",
    "Learning rate halved when validation stalls",
    "Test set used only once for the final report"], size=14, space=4)
notes(s, "Walk through the seven stages left to right.")

# ---------------------------------------------------------------- 6 Model development
s = prs.slides.add_slide(blank); header(s, "Model Development & Implementation", 6, "Implementation")
text(s, 0.7, 1.65, 8, 0.35, "CNN layer structure", size=15, bold=True, color=BLUE)
layers = [("Input", "28×28×1", PALE), ("Conv 3×3 × 2", "32 filters, BN, ReLU", CARD), ("MaxPool + Dropout", "0.25", CARD),
          ("Conv 3×3 × 2", "64 filters, BN, ReLU", CARD), ("MaxPool + Dropout", "0.25", CARD),
          ("Dense", "128 units, Dropout 0.4", CARD), ("Softmax", "10 or 36 classes", "F3E5DD")]
lw = 1.5
for i, (t, d, f) in enumerate(layers):
    x = 0.7 + i * 1.68
    flow_box(s, x, 2.05, lw, 1.25, t, d, fill=f)
    if i < 6: arrow(s, x + lw + 0.0, 2.55, 0.17)
card(s, 0.7, 3.6, 5.85, 3.25, "Hyper-parameters")
rows = [("Optimiser", "Adam (lr 0.001, halves on plateau)"), ("Batch size", "128"), ("Epochs", f"digits {epochs} (max 30), letters {len(ah['loss'])} (max 12)"),
        ("Parameters", f"{h['params']:,} / {ah['params']:,} (36-class)"), ("Augmentation", "rotation, shift, zoom"), ("Random seed", "42 (reproducible)")]
for i, (k, v) in enumerate(rows):
    y = 4.1 + i * 0.44
    text(s, 0.85, y, 1.9, 0.4, k, size=13.5, bold=True, color=MUTED)
    text(s, 2.75, y, 3.8, 0.4, v, size=13.5)
card(s, 6.8, 3.6, 5.85, 3.25, "Tools & code modules")
bullets(s, 6.95, 4.05, 5.6, 2.8, [
    [("Python 3.10", {"bold": True}), "  –  TensorFlow / Keras, scikit-learn"],
    [("OpenCV, SciPy", {"bold": True}), "  –  image preparation"],
    [("Matplotlib, Seaborn", {"bold": True}), "  –  result plots"],
    [("Tkinter", {"bold": True}), "  –  drawing-pad application"],
    "Modules: data.py, preprocess.py, prepare_emnist.py, train_cnn.py, train_alnum.py, app.py"], size=13.5, space=4)
notes(s, "Describe the layers and the hyper-parameters.")

# ---------------------------------------------------------------- 7 Results (digits + letters)
s = prs.slides.add_slide(blank); header(s, "Experimental Results & Performance", 7, "Results")
card(s, 0.7, 1.75, 5.9, 5.1, "Digits (MNIST): CNN vs classical models")
pic(s, "model_comparison.png", 0.85, 2.2, w=5.6)
for i, (big, small) in enumerate([(f"{acc:.2f}%", "accuracy"), (f"{macro_f1*100:.2f}%", "macro F1-score"), (f"{errors}", "errors / 10,000")]):
    mini(s, 0.9 + i * 1.9, 5.15, 1.8, big, small, BLUE if i == 1 else ORANGE)
text(s, 0.85, 6.15, 5.6, 0.7, f"The CNN beats the best classical model (SVM, {cl[best_classical]['accuracy']*100:.2f}%) by "
     f"{acc-cl[best_classical]['accuracy']*100:.2f} points. Most frequent digit error: {ca} predicted as {cb}.", size=12.5)
card(s, 6.8, 1.75, 5.85, 5.1, "Letters (EMNIST): 36-class CNN")
pic(s, "alnum_f1.png", 6.95, 2.2, w=5.55)
text(s, 6.95, 4.15, 5.55, 0.3, "F1-score per class (blue = digits, orange = letters)", size=10.5, color=MUTED)
for i, (big, small) in enumerate([(f"{a_acc:.2f}%", "all 36 classes"), (f"{a_dig:.2f}%", "digits only"), (f"{a_let:.2f}%", "letters only")]):
    mini(s, 6.95 + i * 1.9, 4.55, 1.8, big, small, BLUE if i == 1 else ORANGE)
text(s, 6.95, 5.6, 5.6, 1.2, f"Look-alike characters cause most errors (true\u2192predicted): {pair_txt}. "
     f"{al['digit_letter_confusions']:,} of {al['errors']:,} errors are digit\u2194letter mix-ups.", size=12.5)
notes(s, "Left: five models on MNIST digits. Right: the same CNN design with a 36-way output trained on EMNIST. Letters are harder because O/0, I/1, L/1 look alike.")

# ---------------------------------------------------------------- 10 Application
s = prs.slides.add_slide(blank); header(s, "Application", 8, "Application")
card(s, 0.7, 1.75, 5.9, 5.1, "Live Drawing-Pad Application (app.py)")
for i, t in enumerate(["Draw any digit or letter with the mouse on the black canvas",
                       "On release, the stroke image is converted to MNIST format by preprocess.py",
                       "The CNN predicts instantly; a switch picks digits-only (99.3%) or digits + letters",
                       "Top-3 probabilities are displayed as confidence bars",
                       "Clear and try again; also available as predict.py for image files"]):
    y = 2.3 + i * 0.88
    rect(s, 0.9, y + 0.05, 0.45, 0.45, fill=BLUE, line=None, shape=MSO_SHAPE.OVAL)
    text(s, 0.9, y + 0.05, 0.45, 0.45, str(i + 1), size=14, bold=True, color="FFFFFF", align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    text(s, 1.5, y - 0.02, 5.0, 0.85, t, size=14, anchor=MSO_ANCHOR.MIDDLE)
card(s, 6.85, 1.75, 5.8, 3.1, "Real-world use cases")
bullets(s, 7.0, 2.2, 5.5, 2.6, ["Bank cheque and amount reading", "Postal PIN-code sorting", "Digitising forms, tax and survey sheets",
                                "Automatic grading of numeric answers", "Number-plate and meter reading (with extension)"], size=14.5, space=5)
card(s, 6.85, 5.05, 5.8, 1.8, "Deployment notes", fill=PALE)
text(s, 7.0, 5.45, 5.5, 1.4, f"The model is only {h['params']/1e3:.0f}k parameters, so it runs on an ordinary laptop CPU in milliseconds "
     "and can be converted to TensorFlow Lite for mobile apps.", size=14)
notes(s, "Demonstrate app.py live if possible.")

# ---------------------------------------------------------------- 9 Conclusion, limitations, future scope
s = prs.slides.add_slide(blank); header(s, "Conclusion, Limitations & Future Scope", 9, "Summary")
stat(s, 0.7, 1.7, 3.85, f"{acc:.2f}%", "digit accuracy (10,000 unseen MNIST images)", ORANGE)
stat(s, 4.75, 1.7, 3.85, f"{a_acc:.2f}%", "accuracy on 36 classes (digits + letters)", BLUE)
stat(s, 8.8, 1.7, 3.85, f"{(acc-cl[best_classical]['accuracy']*100):+.2f} pts", "digit CNN gain over best classical (SVM)", ORANGE)
card(s, 0.7, 3.25, 3.85, 3.6, "Conclusion")
bullets(s, 0.85, 3.7, 3.6, 3.1, [
    "Built a full pipeline: data, five models, evaluation and a drawing app",
    f"CNN outperformed every classical model on digits ({acc:.2f}%)",
    f"Extended to A\u2013Z: {a_acc:.1f}% on 36 classes; errors are mostly look-alikes",
    "Centring and scaling inputs is what makes it work on user drawings"], size=14, space=6)
card(s, 4.75, 3.25, 3.85, 3.6, "Limitations")
bullets(s, 4.9, 3.7, 3.6, 3.1, [
    "Single, centred characters only; words and long numbers need segmentation",
    f"Look-alikes ({pair_txt}) cause most errors",
    "Upper and lower case merged (a = A); no symbols",
    "Thin strokes, noise and unusual writing styles reduce accuracy"], size=14, space=6)
card(s, 8.8, 3.25, 3.85, 3.6, "Future Scope")
bullets(s, 8.95, 3.7, 3.6, 3.1, [
    "Case-sensitive recognition (62 classes); Devanagari / Bengali digits",
    "Character segmentation to read words, numbers and cheques",
    "Ensembles or ResNet for lower error",
    "Mobile / web app with TensorFlow Lite; fine-tune on user drawings"], size=14, space=6)
notes(s, "Summarise results, be honest about limits, then outline extensions.")

# ---------------------------------------------------------------- 12 Thank you
s = prs.slides.add_slide(blank); bg(s, BLUE)
rect(s, 0, 0, 13.333, 7.5, fill=BLUE, line=None)
rect(s, 0, 5.2, 13.333, 2.3, fill="264C66", line=None)
text(s, 0, 1.9, 13.333, 1.3, "Thank You", size=60, font=HEAD, bold=True, color="FFFFFF", align=PP_ALIGN.CENTER)
rect(s, 6.17, 3.35, 1.0, 0.06, fill=ORANGE, line=None)
text(s, 0, 3.6, 13.333, 0.6, "Questions are welcome", size=22, color="D6E2EA", align=PP_ALIGN.CENTER)
text(s, 0, 5.55, 13.333, 0.4, "Yogisha Rani   |   Upasna Dey   |   Shreyansh Ray   |   Suyash Gupta", size=16, color="FFFFFF", align=PP_ALIGN.CENTER)
text(s, 0, 6.05, 13.333, 0.4, "Handwritten Digit Recognition Using Machine Learning", size=13, color="D6E2EA", align=PP_ALIGN.CENTER)

prs.save("Handwritten_Digit_Recognition.pptx")
print("saved")
