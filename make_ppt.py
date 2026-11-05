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
TOTAL = 13


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
text(s, 0.85, 2.2, 5.6, 1.8, "Automatically recognise a handwritten digit (0-9) from an image. Handwriting varies in slant, thickness, "
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
card(s, 0.7, 1.75, 6.2, 2.15, "MNIST Dataset")
bullets(s, 0.85, 2.15, 5.95, 1.8, [
    "70,000 grayscale images, 28 × 28 pixels, 10 classes",
    "Split: 55,000 train  |  5,000 validation  |  10,000 test",
    "Letters extension uses EMNIST (36 classes) \u2013 see slide 9",
    "Test set is never used during training or tuning"], size=14.5, space=4)
card(s, 0.7, 4.1, 6.2, 2.75, "Sample Images")
pic(s, "samples.png", 0.85, 4.65, w=5.9)
card(s, 7.15, 1.75, 5.5, 5.1, "Preprocessing Steps")
steps = [("1", "Normalise", "pixels 0–255 scaled to 0–1"),
         ("2", "Reshape", "28×28×1 tensor for CNN; flattened 784 vector for classical models"),
         ("3", "Dimensionality reduction", "PCA to 50 components for k-NN and SVM"),
         ("4", "Augmentation", "random rotation ±6°, shift 8%, zoom 10% (training only)"),
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

# ---------------------------------------------------------------- 7 Results comparison
s = prs.slides.add_slide(blank); header(s, "Experimental Results & Performance", 7, "Results")
pic(s, "model_comparison.png", 0.9, 1.65, w=6.7)
card(s, 8.3, 1.7, 4.35, 3.0, "Test set metrics (10,000 images)")
m = [("Accuracy", f"{acc:.2f}%"), ("Macro F1-score", f"{macro_f1*100:.2f}%"), ("Misclassified", f"{errors} images"),
     ("Best val. accuracy", f"{best_val:.2f}%")]
for i, (k, v) in enumerate(m):
    text(s, 8.45, 2.2 + i * 0.6, 2.4, 0.5, k, size=14.5, color=MUTED)
    text(s, 10.6, 2.2 + i * 0.6, 1.95, 0.5, v, size=17, bold=True, color=ORANGE, align=PP_ALIGN.RIGHT)
card(s, 0.7, 5.2, 11.95, 1.65, "Key observations")
bullets(s, 0.85, 5.6, 11.7, 1.3, [
    f"The CNN is the most accurate model: {acc:.2f}% versus {cl[best_classical]['accuracy']*100:.2f}% for the best classical model ({best_classical}).",
    f"Cost: the CNN trained for {epochs} epochs (about 1 minute each on a laptop CPU); SVM needed {svm['train_s']:.0f}s. Extra training time buys {acc-svm['accuracy']*100:.2f} points of accuracy.",
    "Convolutions learn spatial features directly from pixels, while classical models treat each pixel independently."], size=13.5, space=3)
notes(s, "Compare all five models on the same test split.")

# ---------------------------------------------------------------- 8 Detailed analysis
s = prs.slides.add_slide(blank); header(s, "Detailed Performance Analysis", 8, "Results")
pic(s, "training_curves.png", 0.7, 1.7, w=6.6)
pic(s, "confusion_matrix.png", 7.6, 1.6, h_=3.95)
text(s, 0.7, 4.45, 6.6, 0.4, "Validation accuracy stays above training accuracy because augmentation and dropout act only while training – no over-fitting.", size=12.5, color=MUTED)
text(s, 7.6, 5.55, 4.9, 0.3, "Confusion matrix on test set", size=12.5, color=MUTED)
card(s, 0.7, 5.15, 6.6, 1.7, "Error analysis")
bullets(s, 0.85, 5.55, 6.35, 1.3, [
    f"Best class: digit {best} (F1 {per_class[best]*100:.2f}%); weakest: digit {worst} (F1 {per_class[worst]*100:.2f}%)",
    f"Most frequent confusion: true {ca} predicted as {cb} ({off[ca, cb]} cases) – visually similar shapes",
    f"Most of the {errors} errors are sloppy or ambiguous digits that humans also find hard"], size=12.5, space=2)
card(s, 7.6, 5.9, 5.05, 0.95)
text(s, 7.7, 5.95, 4.85, 0.9, [[("Result: ", {"bold": True, "color": ORANGE}), f"{10000-errors:,} of 10,000 unseen digits classified correctly."]],
     size=14, anchor=MSO_ANCHOR.MIDDLE)
notes(s, "Convergence, confusion matrix and where the model fails.")


# ---------------------------------------------------------------- 9 Letters extension
s = prs.slides.add_slide(blank); header(s, "Extension: Recognising Letters (EMNIST)", 9, "Digits + letters")
card(s, 0.7, 1.75, 5.9, 2.35, "EMNIST dataset (NIST Special Database 19)")
bullets(s, 0.85, 2.15, 5.65, 1.95, [
    f"36 classes: digits 0\u20139 and letters A\u2013Z",
    f"{ah['n_train']+ah['n_val']:,} training and {ah['n_test']:,} test images (28\u00D728)",
    "Upper and lower case merged (a = A) because shapes overlap",
    "Balanced: at most 5,000 training images per class"], size=13.5, space=3)
card(s, 0.7, 4.25, 5.9, 2.6, "Sample images (two per class)")
pic(s, "emnist_samples.png", 0.8, 4.75, w=5.7)
stat(s, 6.85, 1.75, 1.85, f"{a_acc:.2f}%", "all 36 classes", ORANGE, size=24)
stat(s, 8.78, 1.75, 1.85, f"{a_dig:.2f}%", "digits only", BLUE, size=24)
stat(s, 10.71, 1.75, 1.94, f"{a_let:.2f}%", "letters only", ORANGE, size=24)
pic(s, "alnum_f1.png", 6.85, 3.25, w=5.8)
text(s, 6.85, 5.28, 5.8, 0.3, "F1-score per class (blue = digits, orange = letters)", size=11.5, color=MUTED)
card(s, 6.85, 5.6, 5.8, 1.25, "Where it struggles")
text(s, 7.0, 5.95, 5.55, 0.9, f"Look-alike pairs cause most errors (true\u2192predicted): {pair_txt}. "
     f"Digit\u2194letter mix-ups: {al['digit_letter_confusions']} of {al['errors']:,} errors.", size=12.5)
notes(s, "Same CNN design with a 36-way softmax, trained on EMNIST. Accuracy is lower than digits-only because O/0, I/1, S/5, Z/2 look the same.")

# ---------------------------------------------------------------- 10 Application
s = prs.slides.add_slide(blank); header(s, "Application", 10, "Application")
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

# ---------------------------------------------------------------- 10 Conclusion
s = prs.slides.add_slide(blank); header(s, "Conclusion", 11, "Summary")
stat(s, 0.7, 1.75, 3.8, f"{acc:.2f}%", "digit accuracy (10,000 unseen MNIST images)", ORANGE)
stat(s, 4.75, 1.75, 3.8, f"{a_acc:.2f}%", "accuracy on 36 classes (digits + letters)", BLUE)
stat(s, 8.8, 1.75, 3.85, f"{(acc-cl[best_classical]['accuracy']*100):+.2f} pts", "digit CNN gain over best classical (SVM)", ORANGE)
card(s, 0.7, 3.4, 11.95, 3.45, "Key takeaways")
bullets(s, 0.85, 3.85, 11.7, 3.0, [
    "A complete pipeline was built: data preparation, five trained models, evaluation and a working drawing application.",
    f"The CNN with BatchNorm, Dropout and augmentation outperformed every classical model, reaching {acc:.2f}% test accuracy.",
    "Careful preprocessing (centring and scaling) is what makes the model work on digits drawn by real users, not only on MNIST.",
    f"Extended to 36 classes with EMNIST: {a_dig:.1f}% on digits and {a_let:.1f}% on letters; remaining errors are mostly look-alikes ({pair_txt}).",
    "The project shows that deep learning is the most effective approach for image-based recognition tasks."], size=15, space=7)
notes(s, "Summarise results and learning outcomes.")

# ---------------------------------------------------------------- 11 Limitations & future
s = prs.slides.add_slide(blank); header(s, "Limitations & Future Scope", 12, "Next steps")
card(s, 0.7, 1.75, 5.85, 5.1, "Limitations")
bullets(s, 0.85, 2.2, 5.6, 4.6, [
    "Trained on clean, single, centred characters \u2013 words and multi-digit numbers need segmentation",
    "MNIST writers are mostly American; other writing styles (e.g. regional digit shapes) may reduce accuracy",
    "Very thin strokes, heavy noise or unusual backgrounds hurt prediction",
    f"Look-alike characters ({pair_txt}) cause most errors, even for humans",
    "Upper and lower case are merged (a = A); symbols and punctuation are not supported"], size=14.5, space=7)
card(s, 6.8, 1.75, 5.85, 5.1, "Future Scope")
bullets(s, 6.95, 2.2, 5.6, 4.6, [
    "Case-sensitive recognition (62 classes) and Devanagari / Bengali digits",
    "Add character segmentation to read words, numbers and cheques",
    "Ensemble of several CNNs or a ResNet for even lower error",
    "Mobile / web deployment with TensorFlow Lite or TensorFlow.js",
    "Collect user drawings to fine-tune the model for real handwriting"], size=14.5, space=7)
notes(s, "Be honest about limits, then outline extensions.")

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
