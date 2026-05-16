from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

# ── Palette ───────────────────────────────────────────────────
INK    = colors.HexColor('#1a1209')
RED    = colors.HexColor('#c0392b')
GOLD   = colors.HexColor('#b7860b')
TEAL   = colors.HexColor('#1a6b5a')
MUTED  = colors.HexColor('#5a4e40')
RULE   = colors.HexColor('#d4c5a9')
DEFBG  = colors.HexColor('#f5f0e8')
HLBG   = colors.HexColor('#fff8e8')
GREENBG= colors.HexColor('#e8f5f0')
REDBG  = colors.HexColor('#fdf0f0')
DARKBG = colors.HexColor('#1e1a14')
BLUEBG = colors.HexColor('#eef4fb')
CODECLR= colors.HexColor('#c8d8b8')
PURPBG = colors.HexColor('#f5f0ff')
INFERBG= colors.HexColor('#f0f5ff')

W, H = A4

doc = SimpleDocTemplate(
    '/home/claude/Unit2_AI_Notes.pdf',
    pagesize=A4,
    leftMargin=2.0*cm, rightMargin=2.0*cm,
    topMargin=2.2*cm,  bottomMargin=2.2*cm,
    title='Unit 2 - Applications of AI',
)
DW = doc.width

# ── Styles ────────────────────────────────────────────────────
def S(name, **kw): return ParagraphStyle(name, **kw)

cov_tag  = S('cov_tag',  fontName='Helvetica',        fontSize=9,  textColor=MUTED, alignment=TA_CENTER, spaceAfter=6)
cov_h1   = S('cov_h1',   fontName='Helvetica-Bold',   fontSize=28, textColor=RED,   alignment=TA_CENTER, leading=34, spaceAfter=8)
cov_sub  = S('cov_sub',  fontName='Helvetica-Oblique',fontSize=11, textColor=MUTED, alignment=TA_CENTER, spaceAfter=10)

sec_h    = S('sec_h',  fontName='Helvetica-Bold', fontSize=15, textColor=RED,  spaceBefore=22, spaceAfter=4)
top_h    = S('top_h',  fontName='Helvetica-Bold', fontSize=12, textColor=TEAL, spaceBefore=16, spaceAfter=4, leftIndent=4)
sub_h    = S('sub_h',  fontName='Helvetica-Bold', fontSize=10, textColor=GOLD, spaceBefore=10, spaceAfter=3)

body     = S('body',  fontName='Times-Roman',    fontSize=10.5, textColor=INK,  leading=17, spaceAfter=6, alignment=TA_JUSTIFY)
blt      = S('blt',   fontName='Times-Roman',    fontSize=10.5, textColor=INK,  leading=16, spaceAfter=5, leftIndent=18, firstLineIndent=-12)
sblt     = S('sblt',  fontName='Times-Roman',    fontSize=10,   textColor=INK,  leading=14, spaceAfter=3, leftIndent=32, firstLineIndent=-12)
pros_s   = S('pros_s',fontName='Times-Roman',    fontSize=10.5, textColor=colors.HexColor('#1a5a2a'), leading=15, spaceAfter=4, leftIndent=16, firstLineIndent=-10)
cons_s   = S('cons_s',fontName='Times-Roman',    fontSize=10.5, textColor=colors.HexColor('#7a1a1a'), leading=15, spaceAfter=4, leftIndent=16, firstLineIndent=-10)

# Box label styles
def_lbl  = S('def_lbl',   fontName='Helvetica-Bold', fontSize=9,  textColor=GOLD,  spaceAfter=3, spaceBefore=2)
def_bod  = S('def_bod',   fontName='Times-Roman',    fontSize=10.5, textColor=INK, leading=17, spaceAfter=3, alignment=TA_JUSTIFY)
ex_lbl   = S('ex_lbl',    fontName='Helvetica-Bold', fontSize=9,  textColor=colors.HexColor('#7a5500'), spaceAfter=3)
ex_bod   = S('ex_bod',    fontName='Times-Roman',    fontSize=10.5, textColor=INK, leading=16, spaceAfter=3, alignment=TA_JUSTIFY)
inf_lbl  = S('inf_lbl',   fontName='Helvetica-Bold', fontSize=9,  textColor=colors.HexColor('#1a3a6a'), spaceAfter=3)
inf_bod  = S('inf_bod',   fontName='Times-Roman',    fontSize=10.5, textColor=colors.HexColor('#1a3a6a'), leading=16, spaceAfter=3, alignment=TA_JUSTIFY)
tip_lbl  = S('tip_lbl',   fontName='Helvetica-Bold', fontSize=9,  textColor=colors.HexColor('#5a1a5a'), spaceAfter=3)
tip_bod  = S('tip_bod',   fontName='Times-Roman',    fontSize=10, textColor=colors.HexColor('#5a1a5a'), leading=15, spaceAfter=3, alignment=TA_JUSTIFY)
diag_lbl = S('diag_lbl',  fontName='Helvetica-Bold', fontSize=9,  textColor=TEAL, spaceAfter=3)
diag_bod = S('diag_bod',  fontName='Times-Roman',    fontSize=10, textColor=TEAL, leading=15, spaceAfter=3)
kp_s     = S('kp_s',      fontName='Helvetica-Bold', fontSize=10.5, textColor=RED, leading=15, spaceAfter=4)
cod_lbl  = S('cod_lbl',   fontName='Helvetica-Bold', fontSize=9,  textColor=GOLD, spaceAfter=2)
cod_s    = S('cod_s',     fontName='Courier',        fontSize=8.5, textColor=CODECLR, leading=13, spaceAfter=1, leftIndent=4)
tbl_h    = S('tbl_h',     fontName='Helvetica-Bold', fontSize=8.5, textColor=colors.white, alignment=TA_CENTER)
tbl_c    = S('tbl_c',     fontName='Times-Roman',    fontSize=9,   textColor=INK, leading=13)
footer_s = S('footer_s',  fontName='Helvetica',      fontSize=7.5, textColor=MUTED, alignment=TA_CENTER)
step_s   = S('step_s',    fontName='Courier',        fontSize=8.5, textColor=CODECLR, leading=13, spaceAfter=1, leftIndent=4)

# ── Helpers ───────────────────────────────────────────────────
def HR(thick=0.5, clr=RULE): return HRFlowable(width='100%', thickness=thick, color=clr, spaceAfter=6, spaceBefore=6)
def sp(n=6): return Spacer(1, n)
def p(txt, st=body): return Paragraph(txt, st)
def b(t): return f'<b>{t}</b>'
def i(t): return f'<i>{t}</i>'
def bullet(txt, st=blt): return Paragraph(txt, st)
def sbullet(txt): return Paragraph(txt, sblt)
def pro(txt): return Paragraph(f'<b>+</b>  {txt}', pros_s)
def con(txt): return Paragraph(f'<b>-</b>  {txt}', cons_s)

def section_header(txt):
    return [p(txt, sec_h), HR(1.5, RED)]

def topic_header(txt):
    return p(txt, top_h)

def sub_header(txt):
    return p(txt, sub_h)

# ── Core box builders (matching sample doc style) ─────────────
def concept_box(title_text, *paras):
    """DEFINITION / CONCEPT box — gold left border, cream bg"""
    rows = [[p('DEFINITION / CONCEPT', def_lbl)],
            [p(f'<b>{title_text}</b>', def_bod)]]
    for pa in paras:
        rows.append([p(pa, def_bod)])
    t = Table(rows, colWidths=[DW])
    t.setStyle(TableStyle([
        ('BACKGROUND',   (0,0),(-1,-1), DEFBG),
        ('LINEBEFORE',   (0,0),(0,-1),  5, GOLD),
        ('BOX',          (0,0),(-1,-1), 0.4, RULE),
        ('LEFTPADDING',  (0,0),(-1,-1), 12),
        ('RIGHTPADDING', (0,0),(-1,-1), 12),
        ('TOPPADDING',   (0,0),(-1,-1), 8),
        ('BOTTOMPADDING',(0,0),(-1,-1), 8),
    ]))
    return t

def example_box(scenario_title, *paras):
    """REAL-WORLD EXAMPLE box — gold left border, yellow bg"""
    rows = [[p('REAL-WORLD EXAMPLE / SCENARIO', ex_lbl)],
            [p(f'<b>EXAMPLE: </b>{scenario_title}', ex_bod)]]
    for pa in paras:
        rows.append([p(pa, ex_bod)])
    t = Table(rows, colWidths=[DW])
    t.setStyle(TableStyle([
        ('BACKGROUND',   (0,0),(-1,-1), HLBG),
        ('LINEBEFORE',   (0,0),(0,-1),  5, GOLD),
        ('BOX',          (0,0),(-1,-1), 0.4, colors.HexColor('#e0cc88')),
        ('LEFTPADDING',  (0,0),(-1,-1), 12),
        ('RIGHTPADDING', (0,0),(-1,-1), 12),
        ('TOPPADDING',   (0,0),(-1,-1), 8),
        ('BOTTOMPADDING',(0,0),(-1,-1), 8),
    ]))
    return t

def inference_box(*paras):
    """INFERENCE / ANALYSIS box — blue left border, light blue bg"""
    rows = [[p('INFERENCE / ANALYSIS', inf_lbl)]]
    for pa in paras:
        rows.append([p(f'<b>INFERENCE: </b>{pa}', inf_bod)])
    t = Table(rows, colWidths=[DW])
    t.setStyle(TableStyle([
        ('BACKGROUND',   (0,0),(-1,-1), INFERBG),
        ('LINEBEFORE',   (0,0),(0,-1),  5, colors.HexColor('#2a5a9a')),
        ('BOX',          (0,0),(-1,-1), 0.4, colors.HexColor('#b8ccee')),
        ('LEFTPADDING',  (0,0),(-1,-1), 12),
        ('RIGHTPADDING', (0,0),(-1,-1), 12),
        ('TOPPADDING',   (0,0),(-1,-1), 8),
        ('BOTTOMPADDING',(0,0),(-1,-1), 8),
    ]))
    return t

def exam_tip_box(*paras):
    """EXAM TIP box — purple left border, light purple bg"""
    rows = [[p('EXAM TIP', tip_lbl)]]
    for pa in paras:
        rows.append([p(f'<i>{pa}</i>', tip_bod)])
    t = Table(rows, colWidths=[DW])
    t.setStyle(TableStyle([
        ('BACKGROUND',   (0,0),(-1,-1), PURPBG),
        ('LINEBEFORE',   (0,0),(0,-1),  5, colors.HexColor('#7a1a9a')),
        ('BOX',          (0,0),(-1,-1), 0.4, colors.HexColor('#ccaaee')),
        ('LEFTPADDING',  (0,0),(-1,-1), 12),
        ('RIGHTPADDING', (0,0),(-1,-1), 12),
        ('TOPPADDING',   (0,0),(-1,-1), 8),
        ('BOTTOMPADDING',(0,0),(-1,-1), 8),
    ]))
    return t

def diagram_box(title, body_txt):
    rows = [[p(f'DRAW THIS: {title}', diag_lbl)],
            [p(body_txt, diag_bod)]]
    t = Table(rows, colWidths=[DW])
    t.setStyle(TableStyle([
        ('BACKGROUND',   (0,0),(-1,-1), GREENBG),
        ('LINEBEFORE',   (0,0),(0,-1),  5, TEAL),
        ('BOX',          (0,0),(-1,-1), 0.8, TEAL),
        ('LEFTPADDING',  (0,0),(-1,-1), 12),
        ('RIGHTPADDING', (0,0),(-1,-1), 12),
        ('TOPPADDING',   (0,0),(-1,-1), 8),
        ('BOTTOMPADDING',(0,0),(-1,-1), 8),
    ]))
    return t

def formula_box(title, lines):
    rows = [[p(f'FORMULA: {title}', cod_lbl)]]
    for ln in lines:
        rows.append([p(ln if ln.strip() else ' ', cod_s)])
    t = Table(rows, colWidths=[DW])
    t.setStyle(TableStyle([
        ('BACKGROUND',   (0,0),(-1,-1), DARKBG),
        ('LINEBEFORE',   (0,0),(0,-1),  4, GOLD),
        ('LEFTPADDING',  (0,0),(-1,-1), 12),
        ('RIGHTPADDING', (0,0),(-1,-1), 12),
        ('TOPPADDING',   (0,0),(-1,-1), 8),
        ('BOTTOMPADDING',(0,0),(-1,-1), 8),
    ]))
    return t

def algo_box(title, steps):
    rows = [[p(f'ALGORITHM: {title}', cod_lbl)]]
    for txt in steps:
        rows.append([p(txt if txt.strip() else ' ', step_s)])
    t = Table(rows, colWidths=[DW])
    t.setStyle(TableStyle([
        ('BACKGROUND',   (0,0),(-1,-1), DARKBG),
        ('LINEBEFORE',   (0,0),(0,-1),  4, TEAL),
        ('LEFTPADDING',  (0,0),(-1,-1), 12),
        ('RIGHTPADDING', (0,0),(-1,-1), 12),
        ('TOPPADDING',   (0,0),(-1,-1), 8),
        ('BOTTOMPADDING',(0,0),(-1,-1), 8),
    ]))
    return t

def keypoint(txt):
    t = Table([[p(f'KEY POINT:  {txt}', kp_s)]], colWidths=[DW])
    t.setStyle(TableStyle([
        ('BACKGROUND',   (0,0),(-1,-1), REDBG),
        ('LINEBEFORE',   (0,0),(0,-1),  5, RED),
        ('BOX',          (0,0),(-1,-1), 0.4, colors.HexColor('#f5b8b8')),
        ('LEFTPADDING',  (0,0),(-1,-1), 12),
        ('RIGHTPADDING', (0,0),(-1,-1), 12),
        ('TOPPADDING',   (0,0),(-1,-1), 8),
        ('BOTTOMPADDING',(0,0),(-1,-1), 8),
    ]))
    return t

def pros_cons_box(pros_list, cons_list):
    phdr = p('ADVANTAGES / PROS', S('pch', fontName='Helvetica-Bold', fontSize=8, textColor=colors.HexColor('#1a5a2a'), spaceAfter=4))
    chdr = p('DISADVANTAGES / CONS', S('cch', fontName='Helvetica-Bold', fontSize=8, textColor=colors.HexColor('#7a1a1a'), spaceAfter=4))
    pc = [phdr] + [pro(x) for x in pros_list]
    cc = [chdr] + [con(x) for x in cons_list]
    mr = max(len(pc), len(cc))
    while len(pc) < mr: pc.append(p(''))
    while len(cc) < mr: cc.append(p(''))
    data = [[pc[i], cc[i]] for i in range(mr)]
    hw = DW/2 - 3
    t = Table(data, colWidths=[hw, hw])
    t.setStyle(TableStyle([
        ('BACKGROUND',   (0,0),(0,-1), colors.HexColor('#f0faf0')),
        ('BACKGROUND',   (1,0),(1,-1), colors.HexColor('#fdf5f5')),
        ('BOX',          (0,0),(0,-1), 0.4, colors.HexColor('#a8d8a8')),
        ('BOX',          (1,0),(1,-1), 0.4, colors.HexColor('#f5b8b8')),
        ('LEFTPADDING',  (0,0),(-1,-1), 8),
        ('RIGHTPADDING', (0,0),(-1,-1), 8),
        ('TOPPADDING',   (0,0),(-1,-1), 5),
        ('BOTTOMPADDING',(0,0),(-1,-1), 5),
        ('VALIGN',       (0,0),(-1,-1), 'TOP'),
        ('COLPADDING',   (0,0),(-1,-1), 4),
    ]))
    return t

def cmp_table(headers, rows, col_widths=None):
    hrow = [p(h, tbl_h) for h in headers]
    data = [hrow] + [[p(str(c), tbl_c) for c in row] for row in rows]
    cw = col_widths or [DW/len(headers)]*len(headers)
    t = Table(data, colWidths=cw)
    t.setStyle(TableStyle([
        ('BACKGROUND',    (0,0),(-1,0),  DARKBG),
        ('ROWBACKGROUNDS',(0,1),(-1,-1), [colors.white, colors.HexColor('#f5f0e8')]),
        ('GRID',          (0,0),(-1,-1), 0.3, RULE),
        ('LEFTPADDING',   (0,0),(-1,-1), 6),
        ('RIGHTPADDING',  (0,0),(-1,-1), 6),
        ('TOPPADDING',    (0,0),(-1,-1), 4),
        ('BOTTOMPADDING', (0,0),(-1,-1), 4),
        ('VALIGN',        (0,0),(-1,-1), 'TOP'),
    ]))
    return t

def trace_table(headers, rows, col_widths=None):
    mono = S('mono', fontName='Courier', fontSize=8.5, textColor=INK, leading=13)
    hrow = [p(h, tbl_h) for h in headers]
    data = [hrow] + [[p(str(c), mono) for c in row] for row in rows]
    cw = col_widths or [DW/len(headers)]*len(headers)
    t = Table(data, colWidths=cw)
    t.setStyle(TableStyle([
        ('BACKGROUND',    (0,0),(-1,0),  colors.HexColor('#2a3a2a')),
        ('TEXTCOLOR',     (0,0),(-1,0),  colors.white),
        ('ROWBACKGROUNDS',(0,1),(-1,-1), [colors.HexColor('#f8fdf8'), colors.HexColor('#eef8ee')]),
        ('GRID',          (0,0),(-1,-1), 0.3, colors.HexColor('#b8d8b8')),
        ('LEFTPADDING',   (0,0),(-1,-1), 5),
        ('RIGHTPADDING',  (0,0),(-1,-1), 5),
        ('TOPPADDING',    (0,0),(-1,-1), 4),
        ('BOTTOMPADDING', (0,0),(-1,-1), 4),
        ('VALIGN',        (0,0),(-1,-1), 'TOP'),
    ]))
    return t

# ══════════════════════════════════════════════════════════════
# STORY
# ══════════════════════════════════════════════════════════════
story = []

# ── COVER ─────────────────────────────────────────────────────
story += [sp(70),
    p('EMERGING ARTIFICIAL INTELLIGENCE  |  EXAM PREPARATION NOTES', cov_tag),
    p('Unit 2: Applications of AI', cov_h1),
    p('Supervised Machine Learning  ·  Dimension Reduction  ·  SOM  ·  Sammon Mapping', cov_sub),
    p('Recommender Systems  ·  Knowledge Modelling Using UML  ·  CPG Case Study', cov_sub),
    sp(10),
]
badge_data = [[p('15 Hours  |  All Topics  |  Worked Examples  |  Formulae  |  Exam Tips', S('bdg', fontName='Helvetica-Bold', fontSize=9, textColor=colors.white, alignment=TA_CENTER))]]
badge_t = Table(badge_data, colWidths=[DW])
badge_t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),RED),('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),8)]))
story.append(badge_t)
story.append(PageBreak())

# ── TOC ───────────────────────────────────────────────────────
story += section_header('Table of Contents')
for t in [
    '1.   Supervised Machine Learning — Definition, Goal, Setup, Evaluation',
    '2.   Issues of Supervised Learning — Noise, Missing Values, Overfitting, Bias-Variance',
    '3.   Decision Trees — Entropy, Information Gain, C4.5, Pruning',
    '4.   Neural Networks — Perceptron, ANN, Backpropagation',
    '5.   Naive Bayes — Bayes Theorem, Worked Example',
    '6.   k-Nearest Neighbour (kNN) — Euclidean Distance, Worked Example',
    '7.   Support Vector Machines (SVM) — Maximum Margin, Kernels',
    '8.   Combining Classifiers — Bagging, Boosting, Stacking',
    '9.   Overview of Dimension Reduction — PCA, MDS',
    '10.  Self-Organising Map (SOM) — Architecture, Training, Worked Example',
    '11.  Sammon Mapping — Stress Function, Gradient Descent',
    '12.  Recommender Systems — Content-Based, Collaborative, Hybrid',
    '13.  Knowledge Modelling Using UML — Stereotypes, Profile',
    '14.  Case Study: Clinical Practice Guideline (CPG) Recommendations',
]: story.append(bullet(t))
story.append(PageBreak())

# ══════════════════════════════════════════════════════════════
# TOPIC 1 — SUPERVISED MACHINE LEARNING
# ══════════════════════════════════════════════════════════════
story += section_header('1.  Supervised Machine Learning')

story.append(concept_box(
    'Supervised Machine Learning',
    'The goal of supervised learning is to build a concise model of the distribution of class labels in terms of predictor features. The resulting classifier is then used to assign class labels to testing instances where the values of the predictor features are known but the value of the class label is unknown.',
    'It is called "supervised" because every training instance is labelled — the correct output (class label) is provided and supervises the learning process. This contrasts with unsupervised learning where no labels exist and the algorithm must discover structure on its own.'
))
story.append(sp(6))

story.append(p(f'{b("DETAILED EXPLANATION")}', sub_h))
story.append(p('Supervised ML is the most widely used branch of machine learning in real-world AI. The process follows a clear pipeline:', body))
story.append(bullet(f'{b("Instance (Training Example)")}: A single data record described by a set of feature values. Each instance represents one observation from the world. Example: A single patient record with age=45, temperature=38.6°C, cough=Yes.'))
story.append(bullet(f'{b("Feature (Attribute)")}: A measurable property that describes an instance. Features can be: Continuous (real-valued — e.g., temperature), Categorical (discrete labels — e.g., blood type), Binary (0 or 1 — e.g., fever: Yes/No).'))
story.append(bullet(f'{b("Label (Class)")}: The output variable the classifier must predict. Example: "Flu" or "Not Flu." This is the supervised signal.'))
story.append(bullet(f'{b("Training Set")}: A collection of labelled instances used to build (train) the classifier. The classifier learns patterns from these examples.'))
story.append(bullet(f'{b("Test Set")}: A separate collection of instances used to evaluate how well the learned classifier generalises to unseen data. Labels are hidden during testing.'))
story.append(bullet(f'{b("Classifier")}: A function f: Features → Class Label, learned from training data. After training, it can predict labels for new, unseen instances.'))
story.append(bullet(f'{b("Generalisation")}: The ability of a classifier to correctly label instances it has never seen before. The ultimate goal of supervised learning.'))

story.append(sp(6))
story.append(p(f'{b("Evaluation Methods")}', sub_h))
story.append(bullet(f'{b("Hold-out Validation")}: Split data into 2/3 training and 1/3 testing. Simple but results vary depending on which instances happen to be in each split.'))
story.append(bullet(f'{b("k-Fold Cross-Validation")}: Divide data into k equal folds. Train on k-1 folds, test on the remaining fold. Repeat k times (each fold becomes the test set once). Final error = average of k error rates. Provides a more reliable error estimate than hold-out. k=10 is standard in practice.'))
story.append(bullet(f'{b("Leave-One-Out (LOO)")}: Special case of k-fold where k = total number of instances. Each instance is held out once as the test set. Most accurate but very computationally expensive.'))
story.append(formula_box('Prediction Accuracy', [
    'Accuracy = (Number of Correct Predictions / Total Predictions) x 100%',
    '',
    'Error Rate = 1 - Accuracy = (Misclassified instances / Total instances) x 100%',
    '',
    'k-Fold Cross-Validation Error:',
    '  E_CV = (1/k) x Sum of E_i    where E_i = error rate on fold i',
]))
story.append(sp(4))
story.append(diagram_box('Supervised ML Pipeline',
    'Draw a horizontal pipeline with 5 boxes connected by arrows: (1) RAW DATA → (2) PRE-PROCESSING (clean, normalise, feature selection) → (3) TRAINING (classifier learns from labelled data) → (4) MODEL (learned classifier: Decision Tree / ANN / SVM etc.) → (5) PREDICTION (classify new unlabelled instances). Below the pipeline, draw a feedback arrow from PREDICTION back to PRE-PROCESSING labeled "Error analysis / model improvement."'))

story.append(sp(4))
story.append(example_box(
    'Medical Diagnosis Supervised Learning',
    'Training data: 1000 patient records. Each record has features: Age, Temperature, Cough (Yes/No), WhiteBloodCount. Label: Diagnosis (Flu / Cold / Healthy).',
    'A Decision Tree classifier is trained on 700 records. Tested on the remaining 300. If 270 out of 300 are correctly classified, Accuracy = 270/300 = 90%. The classifier can now predict diagnosis for NEW patients based on their features alone.',
))
story.append(sp(4))
story.append(inference_box(
    'Supervised learning is the workhorse of applied AI. Every spam filter, fraud detection system, medical diagnostic tool, and recommendation engine at its core is a supervised classifier learning from labelled historical data. The quality of labels and the representativeness of training data are the most critical factors in classifier performance — "garbage in, garbage out" applies absolutely.'
))
story.append(sp(4))
story.append(exam_tip_box(
    'Supervised ML = learning from labelled data. Remember: Training Set (labelled) builds the model, Test Set (unlabelled) evaluates it. Cross-validation gives more reliable error estimates than hold-out. Accuracy = correct/total. These basics are frequently tested in short-answer and MCQ questions.'
))
story.append(PageBreak())

# ══════════════════════════════════════════════════════════════
# TOPIC 2 — ISSUES OF SUPERVISED LEARNING
# ══════════════════════════════════════════════════════════════
story += section_header('2.  Issues of Supervised Learning Algorithms')

story.append(concept_box(
    'Issues in Supervised Learning',
    'Before applying any supervised ML algorithm, several critical data quality and algorithmic issues must be addressed. These issues directly affect the accuracy, reliability, and generalisability of the learned classifier. The main categories are: Data Quality Issues (Noise, Missing Values, Irrelevant Features) and Algorithmic Issues (Overfitting, Underfitting, Bias-Variance Tradeoff).',
))
story.append(sp(6))

story.append(p(f'{b("DETAILED EXPLANATION")}', sub_h))
story.append(sub_header('Issue 1: Noise (Impossible / Unlikely Values)'))
story.append(bullet(f'{b("Definition")}: Noise refers to incorrect or corrupted feature values in the dataset. These are values that are physically impossible or statistically very unlikely given the domain.'))
story.append(bullet(f'{b("Causes")}: Data entry errors (typing mistakes), sensor malfunctions, transmission errors, outdated records.'))
story.append(bullet(f'{b("Example")}: A patient record shows age = -5 or temperature = 78°C — clearly incorrect values.'))
story.append(bullet(f'{b("Solutions")}: Outlier detection — identify values beyond 3 standard deviations from the mean. If the correct value cannot be re-entered, treat it as a missing value and handle accordingly.'))
story.append(bullet(f'{b("Impact")}: Noisy training data causes classifiers to learn incorrect patterns, reducing accuracy on clean test data.'))

story.append(sub_header('Issue 2: Missing Values'))
story.append(bullet(f'{b("Definition")}: A feature value is absent from a record because it was: not recorded (forgotten), not applicable (a feature that doesn\'t apply to this instance), or deliberately left blank ("don\'t know").'))
story.append(bullet(f'{b("Solution 1 — Delete")}: Remove instances with missing values. Risk: if many instances have missing values, the training set shrinks significantly.'))
story.append(bullet(f'{b("Solution 2 — Imputation")}: Replace missing values with a statistical estimate — mean (for continuous features), mode (for categorical features), or median (for skewed data).'))
story.append(bullet(f'{b("Solution 3 — Algorithm-level")}: Use algorithms that natively handle missing values (e.g., C4.5 decision trees can route instances with missing values probabilistically).'))
story.append(bullet(f'{b("Impact")}: Ignoring missing values causes biased classifiers. Imputation introduces its own uncertainty.'))

story.append(sub_header('Issue 3: Irrelevant and Redundant Features'))
story.append(bullet(f'{b("Irrelevant Feature")}: A feature that provides no information about the class label. Example: Patient\'s shoe size is irrelevant for diagnosing flu.'))
story.append(bullet(f'{b("Redundant Feature")}: A feature that is correlated with another feature — it carries the same information twice. Example: "Height in cm" and "Height in inches" are redundant.'))
story.append(bullet(f'{b("Feature Selection")}: The process of identifying and removing irrelevant and redundant features. Benefits: reduces dimensionality, speeds up training, removes noise, prevents overfitting, often improves accuracy.'))
story.append(bullet(f'{b("Methods")}: Filter methods (statistical tests before learning), Wrapper methods (evaluate feature subsets by training a classifier), Embedded methods (feature selection built into the learning algorithm, e.g., decision trees).'))

story.append(sub_header('Issue 4: Overfitting'))
story.append(concept_box(
    'Overfitting',
    'A classifier overfits when it performs very well on training data but poorly on unseen test data. It has memorised the training noise instead of learning the true underlying pattern. Formally: there exists another classifier h\' with HIGHER training error but LOWER test error — meaning h\' generalises better.',
    'Overfitting occurs when the model is too complex relative to the amount of training data. Example: A decision tree with one leaf per training instance perfectly classifies training data (100% accuracy) but fails on test data.'
))
story.append(bullet(f'{b("Detection")}: Large gap between training accuracy and test accuracy. High training accuracy + low test accuracy = overfitting.'))
story.append(bullet(f'{b("Solutions")}: (1) Stop early — do not train until perfect fit. (2) Pruning — for decision trees, remove branches that do not improve generalisation. (3) Regularisation — penalise complexity in the model. (4) More training data — reduces overfitting.'))

story.append(sub_header('Issue 5: Bias-Variance Tradeoff'))
story.append(formula_box('Bias-Variance Decomposition', [
    'Total Error = Bias^2 + Variance + Irreducible Noise',
    '',
    'Bias   = How far off the average prediction is from the true value.',
    '         HIGH BIAS -> model too simple -> Underfitting',
    '         Example: Linear model trying to fit curved data.',
    '',
    'Variance = How much predictions vary across different training sets.',
    '           HIGH VARIANCE -> model too complex -> Overfitting',
    '           Example: Deep decision tree that memorises training data.',
    '',
    'Irreducible Noise = Inherent randomness in data. Cannot be eliminated.',
    '',
    'Bagging REDUCES VARIANCE (averages multiple models).',
    'Boosting REDUCES BOTH Bias and Variance (sequentially corrects errors).',
]))
story.append(sp(4))
story.append(diagram_box('Overfitting Curve (Bias-Variance)',
    'Draw a graph with X-axis = "Model Complexity" (simple to complex) and Y-axis = "Error." Draw two curves: (1) TRAINING ERROR — starts high, continuously decreases to near zero as complexity increases. (2) TEST ERROR — starts high (underfitting region), decreases to a minimum point, then INCREASES again (overfitting region). Mark three regions: LEFT = "Underfitting (High Bias)," MIDDLE (at minimum test error) = "Optimal Complexity," RIGHT = "Overfitting (High Variance)." Label the minimum test error point as the ideal model complexity to choose.'))

story.append(sp(4))
story.append(example_box(
    'Overfitting in Decision Trees',
    'A decision tree trained on 100 patient records grows until it has 100 leaves — one per training instance. Training accuracy = 100% (perfectly memorised). Test accuracy on 50 new patients = 52% (barely better than random guessing).',
    'Solution: Post-prune the tree. Remove branches where removing them does not significantly reduce accuracy on a validation set. Final pruned tree has 8 leaves and achieves 87% test accuracy.'
))
story.append(sp(4))
story.append(inference_box(
    'The central challenge of supervised learning is not fitting the training data — any sufficiently complex model can do that. The challenge is learning a model that captures the TRUE underlying pattern and generalises to unseen data. Understanding and managing the Bias-Variance tradeoff is the most important skill for any machine learning practitioner.'
))
story.append(exam_tip_box(
    'Three critical data issues: Noise (incorrect values), Missing Values (absent values), Irrelevant Features (useless attributes). Two model issues: Underfitting (too simple, high bias) and Overfitting (too complex, high variance). Bias-Variance = Total Error formula is frequently asked. Bagging reduces Variance; Boosting reduces Bias+Variance.'
))
story.append(PageBreak())

# ══════════════════════════════════════════════════════════════
# TOPIC 3 — DECISION TREES
# ══════════════════════════════════════════════════════════════
story += section_header('3.  Decision Trees')

story.append(concept_box(
    'Decision Tree',
    'A Decision Tree classifies instances by sorting them down a tree based on feature values. Each INTERNAL NODE tests a feature. Each BRANCH represents a feature value outcome. Each LEAF NODE assigns a class label. An instance is classified by starting at the root and following the matching branches until a leaf is reached.',
    'Decision trees are the most comprehensible classification method — they can be directly converted to human-readable IF-THEN rules, making them widely used in medical and legal decision support systems.'
))
story.append(sp(6))

story.append(p(f'{b("DETAILED EXPLANATION")}', sub_h))
story.append(sub_header('Tree Construction — Information Gain'))
story.append(p('The key decision in building a decision tree is: which feature should split the data at each node? The answer is the feature that provides the most INFORMATION GAIN — the greatest reduction in uncertainty (entropy) about the class.', body))
story.append(formula_box('Entropy and Information Gain', [
    'Entropy H(S) = measure of impurity/uncertainty in a dataset S:',
    '  H(S) = - Sum [ p_i x log2(p_i) ]    for each class i',
    '  p_i = proportion of class i in S',
    '  H=0: perfectly pure (all same class). H=1: maximum impurity (50-50 split).',
    '',
    'Information Gain IG(S, A) for feature A:',
    '  IG(S, A) = H(S) - Sum [ (|Sv|/|S|) x H(Sv) ]',
    '  Sv = subset of S where feature A = value v',
    '  |Sv|/|S| = proportion of instances with value v',
    '',
    'Choose the feature with HIGHEST Information Gain as the split node.',
    '',
    'Example: S = {9 Yes, 5 No} (14 instances)',
    '  H(S) = -(9/14)log2(9/14) - (5/14)log2(5/14) = 0.940 bits',
    'After splitting on Outlook:',
    '  Sunny (5 instances): H = 0.971',
    '  Overcast (4 instances): H = 0.0 (all Yes)',
    '  Rain (5 instances): H = 0.971',
    '  IG(Outlook) = 0.940 - [(5/14)x0.971 + (4/14)x0 + (5/14)x0.971] = 0.246 bits',
]))

story.append(sub_header('C4.5 Algorithm'))
story.append(bullet(f'{b("C4.5")}: The most widely used decision tree algorithm, developed by Ross Quinlan. Improvements over ID3: uses Gain Ratio instead of Information Gain (avoids bias toward features with many values), handles continuous features by finding optimal split thresholds, handles missing values during both training and classification.'))
story.append(bullet(f'{b("Gain Ratio")}: IG(S,A) / SplitInfo(A). SplitInfo penalises features with many distinct values (like ID numbers) which would have artificially high Information Gain.'))
story.append(bullet(f'{b("Stopping criteria")}: Stop splitting when all instances belong to one class (pure node), or no more features to split on, or too few instances in a node (minimum leaf size threshold).'))

story.append(sub_header('Pruning — Preventing Overfitting'))
story.append(bullet(f'{b("Pre-Pruning (Early Stopping)")}: Stop growing the tree before it perfectly fits training data. Stop when: information gain of best split falls below threshold, node has fewer than minimum required instances, or statistical test shows split is not significant.'))
story.append(bullet(f'{b("Post-Pruning (Bottom-Up Pruning)")}: Grow the full tree first. Then work bottom-up — at each internal node, evaluate whether replacing its subtree with a leaf improves accuracy on a validation set. If yes, prune. Produces smaller, more generalisable trees.'))
story.append(bullet(f'{b("Occam's Razor principle")}: Among models with similar accuracy, prefer the simpler one. A tree with 5 leaves that achieves 87% accuracy is preferred over a tree with 50 leaves that achieves 88% accuracy.'))

story.append(sp(4))
story.append(diagram_box('Decision Tree — Play Tennis Example',
    'Draw a tree: ROOT = "Outlook" (top). Three branches: "Sunny" (left) → child node "Humidity" → Humidity=High leaf "NO," Humidity=Normal leaf "YES." "Overcast" (middle) → leaf "YES" (all overcast = play). "Rain" (right) → child node "Wind" → Wind=Strong leaf "NO," Wind=Weak leaf "YES." Label each node type: rounded rectangle = decision node, oval = leaf/class. Shade leaf nodes: YES in green, NO in red.'))

story.append(sp(4))
story.append(example_box(
    'Decision Tree — Play Tennis Worked Trace',
    'New instance: Outlook=Sunny, Humidity=High, Wind=Strong. Classify:',
    'Step 1: Root = Outlook. Instance has Outlook=Sunny → follow Sunny branch.',
    'Step 2: Node = Humidity. Instance has Humidity=High → follow High branch.',
    'Step 3: Reach leaf: Class = NO (Do not play tennis).',
    'The tree correctly predicts this player should NOT play given sunny weather with high humidity.',
))
story.append(sp(4))
story.append(inference_box(
    'Decision trees are unique among ML classifiers in that they are BOTH accurate AND interpretable. A doctor can look at a decision tree and immediately understand why a patient is classified as "high risk" — it traced a specific path through clinically meaningful features. This interpretability makes decision trees the preferred choice in medical, legal, and regulatory contexts where decisions must be explainable. Their main weakness — overfitting — is effectively managed through pruning.'
))
story.append(sp(4))
story.append(pros_cons_box(
    ['Highly interpretable — easily converted to IF-THEN rules',
     'No need to normalise or scale features',
     'Handles both continuous and categorical features natively',
     'Handles missing values (in C4.5)',
     'Fast classification — just follow branches from root to leaf',
     'Can model non-linear decision boundaries',
     'Built-in feature selection via Information Gain'],
    ['Prone to overfitting without pruning',
     'Unstable — small changes in data can produce completely different trees',
     'Biased toward features with many values (ID3)',
     'Struggles with features that interact in complex ways',
     'Cannot model smooth decision boundaries well',
     'Building optimal binary trees is NP-complete',
     'Less accurate than ensemble methods (Random Forest, Boosting)']
))
story.append(sp(4))
story.append(exam_tip_box(
    'Decision Tree formula to memorise: H(S) = -Sum[p_i x log2(p_i)]. Information Gain IG(S,A) = H(S) - weighted sum of child entropies. Highest IG feature becomes the root. C4.5 uses Gain Ratio to avoid bias. Pre-pruning = stop early; Post-pruning = prune after full growth. Decision trees are comprehensible but prone to overfitting.'
))
story.append(PageBreak())

# ══════════════════════════════════════════════════════════════
# TOPIC 4 — NEURAL NETWORKS
# ══════════════════════════════════════════════════════════════
story += section_header('4.  Neural Networks (Perceptron-Based)')

story.append(concept_box(
    'Artificial Neural Networks (ANN)',
    'An Artificial Neural Network is a computational model loosely inspired by biological neural networks (the brain). It consists of layers of interconnected processing units (neurons/nodes). Each connection has a weight that is adjusted during training. The network learns by propagating errors backwards and updating weights to reduce the difference between predicted and actual outputs.',
    'ANNs excel at learning complex, non-linear patterns in data. They are the foundation of modern deep learning systems (image recognition, natural language processing, speech recognition).'
))
story.append(sp(6))

story.append(p(f'{b("DETAILED EXPLANATION")}', sub_h))
story.append(sub_header('4.1 The Perceptron — Building Block'))
story.append(bullet(f'{b("Perceptron")}: The simplest neural unit. Takes n inputs (x1, x2,...,xn) with corresponding weights (w1, w2,...,wn). Computes a weighted sum and applies an activation function to produce output.'))
story.append(formula_box('Perceptron Computation', [
    'Weighted Sum:  z = w0 + w1*x1 + w2*x2 + ... + wn*xn',
    '               z = w0 + Sum(wi * xi)   where w0 = bias term',
    '',
    'Activation Function (Step):',
    '  output = 1  if z >= threshold',
    '  output = 0  if z < threshold',
    '',
    'Sigmoid Activation (for smooth gradient):',
    '  sigma(z) = 1 / (1 + e^(-z))   range: (0, 1)',
    '',
    'Limitation: Perceptron can only classify LINEARLY SEPARABLE data.',
    '  XOR problem is NOT linearly separable -> perceptron fails.',
    '  Solution: Use multi-layer ANN with hidden layers.',
]))

story.append(sub_header('4.2 Multi-Layer ANN Architecture'))
story.append(bullet(f'{b("Input Layer")}: Receives raw feature values. One neuron per input feature. No computation — just passes values forward.'))
story.append(bullet(f'{b("Hidden Layer(s)")}: One or more intermediate layers. Each neuron computes a weighted sum of inputs from the previous layer and applies an activation function. Hidden layers allow the network to learn non-linear patterns. The number of hidden neurons is a hyperparameter — too few: underfitting; too many: overfitting.'))
story.append(bullet(f'{b("Output Layer")}: Produces the final prediction. For binary classification: one output neuron (probability of class 1). For multi-class: one neuron per class (softmax activation gives probabilities summing to 1).'))
story.append(bullet(f'{b("Weights")}: Each connection has a weight w_ij (weight from neuron i to neuron j). These are the parameters learned during training. Initialised randomly at the start.'))

story.append(sub_header('4.3 Backpropagation Training Algorithm'))
story.append(p('Backpropagation is the algorithm used to train multi-layer ANNs. It works in two passes:', body))
story.append(bullet(f'{b("Forward Pass")}: Input features are fed in. Values propagate layer by layer to produce an output prediction.'))
story.append(bullet(f'{b("Compute Error")}: Compare prediction to the true label. Compute the loss (error).'))
story.append(bullet(f'{b("Backward Pass")}: Propagate the error backwards through the network. Compute how much each weight contributed to the error (gradient).'))
story.append(bullet(f'{b("Weight Update")}: Adjust each weight in the direction that reduces the error (gradient descent).'))

story.append(formula_box('Backpropagation Weight Update', [
    'Loss Function (Mean Squared Error):',
    '  E = 0.5 x Sum(target_j - output_j)^2    over all output neurons j',
    '',
    'Weight update rule:',
    '  delta_w_ij = eta x delta_j x output_i',
    '',
    'Where:',
    '  eta      = learning rate (e.g., 0.01 to 0.1) - controls step size',
    '  delta_j  = error signal at neuron j',
    '  output_i = output of neuron i (sending neuron)',
    '',
    'Error signal for OUTPUT layer neuron j:',
    '  delta_j = output_j x (1 - output_j) x (target_j - output_j)',
    '',
    'Error signal for HIDDEN layer neuron j:',
    '  delta_j = output_j x (1 - output_j) x Sum(delta_k x w_jk)',
    '           (where sum is over neurons k in the next layer)',
    '',
    'New weight:  w_ij(new) = w_ij(old) + delta_w_ij',
    '  Repeat for all weights, for all training instances, for many epochs.',
]))

story.append(sp(4))
story.append(diagram_box('Feedforward ANN with 3 Layers',
    'Draw three vertical columns of circles: LEFT = Input Layer (3 nodes: x1, x2, x3). MIDDLE = Hidden Layer (4 nodes: h1, h2, h3, h4). RIGHT = Output Layer (2 nodes: Class A, Class B). Draw arrows from EVERY input node to EVERY hidden node (3x4 = 12 arrows). Draw arrows from EVERY hidden node to EVERY output node (4x2 = 8 arrows). Label one arrow with "w_ij" (weight). Label each layer. Below the diagram: "FORWARD PASS →" and "← BACKWARD PASS (Backpropagation)" arrows showing the two-direction information flow.'))

story.append(sp(4))
story.append(example_box(
    'ANN — One Forward Pass (Simplified)',
    'Input: x1=1, x2=0. Weights: w11=0.5, w21=0.3 (to hidden neuron h1). Bias w0=0.1.',
    'Hidden neuron h1: z = 0.1 + (0.5 x 1) + (0.3 x 0) = 0.6. Output = sigmoid(0.6) = 0.646.',
    'If target output = 1.0: Error at output = 0.5 x (1.0 - 0.646)^2 = 0.063.',
    'Backpropagation computes delta for output neuron and propagates back to update w11, w21 etc. After thousands of such updates across all training instances, the network learns the correct weights.',
))
story.append(sp(4))
story.append(inference_box(
    'Neural networks are the most powerful general-purpose classifiers but at the cost of interpretability. A trained ANN is a "black box" — you can see the weights but cannot easily explain WHY a specific input was classified as it was. This makes them unsuitable for high-stakes applications (medical diagnosis, legal decisions) where explanations are legally required. Despite this, their accuracy on complex, high-dimensional data (images, speech, text) is unmatched.'
))
story.append(sp(4))
story.append(pros_cons_box(
    ['Handles non-linear, highly complex decision boundaries',
     'Learns features automatically — no manual feature engineering needed',
     'Scales well with data — more data = better performance',
     'Universally applicable — works for classification, regression, generation',
     'Foundation of deep learning and modern AI (GPT, image recognition)',
     'Can learn from raw inputs (pixels, audio waveforms)'],
    ['Black box — very poor interpretability',
     'Slow training — requires many epochs and large datasets',
     'Sensitive to hyperparameters (learning rate, architecture, activation)',
     'Risk of local minima (though less severe with modern optimisers)',
     'Prone to overfitting on small datasets',
     'Requires careful normalisation of input features',
     'Cannot easily incorporate prior domain knowledge']
))
story.append(sp(4))
story.append(exam_tip_box(
    'ANN formula to memorise: delta_w_ij = eta x delta_j x output_i. For OUTPUT neuron: delta_j = output_j(1-output_j)(target_j - output_j). For HIDDEN neuron: delta_j = output_j(1-output_j) Sum(delta_k x w_jk). Sigmoid = 1/(1+e^(-z)). Three layers: Input, Hidden, Output. Backprop = forward pass to compute output, then backward pass to update weights.'
))
story.append(PageBreak())

# ══════════════════════════════════════════════════════════════
# TOPIC 5 — NAIVE BAYES
# ══════════════════════════════════════════════════════════════
story += section_header('5.  Naive Bayes Classifier')

story.append(concept_box(
    'Naive Bayes Classifier',
    'Naive Bayes is a probabilistic classifier based on Bayes\' Theorem with the "naive" assumption that ALL features are conditionally independent given the class label. Despite this simplifying assumption (which is almost never true in practice), Naive Bayes works remarkably well across many domains, especially text classification.',
    'It is called "naive" because assuming all features are independent is an oversimplification — in reality, features often correlate. Yet this simplification makes the classifier extremely fast and often surprisingly competitive with more complex methods.'
))
story.append(sp(6))

story.append(p(f'{b("DETAILED EXPLANATION")}', sub_h))
story.append(formula_box('Bayes Theorem and Naive Bayes', [
    'Bayes Theorem:',
    '  P(Class | Features) = P(Class) x P(Features | Class) / P(Features)',
    '',
    'Naive Bayes assumes CONDITIONAL INDEPENDENCE:',
    '  P(x1, x2,...,xn | Class) = P(x1|Class) x P(x2|Class) x ... x P(xn|Class)',
    '',
    'Classification rule (choose class C that maximises):',
    '  C* = argmax_C [ P(C) x Product of P(xi | C) for i=1 to n ]',
    '',
    'P(C) = Prior probability of class C = (count of C in training) / (total instances)',
    'P(xi | C) = Conditional probability of feature value xi given class C',
    '           = (count of instances with xi AND C) / (count of class C)',
    '',
    'Laplace Smoothing (avoids zero probabilities):',
    '  P(xi | C) = (count(xi, C) + 1) / (count(C) + number of feature values)',
]))

story.append(sp(4))
story.append(example_box(
    'Naive Bayes — Worked Numerical Example',
    'Training data: 14 days of tennis weather. Class: Play (Yes=9, No=5).',
    'P(Yes) = 9/14 = 0.643.  P(No) = 5/14 = 0.357.',
    'Feature probabilities (from training data):',
    '  P(Outlook=Sunny | Yes) = 2/9.  P(Outlook=Sunny | No) = 3/5.',
    '  P(Humidity=High | Yes) = 3/9.  P(Humidity=High | No) = 4/5.',
    '  P(Wind=Weak | Yes) = 6/9.     P(Wind=Weak | No) = 2/5.',
    'New instance: Outlook=Sunny, Humidity=High, Wind=Weak.',
    'P(Yes | instance) proportional to: 0.643 x (2/9) x (3/9) x (6/9) = 0.643 x 0.222 x 0.333 x 0.667 = 0.0318',
    'P(No  | instance) proportional to: 0.357 x (3/5) x (4/5) x (2/5) = 0.357 x 0.600 x 0.800 x 0.400 = 0.0686',
    'Since P(No) > P(Yes): Classify as NO — Do not play tennis.',
    'Verify by normalising: P(No|instance) = 0.0686/(0.0318+0.0686) = 68.3%.',
))
story.append(sp(4))
story.append(inference_box(
    'Naive Bayes achieves its speed and simplicity by reducing the joint probability P(x1,...,xn|C) — which would require exponential space to represent exactly — to a product of simple marginal probabilities that can be computed in a single pass over the training data. This makes it ideal for large-scale text classification (spam filtering, sentiment analysis) where feature independence is approximately true and speed matters more than maximum accuracy.'
))
story.append(sp(4))
story.append(pros_cons_box(
    ['Extremely fast training — single pass through data',
     'Very fast classification — simple multiplication',
     'Low storage — only class and feature probabilities stored',
     'Handles high-dimensional data well (text classification)',
     'Works with small training sets (probabilistic estimates)',
     'Naturally handles missing values (just skip that term)',
     'Provides probability estimates, not just class labels'],
    ['Naive independence assumption is almost always violated in reality',
     'Zero frequency problem — if feature never appears with a class, probability = 0 (fixed by Laplace smoothing)',
     'Cannot capture interactions between features',
     'Poor at modelling continuous data without discretisation or Gaussian assumption',
     'Outperformed by SVMs and Neural Networks on complex tasks',
     'Sensitive to the quality of probability estimates with small datasets']
))
story.append(exam_tip_box(
    'Naive Bayes formula: C* = argmax P(C) x Product[P(xi|C)]. Remember: it is "naive" because of the independence assumption. P(C) = class frequency. P(xi|C) = conditional frequency. Laplace smoothing adds 1 to numerator and k (number of values) to denominator to avoid zero probabilities. Works very well for spam detection and text classification.'
))
story.append(PageBreak())

# ══════════════════════════════════════════════════════════════
# TOPIC 6 — KNN
# ══════════════════════════════════════════════════════════════
story += section_header('6.  k-Nearest Neighbour (kNN)')

story.append(concept_box(
    'k-Nearest Neighbour (kNN)',
    'kNN is a LAZY LEARNING algorithm — it does not build an explicit model during training. Instead, it stores all training instances. At classification time, for a new instance, it finds the k most similar training instances (nearest neighbours) using a distance metric, and assigns the majority class among those k neighbours.',
    '"Lazy" means no work is done at training time — all computation is deferred to classification time. This contrasts with "eager" learners like Decision Trees and Neural Networks that build a model upfront.'
))
story.append(sp(6))

story.append(p(f'{b("DETAILED EXPLANATION")}', sub_h))
story.append(formula_box('Euclidean Distance (kNN)', [
    'Distance between instance A = (a1, a2,...,an) and B = (b1, b2,...,bn):',
    '',
    '  d(A, B) = sqrt( (a1-b1)^2 + (a2-b2)^2 + ... + (an-bn)^2 )',
    '           = sqrt( Sum(ai - bi)^2 )   for i = 1 to n',
    '',
    'Other distance metrics:',
    '  Manhattan distance: d = Sum(|ai - bi|)',
    '  Minkowski distance: d = (Sum(|ai-bi|^p))^(1/p)',
    '                        p=1: Manhattan, p=2: Euclidean',
    '',
    'Classification rule:',
    '  Find k instances from training set with smallest d(new, training_i)',
    '  Assign class = majority class among those k neighbours',
    '  Ties broken randomly or by distance weighting.',
    '',
    'Choosing k:',
    '  Small k (k=1): sensitive to noise, overfitting risk.',
    '  Large k: smoother boundary, risk of including irrelevant instances.',
    '  Best k found by cross-validation. Use odd k for binary classification.',
]))

story.append(sp(4))
story.append(example_box(
    'kNN Worked Example (k=3)',
    'Training data (2D features: Height, Weight, Class):',
    '  P1: (170cm, 65kg) → Athletic.  P2: (160cm, 85kg) → Overweight.',
    '  P3: (175cm, 70kg) → Athletic.  P4: (155cm, 90kg) → Overweight.',
    '  P5: (180cm, 80kg) → Athletic.',
    'New instance Q: (165cm, 72kg). Classify with k=3.',
    'Compute distances from Q to all training instances:',
    '  d(Q, P1) = sqrt((165-170)^2 + (72-65)^2) = sqrt(25+49) = sqrt(74) = 8.60',
    '  d(Q, P2) = sqrt((165-160)^2 + (72-85)^2) = sqrt(25+169) = sqrt(194) = 13.93',
    '  d(Q, P3) = sqrt((165-175)^2 + (72-70)^2) = sqrt(100+4) = sqrt(104) = 10.20',
    '  d(Q, P4) = sqrt((165-155)^2 + (72-90)^2) = sqrt(100+324) = sqrt(424) = 20.59',
    '  d(Q, P5) = sqrt((165-180)^2 + (72-80)^2) = sqrt(225+64) = sqrt(289) = 17.00',
    'Sort by distance: P1(8.60), P3(10.20), P2(13.93), P5(17.00), P4(20.59)',
    '3 Nearest Neighbours: P1 (Athletic), P3 (Athletic), P2 (Overweight)',
    'Majority vote: Athletic=2, Overweight=1 → Classify Q as ATHLETIC.',
))
story.append(sp(4))
story.append(diagram_box('kNN — 2D Scatter Plot',
    'Draw a 2D scatter plot with X-axis = "Height" and Y-axis = "Weight." Plot 5 training points: P1, P3, P5 as circles (Athletic class, green). P2, P4 as triangles (Overweight class, red). Plot new point Q as a star. Draw a circle around Q that encloses the 3 nearest neighbours (P1, P3, P2). Label the circle "k=3 neighbourhood." Show majority vote: 2 circles vs 1 triangle → Q gets circle (Athletic).'))

story.append(sp(4))
story.append(cmp_table(
    ['Property', 'Lazy Learning (kNN)', 'Eager Learning (Decision Tree, ANN, SVM)'],
    [
        ['Training Time', 'Zero — just store data', 'Significant — builds model from data'],
        ['Classification Time', 'Slow — compute distance to ALL instances', 'Fast — just apply the stored model'],
        ['Memory', 'High — must store entire training set', 'Low — only model parameters stored'],
        ['Adaptation', 'Automatic — new instances just added', 'Must retrain to incorporate new data'],
        ['Interpretability', 'Low (just distances)', 'Variable (DT=high, ANN=low)'],
    ]
))
story.append(sp(4))
story.append(inference_box(
    'kNN\'s "zero training time" is deceptive — all the computational cost is deferred to classification time. For large datasets with millions of training instances, classifying a single new point requires computing millions of distances. This makes kNN impractical for high-throughput applications without specialised data structures (KD-trees, ball trees) to speed up nearest-neighbour search.'
))
story.append(sp(4))
story.append(pros_cons_box(
    ['Zero training time — just store the data',
     'Naturally handles multi-class classification',
     'Non-parametric — makes no assumptions about data distribution',
     'Simple to understand and implement',
     'Automatically adapts to new training data (just add instances)',
     'Works well when training data is large and representative'],
    ['Very slow at classification — O(n) distance computations per query',
     'High memory — entire training set must be stored',
     'Sensitive to irrelevant features (they pollute distances equally)',
     'Sensitive to feature scale — must normalise features first',
     'Performance degrades in high dimensions (curse of dimensionality)',
     'No model is built — no insights into data structure',
     'Choice of k and distance metric requires careful tuning']
))
story.append(exam_tip_box(
    'kNN is LAZY (no training, slow classification). Distance formula: d(A,B) = sqrt(Sum(ai-bi)^2). Choose k by cross-validation — odd k avoids ties. Normalise features before applying kNN (otherwise large-scale features dominate distance). Majority vote among k neighbours gives the class. Curse of dimensionality: kNN degrades with too many features.'
))
story.append(PageBreak())

# ══════════════════════════════════════════════════════════════
# TOPIC 7 — SVM
# ══════════════════════════════════════════════════════════════
story += section_header('7.  Support Vector Machines (SVM)')

story.append(concept_box(
    'Support Vector Machine (SVM)',
    'SVM finds the OPTIMAL HYPERPLANE that separates two classes with the MAXIMUM MARGIN. The margin is the distance between the hyperplane and the nearest data points (called Support Vectors) from each class. Maximising the margin minimises the generalisation error — it gives the classifier the largest possible "safety buffer" between classes.',
    'SVMs are grounded in Statistical Learning Theory (Vapnik-Chervonenkis theory). Unlike neural networks, SVMs have a convex optimisation problem with a guaranteed global minimum — no local minima issues.'
))
story.append(sp(6))

story.append(p(f'{b("DETAILED EXPLANATION")}', sub_h))
story.append(sub_header('Core Concepts'))
story.append(bullet(f'{b("Hyperplane")}: A decision boundary in n-dimensional space. In 2D, it is a line. In 3D, a plane. In n-D, a hyperplane. The equation of the hyperplane: w·x + b = 0, where w = weight vector (normal to hyperplane) and b = bias.'))
story.append(bullet(f'{b("Support Vectors")}: The training instances closest to the hyperplane — on the margin boundary. These are the ONLY instances that matter for defining the optimal hyperplane. All other training instances can be removed and the hyperplane would be unchanged.'))
story.append(bullet(f'{b("Margin")}: The distance between the two margin boundaries (one for each class). Margin = 2/||w||. Maximising the margin is equivalent to minimising ||w||.'))
story.append(bullet(f'{b("Hard Margin SVM")}: No misclassifications allowed. Works only for perfectly linearly separable data.'))
story.append(bullet(f'{b("Soft Margin SVM")}: Allows some misclassifications (controlled by parameter C). C controls the tradeoff: large C = small margin, few errors; small C = large margin, more errors allowed.'))

story.append(formula_box('SVM Optimisation Problem', [
    'Optimal hyperplane: w.x + b = 0',
    'Margin boundaries: w.x + b = +1 (class +1 side)',
    '                   w.x + b = -1 (class -1 side)',
    '',
    'Objective: Maximise Margin = 2 / ||w||',
    '           Equivalently: MINIMISE (1/2)||w||^2',
    '',
    'Subject to constraints (hard margin):',
    '  yi(w.xi + b) >= 1   for all training instances i',
    '  yi = +1 or -1 (class labels)',
    '',
    'Soft Margin SVM (allows errors with slack variables xi_i):',
    '  Minimise: (1/2)||w||^2 + C x Sum(xi_i)',
    '  Subject to: yi(w.xi + b) >= 1 - xi_i,  xi_i >= 0',
    '  C = regularisation parameter. Large C = penalise errors heavily.',
    '',
    'Kernel Trick for Non-linear Data:',
    '  Replace x.z with kernel K(x, z) = phi(x).phi(z)',
    '  Linear kernel:      K(x,z) = x.z',
    '  Polynomial kernel:  K(x,z) = (x.z + 1)^d',
    '  RBF/Gaussian kernel: K(x,z) = exp(-||x-z||^2 / 2*sigma^2)',
    '  RBF most commonly used — maps data to infinite-dimensional space.',
]))

story.append(sp(4))
story.append(diagram_box('SVM Maximum Margin Hyperplane',
    'Draw a 2D scatter plot. Left group: 4 circles (class +1). Right group: 4 crosses (class -1). Draw THREE parallel lines: MIDDLE bold line = "Optimal Hyperplane (w.x+b=0)." LEFT dashed line = "Margin boundary (+1): w.x+b=+1." RIGHT dashed line = "Margin boundary (-1): w.x+b=-1." Circle the data points that lie exactly ON the dashed lines — label them "Support Vectors." Draw a double-headed arrow between the dashed lines = "Margin = 2/||w||." Note: Non-support vectors do NOT affect the hyperplane position.'))

story.append(sp(4))
story.append(example_box(
    'SVM Kernel Trick — Non-linear Data',
    'Problem: Data in 2D (x1, x2) where class boundary is circular (XOR-like). No linear hyperplane can separate them.',
    'Kernel trick: Map data to 3D using phi(x1, x2) = (x1^2, sqrt(2)*x1*x2, x2^2).',
    'In the 3D feature space, the data IS linearly separable — a flat hyperplane works.',
    'Key insight: We NEVER explicitly compute phi(x). The kernel function K(x,z) = (x.z)^2 computes phi(x).phi(z) directly in the ORIGINAL space — much cheaper.',
    'This is the "kernel trick" — get the benefits of high-dimensional feature spaces without the computational cost.',
))
story.append(sp(4))
story.append(inference_box(
    'SVMs are unique among classifiers in that their solution depends only on the SUPPORT VECTORS — a small subset of training instances. This means SVMs are efficient even with high-dimensional data (where the number of features exceeds the number of training instances), which is common in text and genomic data. The kernel trick makes SVMs extremely powerful for non-linear problems without explicitly transforming to high-dimensional spaces.'
))
story.append(sp(4))
story.append(pros_cons_box(
    ['Global optimum guaranteed (convex optimisation, no local minima)',
     'Works well in high-dimensional spaces (text, genomics)',
     'Effective when features >> instances (unlike neural networks)',
     'Only support vectors matter — memory efficient after training',
     'Kernel trick handles non-linear boundaries elegantly',
     'Strong theoretical foundation (VC dimension theory)',
     'Robust to outliers (soft margin)'],
    ['Slow training for large datasets (O(n^2) to O(n^3))',
     'Binary classifier — must extend for multi-class (one-vs-one, one-vs-all)',
     'Kernel selection requires domain expertise and cross-validation',
     'Poor interpretability — black box (like ANN)',
     'Sensitive to feature scaling — must normalise data',
     'C and kernel parameters must be carefully tuned',
     'Does not directly provide probability estimates (needs calibration)']
))
story.append(exam_tip_box(
    'SVM key formulas: Margin = 2/||w||. Objective: minimise (1/2)||w||^2. Constraint: yi(w.xi+b) >= 1. Support Vectors are the ONLY points on the margin boundaries. Kernel trick replaces dot products with K(x,z). RBF kernel: K(x,z) = exp(-||x-z||^2 / 2*sigma^2). Soft margin adds parameter C to control the misclassification penalty.'
))
story.append(PageBreak())

# ══════════════════════════════════════════════════════════════
# TOPIC 8 — COMBINING CLASSIFIERS
# ══════════════════════════════════════════════════════════════
story += section_header('8.  Combining Classifiers (Ensemble Methods)')

story.append(concept_box(
    'Ensemble Methods',
    'Ensemble methods combine multiple classifiers to produce a final prediction that is more accurate than any individual classifier. The intuition: just as a committee of experts makes better decisions than a single expert, a committee of classifiers can outperform any individual one. Three main approaches: Bagging, Boosting, and Stacking.',
    'Ensembles work best when the component classifiers are both ACCURATE (individually above chance) and DIVERSE (making different errors). Combining classifiers that all make the same mistakes provides no benefit.'
))
story.append(sp(6))

story.append(p(f'{b("DETAILED EXPLANATION")}', sub_h))
story.append(sub_header('8.1 Bagging (Bootstrap Aggregating)'))
story.append(bullet(f'{b("Core Idea")}: Train multiple classifiers on different random BOOTSTRAP SAMPLES of the training data (sampling with replacement). Each sample is the same size as the original but some instances appear multiple times and others are omitted. Each sample trains one classifier. Final prediction = majority vote of all classifiers.'))
story.append(bullet(f'{b("Bootstrap Sample")}: Draw n instances randomly WITH REPLACEMENT from n training instances. On average, each sample contains ~63.2% of original instances (some repeated). About 36.8% of instances are omitted (these form the "out-of-bag" set for error estimation).'))
story.append(bullet(f'{b("Effect")}: Reduces VARIANCE. Because each classifier is trained on a different sample, they make different errors. Averaging across classifiers reduces the sensitivity to any one sample.'))
story.append(bullet(f'{b("Best for")}: Unstable classifiers that are sensitive to small changes in training data. Decision trees are highly unstable — small data changes produce completely different trees. Bagging of decision trees = Random Forest.'))
story.append(bullet(f'{b("Number of classifiers")}: Most error reduction occurs within the first 10-25 classifiers. 50-100 classifiers is typical.'))

story.append(sub_header('8.2 Boosting (AdaBoost)'))
story.append(bullet(f'{b("Core Idea")}: Train classifiers SEQUENTIALLY. Each new classifier focuses on the instances that the previous classifiers got WRONG. Misclassified instances get higher weights — more likely to appear in the next training sample. Final prediction = WEIGHTED vote (classifiers with higher accuracy get more weight).'))
story.append(bullet(f'{b("AdaBoost Algorithm")}: (1) Assign equal weights to all instances. (2) Train classifier h1. (3) Increase weights of misclassified instances, decrease weights of correctly classified ones. (4) Train h2 on reweighted data. (5) Repeat. (6) Final classifier = weighted combination of all h1,...,hT.'))
story.append(bullet(f'{b("Effect")}: Reduces BOTH Bias and Variance — making it more powerful than bagging but more sensitive to noise (noisy instances get repeatedly upweighted, which can hurt performance).'))
story.append(bullet(f'{b("Random Forest")}: Bagging + random feature subsets at each split. One of the best general-purpose classifiers.'))

story.append(sp(4))
story.append(cmp_table(
    ['Property', 'Bagging', 'Boosting (AdaBoost)'],
    [
        ['Training', 'Parallel — each classifier independent', 'Sequential — each depends on previous'],
        ['Sampling', 'Uniform random bootstrap samples', 'Weighted — focuses on misclassified'],
        ['Voting', 'Equal weight majority vote', 'Weighted vote (accuracy-proportional)'],
        ['Error Reduction', 'Reduces VARIANCE', 'Reduces BIAS and VARIANCE'],
        ['Noise Sensitivity', 'Robust — noise not upweighted', 'Sensitive — noise gets high weight'],
        ['Best For', 'Unstable classifiers (Decision Trees)', 'Weak classifiers (simple stumps)'],
    ]
))

story.append(sub_header('8.3 Stacking'))
story.append(bullet(f'{b("Core Idea")}: Train diverse base classifiers (Level 0). Use their predictions as features to train a meta-learner (Level 1) which produces the final prediction. Unlike bagging/boosting, classifiers can be of different types (e.g., one DT + one ANN + one kNN).'))
story.append(bullet(f'{b("Diversity")}: Stacking benefits from diverse classifiers that make different types of errors. The meta-learner learns how to best combine the base classifiers\' predictions.'))
story.append(sp(4))
story.append(inference_box(
    'Ensemble methods consistently achieve the highest accuracy in machine learning competitions and real-world applications. The reason: no single algorithm is universally best across all datasets. By combining diverse classifiers, ensembles hedge their bets — if one classifier is weak on certain types of instances, others in the ensemble compensate. The tradeoff: interpretability is completely lost, and computational cost multiplies by the number of classifiers.'
))
story.append(sp(4))
story.append(pros_cons_box(
    ['Consistently higher accuracy than any single classifier',
     'Bagging: reduces variance and prevents overfitting',
     'Boosting: reduces both bias and variance',
     'Robust to outliers (bagging)',
     'Random Forest: one of the best general classifiers available',
     'Out-of-bag error estimate (bagging): no separate validation set needed'],
    ['Complete loss of interpretability',
     'Increased computational cost — train/store/run N classifiers',
     'More complex to tune (N classifier architectures + ensemble parameters)',
     'Boosting is sensitive to noisy data and outliers',
     'Marginal improvements beyond 50-100 classifiers',
     'Not suitable when explanation of individual decisions is required']
))
story.append(exam_tip_box(
    'Bagging = Bootstrap sampling + equal vote → reduces VARIANCE. Boosting = Sequential + weighted vote → reduces BIAS + VARIANCE. Boosting more powerful but sensitive to noise. Random Forest = Bagging + random feature subsets. Stacking = diverse classifiers + meta-learner. Key insight: ensembles need accurate AND diverse component classifiers to work well.'
))
story.append(PageBreak())

# ══════════════════════════════════════════════════════════════
# TOPIC 9 — DIMENSION REDUCTION
# ══════════════════════════════════════════════════════════════
story += section_header('9.  Overview of Dimension Reduction Methods')

story.append(concept_box(
    'Dimension Reduction',
    'Dimension reduction is the process of mapping high-dimensional data (n-dimensional feature vectors) to a lower-dimensional representation (d-dimensional, typically d=2 or 3) while preserving the essential structure of the data — distances, relationships, and clusters. When d=2, the result is a 2D scatter plot that humans can visually analyse.',
    'In real-world AI applications, objects are described by many features (images by pixels, patients by lab values, documents by word counts). High-dimensional data is impossible to visualise and difficult to analyse. Dimension reduction reveals hidden structure.'
))
story.append(sp(6))

story.append(p(f'{b("DETAILED EXPLANATION")}', sub_h))
story.append(sub_header('Why Dimension Reduction?'))
story.append(bullet(f'{b("Visualisation")}: Humans can only perceive 2D or 3D data. Reducing 100-dimensional patient data to 2D allows visual identification of clusters (disease subtypes), outliers, and relationships.'))
story.append(bullet(f'{b("Curse of Dimensionality")}: As dimensions increase, the volume of the space grows exponentially — data becomes increasingly sparse. Distance metrics become meaningless in very high dimensions (all points become equidistant). Dimension reduction mitigates this.'))
story.append(bullet(f'{b("Noise Removal")}: Many features are redundant or noisy. Projecting to lower dimensions often removes noise while preserving signal.'))
story.append(bullet(f'{b("Computational Efficiency")}: Fewer dimensions = faster training for downstream classifiers.'))

story.append(sub_header('Key Methods'))
story.append(bullet(f'{b("Principal Component Analysis (PCA)")}: Linear method. Finds the directions (principal components) of maximum variance in the data. Projects data onto the top d components. Optimal for preserving GLOBAL variance structure. Limitation: linear — cannot capture non-linear manifolds.'))
story.append(bullet(f'{b("Multidimensional Scaling (MDS)")}: Starts from a pairwise dissimilarity matrix. Finds low-dimensional coordinates that best preserve these dissimilarities. Metric MDS preserves exact distances; Non-metric MDS preserves only rank order of distances.'))
story.append(bullet(f'{b("Self-Organising Map (SOM)")}: Neural network approach — maps data to a discrete 2D grid. Simultaneously performs clustering and dimension reduction. Preserves topological neighbourhood structure.'))
story.append(bullet(f'{b("Sammon\'s Mapping")}: A variant of MDS with a special cost function that emphasises preserving SMALL distances (local structure) more than large distances.'))
story.append(bullet(f'{b("GTM (Generative Topographic Mapping)")}: A probabilistic reformulation of SOM. Provides a principled probabilistic framework — unlike SOM, can project new data points.'))

story.append(formula_box('MDS Objective Function', [
    'Given: m objects with pairwise dissimilarities dij* (original n-D space)',
    'Find: m points Yi in d-D space with pairwise distances dij',
    '',
    'Metric MDS — minimise STRESS:',
    '  E_MDS = Sum_wij x (dij* - dij)^2   [all pairs i < j]',
    '  wij = weight (often 1/dij*^2 for normalisation)',
    '',
    'Solved by iterative optimisation (gradient descent or SMACOF algorithm).',
]))

story.append(sp(4))
story.append(diagram_box('Dimension Reduction — Before and After',
    'Draw two plots side by side. LEFT: A 3D scatter of dots in a complex cloud labeled "Original 50-dimensional data (shown as 3D)." Label axes x1, x2, x3. RIGHT: A 2D scatter of the same dots, now clearly showing 3 distinct clusters (circles of dots with space between them) labeled "After dimension reduction to 2D." Draw connecting arrows between the two plots. Show that what was invisible in high dimensions becomes clear cluster structure in 2D.'))

story.append(sp(4))
story.append(exam_tip_box(
    'Key dimension reduction methods: PCA (linear, maximum variance), MDS (preserves distances from dissimilarity matrix), SOM (neural network, grid-based), Sammon Mapping (preserves small distances especially). MDS formula: minimise E = Sum_wij(dij* - dij)^2. Dimension reduction is crucial because humans cannot perceive more than 3 dimensions, and algorithms suffer from the curse of dimensionality in high dimensions.'
))
story.append(PageBreak())

# ══════════════════════════════════════════════════════════════
# TOPIC 10 — SOM
# ══════════════════════════════════════════════════════════════
story += section_header('10.  Self-Organising Map (SOM)')

story.append(concept_box(
    'Self-Organising Map (SOM) — Kohonen Map',
    'The SOM (proposed by Teuvo Kohonen, 1982) is an unsupervised neural network that projects high-dimensional data onto a regular low-dimensional grid (usually 2D) while preserving the TOPOLOGICAL STRUCTURE of the input space. Similar inputs map to NEARBY neurons on the grid — the map is topology-preserving.',
    'Unlike PCA (which preserves variance) or MDS (which preserves distances), SOM preserves NEIGHBOURHOOD RELATIONSHIPS — inputs that are close in the original space are mapped to nearby neurons on the grid. It simultaneously performs clustering and visualisation.'
))
story.append(sp(6))

story.append(p(f'{b("DETAILED EXPLANATION")}', sub_h))
story.append(sub_header('10.1 SOM Architecture'))
story.append(bullet(f'{b("Grid Structure")}: A rectangular or hexagonal grid of neurons. Typical sizes: 10x10, 20x20, etc. Each neuron has: (a) a POSITION on the grid (i,j), and (b) a CODEBOOK VECTOR m_ij — an n-dimensional vector in the same space as the input data.'))
story.append(bullet(f'{b("Codebook Vectors")}: Initially set to random values. After training, each codebook vector represents the centre of a cluster of input vectors that map to that neuron. The grid of codebook vectors forms a low-dimensional "map" of the input space.'))
story.append(bullet(f'{b("Best Matching Unit (BMU)")}: For any input vector X, the BMU is the neuron whose codebook vector is most similar to X (smallest Euclidean distance).'))

story.append(sub_header('10.2 SOM Training Algorithm'))
story.append(algo_box('SOM Training Algorithm', [
    'STEP 1: Initialise all codebook vectors m_ij randomly (or with PCA).',
    '',
    'STEP 2: For each training input vector X (repeat for many epochs):',
    '   a) Find BMU (Best Matching Unit) c:',
    '      c = argmin { ||X - m_ij|| }   (neuron with smallest distance to X)',
    '',
    '   b) Update BMU and ALL neurons in its neighbourhood:',
    '      m_ij(t+1) = m_ij(t) + alpha(t) x h(c, ij, t) x [X - m_ij(t)]',
    '',
    '      Where:',
    '        alpha(t) = learning rate at time t  (decreases over time, e.g., 0.5 to 0.01)',
    '        h(c, ij, t) = neighbourhood function:',
    '          h(c, ij, t) = exp( -||r_c - r_ij||^2 / (2 x sigma(t)^2) )',
    '          r_c, r_ij = grid positions of BMU and neuron ij',
    '          sigma(t) = neighbourhood radius (decreases over time)',
    '',
    '   c) Effect: BMU moves TOWARDS X. Nearby neurons move a little towards X.',
    '      Distant neurons barely move. Result: topology preserved.',
    '',
    'STEP 3: Reduce alpha(t) and sigma(t) gradually (cooling schedule).',
    'STEP 4: Repeat until convergence (codebook vectors stabilise).',
]))

story.append(sp(4))
story.append(example_box(
    'SOM — Step-by-Step Worked Trace (Simplified)',
    'Setup: 3x3 SOM grid. Input data: 2D vectors representing colours (R, G values).',
    'Initial codebook vectors: random values for all 9 neurons.',
    'Input X = (0.8, 0.2) [high red, low green — reddish colour]:',
    'Step 1 — Find BMU: Compute distance from X to all 9 codebook vectors. Suppose neuron (2,1) is closest: d=0.12. BMU = (2,1).',
    'Step 2 — Update: alpha=0.5, sigma=1.5. Neighbourhood function h:',
    '  h(BMU, BMU) = exp(0) = 1.0  (BMU itself — full update)',
    '  h(BMU, adjacent) = exp(-1/(2x2.25)) = exp(-0.22) = 0.80  (neighbours — 80% update)',
    '  h(BMU, diagonal) = exp(-2/(2x2.25)) = exp(-0.44) = 0.64  (diagonal — 64% update)',
    '  h(BMU, far) = exp(-4/4.5) = 0.41  (far neurons — 41% update)',
    'Update BMU: m_21(new) = m_21(old) + 0.5 x 1.0 x (X - m_21(old)) [moves halfway to X]',
    'Update adjacent neuron (1,1): m_11(new) = m_11(old) + 0.5 x 0.80 x (X - m_11(old))',
    'After many epochs: neurons in region (2,1) will represent "reddish" inputs. Nearby neurons represent similar hues. The grid becomes a colour map.',
))
story.append(sp(4))
story.append(diagram_box('SOM Grid with BMU and Neighbourhood',
    'Draw a 5x5 grid of squares. Shade one square in the centre dark = "BMU (Best Matching Unit)." Shade the 8 immediately adjacent squares medium grey = "Neighbourhood ring 1 (strong update)." Shade the next ring light grey = "Neighbourhood ring 2 (weak update)." Outer squares = white = "Minimal/no update." Draw an arrow from a data point X (outside the grid) pointing to the BMU. Beside the grid, write the neighbourhood function: h = exp(-||r_c - r_ij||^2 / 2*sigma^2). Show sigma shrinking over time with a decreasing curve.'))

story.append(sp(4))
story.append(inference_box(
    'The SOM\'s key advantage over other dimension reduction methods is its ability to project NEW data points not seen during training — simply compute the BMU for the new point. This makes SOM applicable for online, real-time applications where new data continuously arrives. Its simultaneous clustering and visualisation in a single framework also distinguishes it from PCA and MDS which only do projection.'
))
story.append(sp(4))
story.append(pros_cons_box(
    ['Preserves topological neighbourhood structure of data',
     'Simultaneously performs clustering AND visualisation',
     'Can project new data points not seen during training',
     'Visual output (map) is intuitive and interpretable',
     'Works with any type of input data (no distributional assumptions)',
     'Scales to large datasets with online (incremental) learning',
     'U-matrix visualisation reveals cluster boundaries on the map'],
    ['Grid size (kx x ky) must be specified in advance',
     'Results depend on random initialisation and may vary across runs',
     'Training is slow for large datasets and large grids',
     'No probabilistic framework — cannot assign confidence to mappings',
     'Boundaries between clusters are approximate (grid is discrete)',
     'Cannot handle dynamic changes in cluster number naturally',
     'Neighbourhood function and learning rate schedules require tuning']
))
story.append(exam_tip_box(
    'SOM key formula: m_ij(t+1) = m_ij(t) + alpha(t) x h(c,ij,t) x [X - m_ij(t)]. Neighbourhood function: h = exp(-||r_c - r_ij||^2 / 2*sigma^2). BMU = argmin ||X - m_ij||. Both alpha (learning rate) and sigma (neighbourhood radius) DECREASE over time. SOM preserves topology: similar inputs → nearby neurons. Can project NEW data points (advantage over Sammon Mapping and MDS).'
))
story.append(PageBreak())

# ══════════════════════════════════════════════════════════════
# TOPIC 11 — SAMMON MAPPING
# ══════════════════════════════════════════════════════════════
story += section_header('11.  Sammon\'s Mapping')

story.append(concept_box(
    'Sammon\'s Mapping',
    'Sammon\'s Mapping (1969) is a non-linear dimension reduction technique and a special case of metric MDS. It uses a stress function that NORMALISES errors by the original distance — giving proportionally MORE importance to SMALL distances than large ones. This ensures local structure (nearby neighbours, cluster membership) is preserved more faithfully than global structure.',
    'The key insight: an error of 0.1 in a distance of 0.2 (50% error) is far more damaging than the same error of 0.1 in a distance of 10.0 (1% error). Sammon\'s normalisation term dij* in the denominator enforces this priority.'
))
story.append(sp(6))

story.append(p(f'{b("DETAILED EXPLANATION")}', sub_h))
story.append(formula_box('Sammon\'s Stress Function and Update Rule', [
    'Sammon Stress Function E_S:',
    '  E_S = (1 / Sum(dij*)) x Sum [ (dij* - dij)^2 / dij* ]',
    '                               [all pairs i < j]',
    '',
    'Where:',
    '  dij* = Euclidean distance between Xi and Xj in ORIGINAL n-D space',
    '  dij  = Euclidean distance between Yi and Yj in PROJECTED 2D space',
    '  Sum(dij*) = normalisation constant (sum of all original distances)',
    '  Division by dij*: errors in SMALL distances penalised MORE than large ones',
    '',
    'Minimised by Gradient Descent (Newton\'s method):',
    '  y_ik(t+1) = y_ik(t) - alpha x [dE_S/dy_ik] / [d^2E_S/dy_ik^2]',
    '',
    'Where:',
    '  y_ik = k-th coordinate of projected point Yi in 2D space',
    '  t    = iteration number',
    '  alpha = learning rate / magic factor (typically 0.3-0.4)',
    '',
    'First derivative (gradient):',
    '  dE_S/dy_ik = (-2/C) x Sum_j [ (dij* - dij)/(dij* x dij) x (y_ik - y_jk) ]',
    '  where C = Sum(dij*)',
    '',
    'Stopping criterion: E_S < threshold, or maximum iterations reached.',
]))

story.append(sub_header('Interpretation of the Formula'))
story.append(bullet(f'{b("Numerator (dij* - dij)^2")}: The squared error in distance — how far off the 2D distance is from the original n-D distance. Standard distance preservation criterion.'))
story.append(bullet(f'{b("Denominator dij*")}: Normalises by original distance. When dij* is SMALL (nearby points), a given error is magnified → these pairs are strongly penalised for distance errors. When dij* is LARGE (distant points), the same error is reduced → less important.'))
story.append(bullet(f'{b("Result")}: Points that are close in the original space will be close in 2D. Local neighbourhood structure and cluster membership are faithfully preserved. Far-away points may be distorted but this is acceptable.'))
story.append(bullet(f'{b("Performance")}: Empirical studies consistently find Sammon\'s mapping outperforms PCA and classical MDS for preserving local structure. Analysis comparing methods finds it among the best of its class.'))
story.append(bullet(f'{b("SAMANN")}: A neural network implementation of Sammon\'s mapping (similar to SOM). Trains a neural network to approximate the Sammon mapping function, gaining the ability to project new data points.'))

story.append(sp(4))
story.append(example_box(
    'Sammon Mapping — Conceptual Trace',
    'Initial setup: 5 data points in 10D. Place them randomly in 2D (Y1,...,Y5).',
    'Compute dij* (all pairwise distances in 10D) and dij (all pairwise distances in 2D).',
    'Suppose for points 1 and 2: d12* = 0.3 (very close in 10D), d12 = 2.1 (far in 2D).',
    '  Error contribution: (0.3 - 2.1)^2 / 0.3 = 3.24 / 0.3 = 10.80  [LARGE PENALTY]',
    'For points 3 and 4: d34* = 8.0 (far in 10D), d34 = 7.5 (close in 2D).',
    '  Error contribution: (8.0 - 7.5)^2 / 8.0 = 0.25 / 8.0 = 0.031  [small penalty]',
    'Gradient descent updates Y1, Y2 to move closer in 2D (reducing the large error).',
    'After many iterations, nearby points in 10D become nearby in 2D.',
    'Final E_S value indicates quality: E_S close to 0 = excellent preservation.',
))
story.append(sp(4))
story.append(cmp_table(
    ['Property', 'SOM', 'Sammon Mapping', 'PCA'],
    [
        ['Type', 'Neural network (unsupervised)', 'Iterative optimisation', 'Linear algebraic (SVD)'],
        ['Output space', 'Discrete 2D grid', 'Continuous 2D coordinates', 'Continuous d-D coordinates'],
        ['Preserves', 'Topology / neighbourhoods', 'Small distances (local structure)', 'Maximum variance (global)'],
        ['New data points', 'YES — find BMU', 'NO — must rerun optimisation', 'YES — project onto components'],
        ['Non-linear?', 'YES', 'YES', 'NO (linear only)'],
        ['Cluster visibility', 'Excellent (U-matrix)', 'Good', 'Moderate'],
        ['Computational cost', 'High (iterative training)', 'High (gradient descent)', 'Low (eigendecomposition)'],
    ]
))
story.append(sp(4))
story.append(inference_box(
    'Sammon\'s Mapping is the method of choice when LOCAL STRUCTURE preservation is paramount — for example, when you need to detect fine-grained clusters in medical data, identify subtypes of diseases, or reveal overlapping populations in genetic data. Its normalisation by original distance is a principled way of encoding the perceptual reality that errors among nearby points are more damaging than errors among distant points.'
))
story.append(exam_tip_box(
    'Sammon Mapping formula: E_S = (1/Sum_dij*) x Sum[(dij* - dij)^2 / dij*]. Key: division by dij* means SMALL distance errors are penalised MORE. Update rule: Newton\'s gradient descent on E_S. Limitation: cannot project new data points (must rerun). SAMANN = neural network version that CAN project new points. Compared to SOM and MDS: Sammon best for local structure, SOM best for online/new data, PCA fastest but linear only.'
))
story.append(PageBreak())

# ══════════════════════════════════════════════════════════════
# TOPIC 12 — RECOMMENDER SYSTEMS
# ══════════════════════════════════════════════════════════════
story += section_header('12.  Recommender Systems — A Brief Overview')

story.append(concept_box(
    'Recommender Systems',
    'Recommender systems are user support tools that help people by facilitating access to items (news, products, web pages, music, films) that they are likely to find useful, interesting, or relevant — without the user having to explicitly search for them.',
    'Two main modes: (1) PROACTIVE — the system anticipates user needs and pushes recommendations without being asked. (2) ON-DEMAND — the user explicitly requests recommendations ("show me similar movies"). Both modes require a model of user preferences.'
))
story.append(sp(6))

story.append(p(f'{b("DETAILED EXPLANATION")}', sub_h))
story.append(sub_header('12.1 Content-Based Filtering'))
story.append(bullet(f'{b("Core Principle")}: Driven by the premise that user preferences persist over time. The system builds a PROFILE of the active user based on items they have already rated or interacted with. New items are recommended by measuring similarity between new items and the user\'s existing profile items.'))
story.append(bullet(f'{b("Process")}: (1) Build user profile — list of items the user has rated + the content features of those items. (2) For each new item, compute content similarity to the profile items. (3) Recommend items with highest similarity scores.'))
story.append(bullet(f'{b("Item Features")}: For movies — genre, director, cast, keywords. For articles — topic, keywords, author. For music — tempo, genre, artist.'))
story.append(bullet(f'{b("Similarity Measure")}: Cosine similarity for text content: sim(A, B) = (A·B) / (||A|| x ||B||). TF-IDF weighting for document features.'))
story.append(bullet(f'{b("Example")}: A user frequently reads articles about machine learning → content-based system recommends more AI/ML articles based on keyword overlap.'))
story.append(bullet(f'{b("Advantage")}: No cold-start for items — can recommend new items immediately as long as their content is analysable. Does not require data from other users.'))
story.append(bullet(f'{b("Disadvantage")}: Limited to items similar to what the user already knows. Cannot recommend items in a different category even if the user might enjoy them (over-specialisation / "filter bubble").'))

story.append(sub_header('12.2 Collaborative Filtering'))
story.append(bullet(f'{b("Core Principle")}: "Users who agreed in the past tend to agree in the future." Maintains profiles of many users. Finds users with similar rating patterns (neighbours) to the active user. Recommends items those similar users liked that the active user has not yet seen.'))
story.append(bullet(f'{b("Process")}: (1) Build user-item rating matrix. (2) Find k most similar users to the active user (using Pearson correlation or cosine similarity of rating vectors). (3) Compute predicted rating for unrated items by the active user as weighted average of neighbour ratings. (4) Recommend top-N items with highest predicted ratings.'))
story.append(bullet(f'{b("Pearson Correlation (Similarity)")}: sim(A, B) = Σ(r_Ai - r_A_mean)(r_Bi - r_B_mean) / sqrt[Σ(r_Ai - r_A_mean)^2 x Σ(r_Bi - r_B_mean)^2]'))
story.append(bullet(f'{b("Example")}: Netflix — "Users who watched and liked Inception also liked Interstellar." "Customers who bought X also bought Y" (Amazon).'))
story.append(bullet(f'{b("Advantage")}: Can recommend items completely different from what the user has seen before (serendipitous discovery). Does not require content analysis of items.'))
story.append(bullet(f'{b("Disadvantage")}: Cold Start Problem — new users with no ratings cannot receive recommendations. New items with no ratings cannot be recommended. Sparsity problem — most users rate very few items (sparse matrix).'))

story.append(sub_header('12.3 Hybrid Recommender Systems'))
story.append(bullet(f'{b("Motivation")}: Neither content-based nor collaborative filtering alone is perfect. Hybrid systems combine both to leverage strengths and mitigate weaknesses of each.'))
story.append(bullet(f'{b("Example systems")}: Netflix Prize winner — used a sophisticated ensemble combining collaborative filtering, matrix factorisation, and content features. P-Tango, Fab, NewsDude.'))
story.append(bullet(f'{b("Integration strategies")}: (1) Weighted hybrid — combine scores from both systems linearly. (2) Switching hybrid — use one or the other based on context (e.g., use content-based for new users, switch to collaborative once enough ratings are collected). (3) Feature combination — use content features as input to collaborative filtering.'))

story.append(sp(4))
story.append(diagram_box('Hybrid Recommender System Architecture',
    'Draw a central "Active User" box at the top. Two parallel boxes below: LEFT = "Content-Based Engine" (fed by "Item Content Database"). RIGHT = "Collaborative Filtering Engine" (fed by "User Ratings Database"). Both engines receive the "User Profile / Query" from the Active User. Both engines produce "Candidate Recommendations." Both feed into a central "HYBRID COMBINER" box that merges and ranks. Output: "Top-N Recommendations" displayed to user. Draw a feedback arrow from user back up (ratings/clicks → update both databases).'))

story.append(sub_header('12.4 Key Issues in Recommender Systems'))
story.append(bullet(f'{b("Sparsity Problem")}: The user-item rating matrix is typically over 99% sparse (most users rate very few items). This makes similarity computations unreliable. Solution: implicit feedback (clicks, time spent, purchase history) to supplement explicit ratings.'))
story.append(bullet(f'{b("Cold Start Problem")}: New users (no ratings) and new items (no ratings) cannot be effectively recommended. Solution: Ask new users for initial preferences. Use content-based for new items.'))
story.append(bullet(f'{b("Scalability")}: Netflix-scale systems have millions of users and millions of items. Exhaustive computation is impossible. Solution: Matrix factorisation (SVD), approximate nearest neighbours.'))
story.append(bullet(f'{b("Trustworthiness / Transparency")}: Users cannot understand WHY a system recommends something ("black box"). Solution: Argumentation-based recommenders (e.g., ArguNet uses Defeasible Logic Programming — DeLP — to provide rationally justified recommendations that can be explained and challenged).'))
story.append(bullet(f'{b("Shilling Attacks")}: Malicious users inject fake ratings to manipulate recommendations. Solution: Robust estimation methods, attack detection algorithms.'))

story.append(sp(4))
story.append(inference_box(
    'Recommender systems are among the most commercially successful AI applications — directly responsible for 35% of Amazon\'s revenue and 75% of Netflix viewing time. The tension between accuracy (what the user will definitely like) and serendipity (surprising recommendations that expand preferences) is the central design challenge. Hybrid systems that balance personalisation with discovery consistently outperform single-approach systems in industrial deployments.'
))
story.append(sp(4))
story.append(pros_cons_box(
    ['Content-Based: no cold start for new items, no need for other users\' data',
     'Collaborative: discovers serendipitous cross-category recommendations',
     'Hybrid: best accuracy by combining both approaches',
     'Implicit feedback reduces sparsity without burdening users',
     'Matrix factorisation handles large-scale systems efficiently',
     'Argumentation-based: provides explainable, trustworthy recommendations'],
    ['Content-Based: over-specialisation / filter bubble',
     'Collaborative: cold start for new users and items',
     'Both: sparsity in rating matrix reduces accuracy',
     'Privacy concerns — all approaches require collecting user data',
     'Scalability challenges for systems with millions of users and items',
     'Shilling attacks can corrupt collaborative filtering recommendations']
))
story.append(exam_tip_box(
    'Content-Based = recommend items SIMILAR TO WHAT USER LIKED (item features). Collaborative = recommend items that SIMILAR USERS liked (user-user similarity). Hybrid = best of both. Three key issues: Sparsity (most users rate few items), Cold Start (new user/item has no data), Scalability (millions of users). Argumentation-based systems (DeLP/ArguNet) = explainable recommendations.'
))
story.append(PageBreak())

# ══════════════════════════════════════════════════════════════
# TOPIC 13 — KNOWLEDGE MODELLING UML
# ══════════════════════════════════════════════════════════════
story += section_header('13.  Knowledge Modelling Using UML')

story.append(concept_box(
    'Knowledge Modelling Using UML',
    'Knowledge Modelling is the process of formally representing the domain knowledge of a Knowledge-Based System (KBS) during the analysis and design phase of development. Using UML (Unified Modelling Language) for knowledge modelling bridges the gap between Knowledge Engineering (KBS development) and Software Engineering (conventional systems development), enabling a common modelling language for integrated enterprise systems.',
    'Traditional KBS used specialised notations (CommonKADS, KADS) that were unfamiliar to software engineers. Adopting UML allows KBS developers and software developers to collaborate using the same diagrams and notation, facilitating integration of KBS into larger enterprise systems.'
))
story.append(sp(6))

story.append(p(f'{b("DETAILED EXPLANATION")}', sub_h))
story.append(sub_header('Why UML for Knowledge Modelling?'))
story.append(bullet(f'{b("Universal Standard")}: UML is the de facto standard for object-oriented system modelling. Adopting UML for KBS means system analysts already familiar with UML can work on KBS projects without learning a new notation.'))
story.append(bullet(f'{b("Enterprise Integration")}: Modern KBS are components within larger enterprise systems (CAD, GIS, ERP, SCADA). A common modelling notation enables seamless integration, reuse, and maintenance.'))
story.append(bullet(f'{b("OO Benefits")}: UML supports abstraction, inheritance, polymorphism, and encapsulation — all of which are valuable in KBS design. Knowledge concepts can be modelled as classes; inference procedures as operations.'))
story.append(bullet(f'{b("Tool Support")}: Rich ecosystem of UML tools (Rational Rose, Enterprise Architect, ArgoUML) can be used directly for KBS modelling.'))

story.append(sub_header('UML Extension Mechanisms for KBS'))
story.append(cmp_table(
    ['Mechanism', 'Description', 'When to Use'],
    [
        ['Lightweight Extension (UML Profile)', 'Predefined set of Stereotypes, Tagged Values, and Constraints. Extends UML semantics without changing its metamodel. Stereotypes marked with guillemets: «stereotype».', 'When UML constructs are sufficient with minor specialisation — less overhead, backward compatible with UML tools.'],
        ['Heavyweight Extension (Metamodel Extension)', 'Defined through the Meta-Object Facility (MOF). Creates an entirely new metamodel. Introduces new modelling constructs not present in UML.', 'When the semantic gap between UML and the domain is too large — when UML constructs fundamentally cannot represent the required concepts.'],
    ],
    col_widths=[2.5*cm, 6.5*cm, 6.5*cm]
))

story.append(sub_header('KBS Modelling Concepts (CommonKADS UML Profile)'))
story.append(p('The UML Profile for KBS is based on the CommonKADS Conceptual Modelling Language (CML). Key stereotypes and their roles:', body))
story.append(cmp_table(
    ['Stereotype', 'Concept', 'Role in KBS'],
    [
        ['«Concept»', 'Domain Class', 'Represents a category of domain objects with attributes. Example: «Concept» Patient with slots: name, age, symptoms.'],
        ['«KnowledgeBase»', 'Knowledge Base', 'Collection of domain knowledge (production rules) organised into modules. The IF-THEN rules reside here.'],
        ['«ProductionRule»', 'IF-THEN Rule', 'Single knowledge rule: IF <condition> THEN <action>. Resides inside the KnowledgeBase.'],
        ['«FactBase»', 'Working Memory', 'Stores the current attribute values (facts) of domain concept instances during inference.'],
        ['«Inference»', 'Reasoning Step', 'Lowest-level functional decomposition — a primitive reasoning/matching step that applies rules to facts.'],
        ['«StaticRole»', 'Rule Selector', 'Fetches relevant production rules from the KnowledgeBase for a specific inference step.'],
        ['«DynamicRole»', 'Fact Provider', 'Specifies information flow — the attribute instances (facts) that flow into and out of an inference step.'],
        ['«Task»', 'Overall Goal', 'Defines the overall reasoning function: what the KBS must accomplish (inputs, outputs, goal).'],
        ['«TaskMethod»', 'Task Realisation', 'Specifies HOW the Task is achieved through decomposition into sub-functions and inferences.'],
        ['«TransferFunction»', 'I/O Handler', 'Handles information transfer between the KBS and external entities (user interface, databases).'],
    ]
))

story.append(sp(4))
story.append(diagram_box('KBS Knowledge Model UML Diagram',
    'Draw a UML class diagram with the following boxes using the stereotype notation: (1) «Concept» CPG — with attributes: name, factors, evidence_strength. (2) «FactBase» — connected to «Concept» by dependency arrow labeled "stores attribute instances of." (3) «DynamicRole» — connected to «FactBase» by arrow labeled "gets facts from." (4) «StaticRole» — connected to «KnowledgeBase» by arrow labeled "fetches rules from." (5) «KnowledgeBase» — box listing rule types inside. (6) «Inference» — central box connected to both «DynamicRole» (gets facts) and «StaticRole» (gets rules). (7) «TaskMethod» — connected to «Inference» by arrow. (8) «Task» — connected to «TaskMethod». (9) «TransferFunction» — connected to «TaskMethod» on one side and "User Interface" on the other. This is the standard KBS architecture diagram.'))

story.append(sp(4))
story.append(inference_box(
    'The UML profile approach for KBS modelling is a pragmatic engineering solution to a real problem: knowledge engineers and software engineers have historically used incompatible notations, making it difficult to build integrated systems. By grounding KBS modelling in UML, the approach brings KBS development into the mainstream of enterprise software engineering — enabling code generation from models, reuse of UML tool ecosystems, and easier collaboration between KBS and conventional development teams.'
))
story.append(exam_tip_box(
    'UML for KBS uses STEREOTYPES to extend UML for knowledge modelling. Key stereotypes: «Concept» (domain class), «KnowledgeBase» (IF-THEN rules), «FactBase» (working memory), «Inference» (reasoning step), «StaticRole» (fetches rules), «DynamicRole» (carries facts), «Task» (overall goal), «TransferFunction» (I/O). Lightweight extension = UML Profile (add stereotypes). Heavyweight extension = new metamodel via MOF.'
))
story.append(PageBreak())

# ══════════════════════════════════════════════════════════════
# TOPIC 14 — CPG CASE STUDY
# ══════════════════════════════════════════════════════════════
story += section_header('14.  Case Study: Clinical Practice Guideline (CPG) Recommendations')

story.append(concept_box(
    'CPG Recommendation System — Case Study Overview',
    'The CPG (Clinical Practice Guideline) system is a Knowledge-Based System implemented to provide medical recommendations based on evidence strength, recommendation category, and clinical factors. It serves as a complete, practical illustration of applying the UML Knowledge Modelling Profile to a real healthcare domain.',
    'CPGs are systematically developed statements to assist clinical decisions about appropriate healthcare for specific clinical circumstances. The KBS encodes CPG knowledge as production rules and uses inference to recommend the appropriate clinical action given patient-specific inputs.'
))
story.append(sp(6))

story.append(p(f'{b("DETAILED EXPLANATION")}', sub_h))
story.append(sub_header('14.1 CPG Domain Model — Five Recommendation Categories'))
story.append(p('The domain concept "CPG" has five sub-categories, each modelled as a «Concept» class:', body))
story.append(bullet(f'{b("CPGManagement")}: Rules related to overall patient management protocols — admission, monitoring, discharge decisions.'))
story.append(bullet(f'{b("CPGCleansing")}: Rules for clinical cleansing and wound care — antiseptic choice, frequency, technique.'))
story.append(bullet(f'{b("CPGQualityAssurance")}: Rules for maintaining care quality standards — documentation, review, audit.'))
story.append(bullet(f'{b("CPGAssessment")}: Rules for patient assessment — initial evaluation, risk stratification, follow-up.'))
story.append(bullet(f'{b("CPGEducation")}: Rules for patient and staff education — information provision, training requirements.'))
story.append(p('Each CPG concept has three key attributes:', body))
story.append(bullet(f'{b("name")}: The specific guideline or recommendation identifier.'))
story.append(bullet(f'{b("factors")}: Patient-specific clinical factors that modulate the recommendation (e.g., immunosuppression, diabetes, age).'))
story.append(bullet(f'{b("evidence_strength")}: The level of scientific evidence supporting the recommendation: Level I (strongest — from randomised controlled trials), Level II (from case-controlled studies), Level III (expert consensus opinion).'))

story.append(sub_header('14.2 Four Types of Production Rules'))
story.append(p('The CPG Knowledge Base contains production rules of four types, depending on which attributes are used:', body))
story.append(cmp_table(
    ['Rule Type', 'Condition', 'Example Rule', 'When Used'],
    [
        ['Type (a)', 'Evidence strength only', 'IF evidence_strength = "Level I" THEN recommendation_id = "I1" (Grade A)', 'When the category and factors are irrelevant — highest evidence applies universally'],
        ['Type (b)', 'Evidence AND category', 'IF evidence = "Level II" AND category = "Cleansing" THEN id = "II2-C"', 'When recommendation differs by clinical category but not patient factors'],
        ['Type (c)', 'Category only', 'IF category = "Assessment" THEN id = "Cat-A"', 'When recommendation is category-specific regardless of evidence level'],
        ['Type (d)', 'Factors + evidence + category', 'IF factors = "immunocompromised" AND evidence = "Level I" AND category = "Management" THEN id = "I1-M-F"', 'Most specific — personalised recommendation based on all three attributes'],
    ]
))

story.append(sub_header('14.3 System Architecture'))
story.append(bullet(f'{b("Implementation")}: Built using JESS (Java Expert System Shell) — a Java-based implementation of the CLIPS rule engine. JESS implements JSR 94 (Java Rule Engine API — the standard interface for Java rule engines).'))
story.append(bullet(f'{b("Knowledge Base Module")}: All production rules organised into JESS defmodule constructs by recommendation category. Each module contains the 4 rule types for its category.'))
story.append(bullet(f'{b("Fact Base")}: JESS working memory stores the current patient\'s CPG attribute instances (evidence_strength value, category, factors) as JESS facts.'))
story.append(bullet(f'{b("Inference Engine")}: JESS\'s built-in Rete algorithm matches fact patterns against rule conditions. Single execution mode: one inference step fires the single most specific matching rule.'))
story.append(bullet(f'{b("Interface Module")}: A question-based interview collects patient attribute values. Questions presented sequentially: first ask about evidence strength, then category, then patient-specific factors.'))
story.append(bullet(f'{b("Report Module")}: Generates a formatted recommendation report after inference. Displays: matched rule ID, recommendation text, evidence basis, and category context.'))

story.append(sub_header('14.4 System Operation — Sequence of Events'))
story.append(p('The dynamic behaviour of the CPG system, captured in a UML Sequence Diagram:', body))
story.append(trace_table(
    ['Step', 'From', 'To', 'Message / Action'],
    [
        ['1', 'User', 'Interface', 'Provide evidence_strength, category, factors (via interview questions)'],
        ['2', 'Interface', 'FactBase', 'upload_facts(evidence_strength, category, factors)'],
        ['3', 'DynamicRole', 'FactBase', 'get_facts() → returns current attribute instances'],
        ['4', 'DynamicRole', 'Inference', 'inference_matching_facts(fact_values)'],
        ['5', 'StaticRole', 'KnowledgeBase', 'inference_matching_rules() → returns applicable production rules'],
        ['6', 'Inference', 'Inference', 'Match facts against rules → identify fired rule → get recommendation_id'],
        ['7', 'Inference', 'DynamicRole', 'return recommendation_result'],
        ['8', 'DynamicRole', 'Interface', 'return recommendation_id and recommendation_text'],
        ['9', 'Interface', 'User', 'Display recommendation report (rule fired, evidence basis, category)'],
    ],
    col_widths=[1.0*cm, 2.5*cm, 2.5*cm, 9.5*cm]
))

story.append(sp(4))
story.append(diagram_box('CPG UML Sequence Diagram',
    'Draw 7 vertical swimlanes (lifelines) labeled from left to right: User, Interface, DynamicRole, FactBase, TransferFunction, Inference, StaticRole, KnowledgeBase. Draw horizontal arrows (messages) in order: (1) User → Interface: "provide attributes." (2) Interface → FactBase: "upload facts." (3) DynamicRole ← FactBase: "get facts." (4) DynamicRole → Inference: "matching facts." (5) StaticRole → KnowledgeBase: "get matching rules." (6) KnowledgeBase → StaticRole: "rules returned." (7) Inference → DynamicRole: "recommendation result." (8) Interface ← DynamicRole: "recommendation." (9) User ← Interface: "display report." Number each arrow 1-9. This is the standard CPG sequence diagram from the textbook.'))

story.append(sp(4))
story.append(example_box(
    'CPG Production Rule Examples (JESS Syntax)',
    'Rule Type (a) — Evidence strength only:',
    '  (defrule Level-I-recommendation',
    '    (CPG (evidence_strength "Level I"))',
    '    => (assert (Recommendation (id "I1") (grade "A") (text "Strong evidence — follow guideline without exception"))))',
    'Rule Type (d) — Most specific (factors + evidence + category):',
    '  (defrule Level-I-Management-Immunocompromised',
    '    (CPG (evidence_strength "Level I") (category "Management") (factors "immunocompromised"))',
    '    => (assert (Recommendation (id "I1-M-F") (text "Modified management protocol for immunocompromised patients — more frequent monitoring required"))))',
    'The Rete algorithm selects the MOST SPECIFIC matching rule. Type (d) fires over Type (a) if both match, because Type (d) has more specific conditions.',
))
story.append(sp(4))
story.append(inference_box(
    'The CPG case study demonstrates the complete KBS development lifecycle using UML: domain analysis (5 CPG categories), knowledge modelling (UML profile with stereotypes), implementation (JESS rules), and validation (clinical expert review). The use of a standard UML profile allowed the KBS to be documented, reviewed, and maintained using the same tools as the surrounding hospital information system — dramatically reducing integration costs and improving long-term maintainability.'
))
story.append(sp(4))
story.append(pros_cons_box(
    ['UML profile provides standard, familiar notation for KBS modelling',
     'JESS provides efficient Rete-based rule matching',
     'Four rule types allow varying levels of specificity',
     'Sequence diagram clearly documents system interactions',
     'Can be integrated with hospital EMR systems using standard Java APIs',
     'Modular Knowledge Base allows updating one category without affecting others',
     'Explainable — can trace which rule fired and why'],
    ['JESS/CLIPS syntax is not user-friendly for non-programmers to add rules',
     'Single execution mode fires only one rule — may miss complex interactions',
     'Evidence levels are manually assigned — subjective for Level III (expert opinion)',
     'System requires redeployment when new CPG rules are added',
     'Cannot handle uncertainty in factors (e.g., partial immunosuppression)',
     'No learning capability — rules must be manually updated when evidence changes']
))
story.append(exam_tip_box(
    'CPG domain: 5 categories (Management, Cleansing, QualityAssurance, Assessment, Education). 3 attributes: name, factors, evidence_strength (Level I/II/III). 4 rule types based on attribute combinations. Implementation: JESS (Java Expert System Shell). Sequence: User → Interface → FactBase → DynamicRole → Inference ← StaticRole ← KnowledgeBase → result → User. UML stereotypes used: «Concept», «FactBase», «DynamicRole», «StaticRole», «Inference», «KnowledgeBase», «TaskMethod», «TransferFunction».'
))

story.append(sp(10))
story.append(HR(1.5, RED))
story.append(p('UNIT 2  ·  EMERGING ARTIFICIAL INTELLIGENCE  ·  15 HOURS  ·  ALL TOPICS COVERED', footer_s))
story.append(p('Sources: EAI Reference Book (Kotsiantis, Dzemyda, Chesnevar, Abdullah et al.)', S('fs2', fontName='Helvetica-Oblique', fontSize=7.5, textColor=MUTED, alignment=TA_CENTER)))

doc.build(story)
print('Unit 2 PDF generated successfully!')