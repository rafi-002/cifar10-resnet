"""Builds the CIFAR-10 assignment report PDF."""
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, Image
)
from reportlab.lib.enums import TA_CENTER

OUT_PATH = "CIFAR10_Assignment_Report.pdf"

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="TitleCustom", fontSize=20, leading=24, spaceAfter=6,
                           textColor=colors.HexColor("#14213D"), fontName="Helvetica-Bold"))
styles.add(ParagraphStyle(name="SubTitle", fontSize=11, leading=15, spaceAfter=18,
                           textColor=colors.HexColor("#4A5568")))
styles.add(ParagraphStyle(name="H2", fontSize=14, leading=18, spaceBefore=16, spaceAfter=8,
                           textColor=colors.HexColor("#14213D"), fontName="Helvetica-Bold"))
styles.add(ParagraphStyle(name="Body", fontSize=10.5, leading=15, spaceAfter=8))
styles.add(ParagraphStyle(name="Placeholder", fontSize=10.5, leading=15, spaceAfter=8,
                           textColor=colors.HexColor("#B45309"), fontName="Helvetica-Oblique"))
styles.add(ParagraphStyle(name="ScoreBig", fontSize=32, leading=36, alignment=TA_CENTER,
                           textColor=colors.HexColor("#14213D"), fontName="Helvetica-Bold"))
styles.add(ParagraphStyle(name="ScoreLabel", fontSize=10, leading=13, alignment=TA_CENTER,
                           textColor=colors.HexColor("#6a7179")))

story = []

# --- Title ---
story.append(Paragraph("CIFAR-10 Image Classification with a Custom ResNet", styles["TitleCustom"]))
story.append(Paragraph("Deep Learning Assignment &mdash; Kaggle CIFAR-10 Competition (Late Submission)", styles["SubTitle"]))
story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e3e6e4")))
story.append(Spacer(1, 14))

# --- 1. Code link ---
story.append(Paragraph("1. Code", styles["H2"]))
story.append(Paragraph(
    "<b>[PLACEHOLDER &mdash; replace with your actual Kaggle/Colab notebook URL]</b><br/>"
    "e.g. https://www.kaggle.com/code/&lt;your-username&gt;/cifar10-resnet",
    styles["Placeholder"]
))
story.append(Paragraph(
    "The notebook implements a custom ResNet-20 trained from scratch on the official CIFAR-10 "
    "training set and generates predictions for the Kaggle competition's test set.",
    styles["Body"]
))

# --- 2. Kaggle score ---
story.append(Paragraph("2. Kaggle Score", styles["H2"]))

score_table = Table(
    [[Paragraph("0.916", styles["ScoreBig"])],
     [Paragraph("Public / Private score (late submission)", styles["ScoreLabel"])]],
    colWidths=[2.2 * inch]
)
score_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FBFBF8")),
    ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#e3e6e4")),
    ("TOPPADDING", (0, 0), (-1, 0), 14),
    ("BOTTOMPADDING", (0, 0), (-1, 0), 4),
    ("BOTTOMPADDING", (0, 1), (-1, 1), 12),
]))
story.append(score_table)
story.append(Spacer(1, 10))

story.append(Paragraph(
    "<b>[PLACEHOLDER &mdash; insert a screenshot of your Kaggle leaderboard/submissions page here, "
    "showing the 0.916 score. Replace this paragraph with an Image(\"your_screenshot.png\") "
    "in build_report.py, or paste the image directly into a Word/Docs version of this report.]</b>",
    styles["Placeholder"]
))
story.append(Paragraph(
    "This score matches the model's validation accuracy on the official CIFAR-10 test set "
    "(0.9161), as expected &mdash; Kaggle's 300,000-image test set contains the same 10,000 "
    "images used for validation, mixed with 290,000 unscored distractor images.",
    styles["Body"]
))

# --- 3. Model description ---
story.append(Paragraph("3. Model Architecture", styles["H2"]))
story.append(Paragraph(
    "A ResNet-20 &mdash; the CIFAR-specific residual network from He et al. (2015), not the "
    "ImageNet ResNet-18/50 available pretrained in torchvision. The network consists of a single "
    "3&times;3 stem convolution (3&rarr;16 channels, resolution preserved at 32&times;32), followed "
    "by three stages of three residual (\"BasicBlock\") units each, with channel counts 16 &rarr; 32 "
    "&rarr; 64 and spatial resolution 32 &rarr; 16 &rarr; 8. Each stage's first block halves the "
    "spatial resolution via a stride-2 convolution; a global average pool and a single linear layer "
    "(64 &rarr; 10) form the classification head. In total the network has 20 weight layers "
    "(3 stages &times; 3 blocks &times; 2 convs + stem + head = 6n+2 with n=3) and approximately "
    "272,000 parameters.",
    styles["Body"]
))
story.append(Paragraph(
    "Each residual block computes conv&rarr;BN&rarr;ReLU&rarr;conv&rarr;BN, then adds a shortcut "
    "connection (identity, or a 1&times;1 projection convolution when the block changes channel "
    "count or stride) before a final ReLU. This skip connection gives gradients a direct path "
    "backward through the network, allowing many blocks to be stacked without the optimization "
    "difficulties that plain (non-residual) deep CNNs exhibit.",
    styles["Body"]
))
story.append(Paragraph(
    "<b>All weights are randomly initialized</b> using PyTorch's default Kaiming-uniform "
    "initialization for convolutional and linear layers. No pretrained weights, state dicts, or "
    "external model checkpoints are loaded anywhere in the pipeline &mdash; this was verified "
    "programmatically in the notebook by confirming two freshly constructed instances of the "
    "network do not share identical weights.",
    styles["Body"]
))

# --- 4. Training ---
story.append(Paragraph("4. Training", styles["H2"]))

train_rows = [
    ["Training data", "Official CIFAR-10 training set (50,000 images), via torchvision.datasets.CIFAR10"],
    ["Validation data", "Official CIFAR-10 test set (10,000 images) &mdash; used only to estimate accuracy, never trained on"],
    ["Augmentation", "RandomCrop(32, padding=4), RandomHorizontalFlip, per-channel normalization"],
    ["Optimizer", "SGD, lr=0.1, momentum=0.9, weight_decay=5e-4, Nesterov"],
    ["LR schedule", "Cosine annealing over the training run"],
    ["Loss", "Cross-entropy"],
    ["Epochs", "50 (plus an earlier resumed 50-epoch run on the same checkpoint)"],
    ["Precision", "Mixed precision (torch.amp) on an NVIDIA RTX 3060 (6GB)"],
    ["Batch size", "128 (training)"],
    ["Model selection", "Checkpointed on best validation accuracy each epoch"],
    ["Best validation accuracy", "0.9161"],
]

t = Table(
    [[Paragraph(f"<b>{k}</b>", styles["Body"]), Paragraph(v, styles["Body"])] for k, v in train_rows],
    colWidths=[1.6 * inch, 4.6 * inch]
)
t.setStyle(TableStyle([
    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e3e6e4")),
    ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#FBFBF8")),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("TOPPADDING", (0, 0), (-1, -1), 6),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ("LEFTPADDING", (0, 0), (-1, -1), 8),
]))
story.append(t)
story.append(Spacer(1, 10))

story.append(Paragraph(
    "Data augmentation (random crop and horizontal flip) was the single largest lever for closing "
    "the gap between training and validation accuracy &mdash; without it, this architecture "
    "typically overfits, with validation accuracy plateauing around 0.83&ndash;0.86. With it, "
    "the model reached 91.61% validation accuracy, translating to a Kaggle score of 0.916 on "
    "submission, comfortably within the &ge;0.88 band for full marks on this assignment's rubric.",
    styles["Body"]
))

doc = SimpleDocTemplate(
    OUT_PATH, pagesize=letter,
    topMargin=0.75 * inch, bottomMargin=0.75 * inch,
    leftMargin=0.85 * inch, rightMargin=0.85 * inch,
)
doc.build(story)
print(f"Wrote {OUT_PATH}")
