#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fill the official Genes Word template (.dot) with manuscript content."""
from __future__ import annotations

import re
import zipfile
from io import BytesIO
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt
from docx.text.paragraph import Paragraph

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "图件"
DOT = ROOT / "论文" / "官方模板" / "genes-template.dot"
TEMPLATE_DOCX = ROOT / "论文" / "官方模板" / "genes-template.docx"
OUT = ROOT / "论文" / "genes_paper.docx"


def dot_to_docx(dot: Path, dest: Path) -> None:
    buf = BytesIO(dot.read_bytes())
    out = BytesIO()
    with zipfile.ZipFile(buf, "r") as zin, zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "[Content_Types].xml":
                data = data.replace(
                    b"application/vnd.openxmlformats-officedocument.wordprocessingml.template.main+xml",
                    b"application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml",
                )
            zout.writestr(item, data)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(out.getvalue())


def clear_body(doc: Document) -> None:
    body = doc.element.body
    for child in list(body):
        if child.tag == qn("w:sectPr"):
            continue
        body.remove(child)


def add_styled(doc: Document, style: str, text: str = "") -> Paragraph:
    p = doc.add_paragraph(style=style)
    if text:
        fill_runs(p, text)
    return p


def fill_runs(paragraph: Paragraph, text: str, *, bold_prefix: str | None = None) -> None:
    """Write text; *...* becomes italic. Optional bold_prefix is written bold first."""
    paragraph.clear()
    if bold_prefix:
        run = paragraph.add_run(bold_prefix)
        run.bold = True
    parts = re.split(r"(\*[^*]+\*)", text)
    for part in parts:
        if not part:
            continue
        if part.startswith("*") and part.endswith("*") and len(part) > 2:
            run = paragraph.add_run(part[1:-1])
            run.italic = True
        else:
            paragraph.add_run(part)


def add_back(doc: Document, label: str, body: str) -> None:
    p = add_styled(doc, "MDPI_6.2_back_matter")
    fill_runs(p, body, bold_prefix=f"{label}: ")


def add_figure(doc: Document, path: Path, caption: str, width_in: float = 5.8) -> None:
    p = add_styled(doc, "MDPI_5.2_figure")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if path.exists():
        run = p.add_run()
        run.add_picture(str(path), width=Inches(width_in))
    else:
        fill_runs(p, f"[Missing figure: {path.name}]")
    cap = add_styled(doc, "MDPI_5.1_figure_caption")
    fill_runs(cap, caption)


def add_table(doc: Document, headers: list[str], rows: list[list[str]], caption: str) -> None:
    cap = add_styled(doc, "MDPI_4.1_table_caption")
    fill_runs(cap, caption)
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    try:
        table.style = "MDPI_table"
    except KeyError:
        table.style = "Table Grid"
    for j, h in enumerate(headers):
        cell = table.rows[0].cells[j]
        cell.text = ""
        run = cell.paragraphs[0].add_run(h)
        run.bold = True
        run.font.size = Pt(9)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = table.rows[i + 1].cells[j]
            cell.text = ""
            run = cell.paragraphs[0].add_run(str(val))
            run.font.size = Pt(9)


def main() -> None:
    if not DOT.exists():
        raise SystemExit(f"Missing official template: {DOT}")
    dot_to_docx(DOT, TEMPLATE_DOCX)
    doc = Document(str(TEMPLATE_DOCX))
    clear_body(doc)

    # --- Front matter ---
    add_styled(doc, "MDPI_1.1_article_type", "Article")
    title = (
        "Benchmarking AgroNT against a DeepSEA-style Convolutional Network for "
        "Tissue-Averaged Gene Expression Classification in *Arabidopsis thaliana*"
    )
    add_styled(doc, "MDPI_1.2_title", title)
    add_styled(doc, "MDPI_1.3_authornames", "Firstname Lastname 1,† and Secondname Lastname 1,*")
    add_styled(
        doc,
        "MDPI_1.6_affiliation",
        "1\tDepartment of Plant Sciences, University Name, City ZIP, Country; email1@university.edu",
    )
    add_styled(
        doc,
        "MDPI_1.6_affiliation",
        "*\tCorrespondence: email2@university.edu",
    )
    add_styled(doc, "MDPI_1.7_abstract", "Abstract")
    add_styled(
        doc,
        "MDPI_1.7_abstract",
        "Background/Objectives: Compact convolutional neural networks (CNNs) and plant DNA foundation "
        "models are both used for sequence-based gene-expression prediction, yet they are rarely compared "
        "under identical splits, labels, and metrics. We compare a DeepSEA-style 1D-CNN with LoRA-tuned "
        "AgroNT-1B on a derived high-versus-low task in *Arabidopsis thaliana*. "
        "Methods: High/low classes were obtained from the Plants Genomic Benchmark (PGB) Arabidopsis "
        "gene-expression FASTA by averaging 56 tissue/sample abundances and thresholding at the "
        "training-set median; this is a derived binary collapse, not the official multi-tissue "
        "regression. Both models were evaluated with sliding-window mutagenesis on a concatenated "
        "flanking representation (CNN retrained on that set; AgroNT applied as the PGB LoRA checkpoint), "
        "in-silico scoring of natural flanking variants on 250 structured genes, application of "
        "Arabidopsis checkpoints to four crop PGB sets, and midpoint shortening of the published 6 kb "
        "window. CNN-only analyses were DeepLift/TF-MoDISco motif recovery and three-seed replication. "
        "Tissue-label concordance quantified the averaged-label design. "
        "Results: Under the reported budgets, AgroNT outperformed the CNN (accuracy 86.8% vs. 75.4%; "
        "AUROC 0.939 vs. 0.842; ΔAUROC bootstrap 95% CI 0.086–0.108; McNemar p = 6.7×10⁻⁵⁴; 532 vs. 142 "
        "discordant pairs). Multi-seed CNN AUROC was 0.844±0.003; AgroNT is reported from a single seed. "
        "Matched mutagenesis placed predictive "
        "signal in proximal annotated flanks for both models; DeepLift/TOMTOM hits included DOF-, SPL-, "
        "and NAC-family motifs (provisional). CNN variant |Δ| ranked UTRs above terminator and promoter, "
        "whereas AgroNT ranked terminator highest; promoter was lowest for both variant maps. "
        "Single-tissue labels agreed with the averaged label for a mean of 92.1% of genes "
        "(range 78.7–95.4%). Midpoint-cropping the 6 kb window to 3 kb or 1.5 kb reduced CNN AUROC to "
        "0.542 and 0.545 and AgroNT AUROC to 0.625 and 0.580. Applied without further fine-tuning, the Arabidopsis CNN transferred to rice, "
        "maize, tomato, and soybean at AUROC 0.69–0.76, and the Arabidopsis AgroNT LoRA checkpoint at "
        "AUROC 0.815–0.894. "
        "Conclusions: On this derived tissue-averaged task and under the reported budgets, LoRA-tuned "
        "AgroNT outperforms this DeepSEA-style CNN by a large, statistically supported margin. "
        "Mutagenesis on the structured representation concentrates in proximal flanks; variant |Δ| ranks "
        "are architecture-specific except for a shared promoter-lowest pattern. These results quantify "
        "this model pair on this protocol.",
    )
    add_styled(
        doc,
        "MDPI_1.8_keywords",
        "Keywords: gene expression prediction; tissue-averaged expression; DNA language model; AgroNT; "
        "convolutional neural network; Arabidopsis thaliana; region importance; cross-species transfer",
    )
    add_styled(doc, "MDPI_1.9_line", "")

    # --- 1. Introduction ---
    add_styled(doc, "MDPI_2.1_heading1", "1. Introduction")
    add_styled(
        doc,
        "MDPI_3.1_text",
        "Most of a plant genome consists of non-coding DNA, and the cis-regulatory elements (CREs) "
        "embedded in promoters, 5′/3′ untranslated regions (UTRs), and terminators determine when, "
        "where, and at what level genes are expressed [1]. Models that read this sequence-to-expression "
        "code directly from DNA would enable genome-wide annotation of uncharacterized genes, "
        "prioritization of non-coding variants, and rational design of synthetic regulatory DNA [2]; the "
        "feasibility of such design has been demonstrated quantitatively in engineered "
        "sequence-to-expression systems [3].",
    )
    add_styled(
        doc,
        "MDPI_3.1_text",
        "Deep learning has matured into a standard modeling tool for regulatory genomics [4]. "
        "Convolutional neural networks (CNNs) in particular established that proximal DNA sequence alone "
        "carries substantial information about transcript abundance and regulatory activity: DeepSEA "
        "predicts the functional consequences of non-coding variants [5], Basset learns the regulatory "
        "code of chromatin accessibility [6], and DeepBind popularized attribution-driven in-silico "
        "mutagenesis for interpreting learned specificities [7]. In plants, this line of work now spans "
        "high-versus-low expression classification from flanking sequence across multiple species [8], "
        "cross-clade mRNA-abundance prediction together with regulatory-sequence design [9], and "
        "transformer–CNN hybrids for cis-regulatory element discovery [10].",
    )
    add_styled(
        doc,
        "MDPI_3.1_text",
        "In parallel, DNA language models pretrained on whole genomes have emerged as general-purpose "
        "feature extractors for genomics. The Nucleotide Transformer showed robust transfer across "
        "diverse genomic tasks [11], Enformer demonstrated the value of long-range context for "
        "expression prediction [12], and genomic pre-trained networks attain state-of-the-art zero-shot "
        "variant-effect prediction even in species without functional-genomics data [13]. Plant-specific "
        "pretraining culminated in AgroNT, a one-billion-parameter transformer pretrained on ~10.5 "
        "million sequences from 48 edible plant genomes, which reported strong results across the Plants "
        "Genomic Benchmark (PGB), including the Arabidopsis gene-expression task considered here [14,15].",
    )
    add_styled(
        doc,
        "MDPI_3.1_text",
        "These two model families—compact supervised CNNs and pretrained foundation models—are both "
        "credible options for plant expression prediction, yet they have rarely been compared under "
        "locked conditions. For practitioners deciding what to train, the decisive questions are "
        "quantitative: how much accuracy does foundation-model fine-tuning add over this DeepSEA-style CNN when "
        "splits, labels, and metrics are held fixed; what does that gain cost in trainable parameters and "
        "training time; and which sequence compartments carry the predictive signal under tissue-averaged "
        "labels?",
    )
    add_styled(
        doc,
        "MDPI_3.1_text",
        "Here we answer these questions in *Arabidopsis thaliana*. Under identical data splits and "
        "evaluation protocols, we fine-tune (i) a DeepSEA-style 1D-CNN baseline and (ii) AgroNT with "
        "low-rank adaptation (LoRA) on the same derived PGB binary task, and pair the head-to-head "
        "comparison with matched region-importance analysis on a structured flanking representation, "
        "DeepLift/TF-MoDISco motif recovery on the CNN, natural-variant scoring, tissue-label concordance "
        "quantification, center-cropped input-length ablation of both models, and transfer of the Arabidopsis-trained "
        "CNN and AgroNT checkpoints to four crop species. The contribution is a locked-protocol comparison "
        "of this model pair under tissue-averaged abundance classes, together with an explicit account of "
        "what those labels can and cannot support.",
    )

    # --- 2. Methods ---
    add_styled(doc, "MDPI_2.1_heading1", "2. Materials and Methods")
    add_styled(doc, "MDPI_2.2_heading2", "2.1. Data")
    add_styled(
        doc,
        "MDPI_3.1_text",
        "We used the *Arabidopsis thaliana* gene-expression FASTA of the Plants Genomic Benchmark (PGB) "
        "[14,15]. Each gene is represented by a 6-kb genomic window constructed as 5 kb upstream plus "
        "1 kb downstream of the annotated transcription start site (TSS), matching the promoter-proximal "
        "input used for AgroNT gene-expression models [14]. Expression labels in the FASTA comprise 56 "
        "tissue/sample RNA-sequencing measurements. The official split "
        "comprises 25,731 training, 3,401 validation, and 3,402 test genes, with no gene overlap across "
        "splits. The official PGB gene-expression task is multi-tissue regression; here we derived binary "
        "high/low labels from the median of per-gene mean expression on the training set, yielding a "
        "balanced two-class problem. Because labels aggregate expression across "
        "56 tissues/samples, the benchmark targets sequence correlates of tissue-averaged steady-state "
        "abundance class (median-thresholded), not organ-specific regulatory programs.",
    )
    add_styled(
        doc,
        "MDPI_3.1_text",
        "In parallel, we built a structured gene-centric dataset for region-level interpretability. For "
        "each of 27,436 protein-coding *A. thaliana* genes with usable flanking annotation and expression "
        "(TAIR10/Araport11 annotation [16,17]; Ensembl Plants release 59 [18]), we extracted the promoter "
        "(1 kb upstream of the transcription start site), 5′UTR (500 nt), 3′UTR (500 nt), and terminator "
        "(1 kb downstream of the transcription termination site) in transcriptional orientation, "
        "concatenated the four regions 5′→3′ with a 20-nt N-padding gap, and one-hot encoded the "
        "resulting 3,020-bp sequence. Genes were assigned to fixed chromosomal splits (train: chromosome "
        "5, n=6,331; validation: chromosomes 3 and 4, n=9,636; test: chromosomes 1 and 2, n=11,469) to "
        "reduce sequence-homology leakage. Binary labels used the training-split median of log₂(mean "
        "expression+1) (threshold 3.038), matching the PGB train-median protocol and avoiding threshold "
        "leakage from held-out genes.",
    )
    add_styled(doc, "MDPI_2.2_heading2", "2.2. Model Architectures")
    add_styled(
        doc,
        "MDPI_3.1_text",
        "1D-CNN baseline. We implemented a DeepSEA-style convolutional network [5] with three "
        "convolutional blocks, each containing two 1D convolutions (256 filters, kernel size 8), batch "
        "normalization, ReLU activation, and max pooling (pool size 4), followed by global max pooling "
        "and two fully connected layers with dropout. The model has 2.7 M parameters and accepts one-hot "
        "DNA of arbitrary length; global max pooling makes the classification head independent of input length.",
    )
    add_styled(
        doc,
        "MDPI_3.1_text",
        "DNA foundation model. AgroNT-1B [14] is a transformer of 985 M parameters (40 layers, hidden "
        "size 1500) pretrained on ~10.5 million sequences from 48 edible plant genomes with a 6-mer "
        "tokenizer and a 1024-token context. We adapted it for sequence classification with low-rank "
        "adaptation (LoRA) adapters [19] (rank 16, α=32) on the query/key/value/output projections of "
        "every self-attention block, plus a two-layer head over mean-pooled hidden states. Only the 16.2 M "
        "LoRA and head parameters (1.6% of the total) were trainable; gradient checkpointing kept peak "
        "GPU memory within 12 GB.",
    )
    add_styled(doc, "MDPI_2.2_heading2", "2.3. Training and Evaluation")
    add_styled(
        doc,
        "MDPI_3.1_text",
        "Both models were trained with cross-entropy loss and early stopping on validation AUROC "
        "(patience of 2–5 epochs). The CNN used batch size 64 and AdamW (learning rate 10⁻³, weight "
        "decay 10⁻⁴) for up to 30 epochs. AgroNT used batch size 4 with gradient accumulation over 8 "
        "steps (effective batch size 32) and AdamW (learning rate 2×10⁻⁴) for up to 3 epochs. All "
        "training used a single NVIDIA RTX 4090 (24 GB). Reported metrics were accuracy, AUROC, and "
        "AUPRC. For the PGB CNN, we repeated training with three random seeds (42/43/44) and report the "
        "mean±standard deviation of test AUROC/accuracy alongside the primary seed-42 checkpoint used in "
        "the AgroNT head-to-head comparison. The reported AgroNT checkpoint is a single seed (42).",
    )
    add_styled(doc, "MDPI_2.2_heading2", "2.4. Statistical Analysis")
    add_styled(
        doc,
        "MDPI_3.1_text",
        "We assessed the CNN–AgroNT difference with 500-fold bootstrap resampling of the test set to "
        "obtain 95% confidence intervals for each AUROC and for the AUROC difference; intervals excluding "
        "zero were treated as statistically supported. We additionally applied McNemar's test to paired "
        "correct/incorrect classifications at probability threshold 0.5. An input-length ablation "
        "retrained both the CNN and AgroNT after midpoint (center) cropping of the published 6 kb "
        "TSS-anchored windows to "
        "3 kb and 1.5 kb. Because that window is 5 kb upstream plus 1 kb downstream of the TSS, its "
        "midpoint lies ~2 kb upstream of the TSS; a 3 kb midpoint crop therefore excludes most of the "
        "TSS-proximal 1 kb of gene body. "
        "Performance of the structured ~3 kb flanking representation was evaluated on its held-out "
        "chromosomal test split; absolute AUROC is not compared head-to-head with PGB truncation numbers "
        "because split and sequence construction differ.",
    )
    add_styled(doc, "MDPI_2.2_heading2", "2.5. Interpretability")
    add_styled(
        doc,
        "MDPI_3.1_text",
        "Matched region importance used the same in-silico sliding-window mutagenesis protocol on "
        "structured flanking sequences (50-bp windows of N substitution, 25-bp step; n=200 CNN test genes "
        "and n=100 AgroNT sequences): for each window we recorded the absolute change in predicted "
        "high-expression probability and aggregated mean |Δ| per base within promoter, 5′UTR, 3′UTR, and "
        "terminator (excluding the artificial gap). The CNN for this analysis was retrained on the "
        "concatenated flanking set (test AUROC 0.805). AgroNT mutagenesis applied the Arabidopsis PGB LoRA "
        "checkpoint to the same concatenated strings without retraining on that representation, so the "
        "two maps share a scoring protocol but not a matched training regime. Separately, DeepLift "
        "attributions [20] (captum) on structured test genes supported motif discovery with "
        "TF-MoDISco-lite [21] and TOMTOM annotation against the JASPAR 2024 core collection [22]; "
        "architecture comparisons of region ranks use the matched mutagenesis scores rather than DeepLift "
        "magnitudes.",
    )
    add_styled(doc, "MDPI_2.2_heading2", "2.6. Tissue Heterogeneity of Expression Labels")
    add_styled(
        doc,
        "MDPI_3.1_text",
        "To characterize the biological scope of tissue-averaged labels, we parsed the 56 tissue/sample "
        "expression values encoded in PGB FASTA headers for 32,534 unique Arabidopsis genes. We computed "
        "pairwise Spearman correlations among tissues, the per-gene cross-tissue coefficient of variation "
        "(CV), and the agreement between the tissue-averaged high/low label (thresholded at the "
        "training-set median of gene means) and each tissue's own median-based high/low label.",
    )
    add_styled(doc, "MDPI_2.2_heading2", "2.7. Cross-Species Transfer")
    add_styled(
        doc,
        "MDPI_3.1_text",
        "The Arabidopsis-trained CNN and the Arabidopsis AgroNT LoRA checkpoint, both with frozen "
        "weights, were applied to the PGB gene-expression test sets of rice (*Oryza sativa*), maize "
        "(*Zea mays*), tomato (*Solanum lycopersicum*), and soybean (*Glycine max*). For each species, "
        "binary labels used that species' training-split median expression, then metrics were computed "
        "on the test split—matching the Arabidopsis label protocol. AgroNT was pretrained on ~10.5 "
        "million sequences from 48 edible plant genomes that include these crops [14]; the crop "
        "evaluation is therefore application of an Arabidopsis task adapter on a multi-species backbone, "
        "not species-unseen transfer in the same sense as the CNN.",
    )
    add_styled(doc, "MDPI_2.2_heading2", "2.8. Variant Effect Prediction")
    add_styled(
        doc,
        "MDPI_3.1_text",
        "Natural single-nucleotide variants from the Ensembl Plants *A. thaliana* variation VCF [18] were "
        "mapped onto the structured flanking representation (promoter, 5′/3′ UTR, terminator) of 250 "
        "genes, excluding the artificial N-padding gap. The CNN scored 27,662 variants by substituting "
        "the mapped alternate allele and recording the absolute change in expression probability (|Δ|), "
        "following mutation-map practice introduced for regulatory models [7]. The Arabidopsis AgroNT LoRA "
        "checkpoint was applied to the same sites as the mean |Δ| over the three non-reference bases at "
        "each position (26,518 non-gap sites after dropping genes absent from the structured set). Region "
        "ranks are compared within each model; absolute |Δ| is not compared across architectures because "
        "the allele protocol differs. We compared |Δ| across flanking regions with Mann–Whitney U tests. "
        "As an exploratory check, 17 variants overlapping reported eQTL associations were contrasted with "
        "the CNN genome-wide scored background; AgroNT eQTL scores used exact alleles and are not "
        "contrasted with the three-alt site-mean background.",
    )

    # --- 3. Results ---
    add_styled(doc, "MDPI_2.1_heading1", "3. Results")
    add_styled(doc, "MDPI_2.2_heading2", "3.1. AgroNT Outperforms the CNN on a Unified Arabidopsis Benchmark")
    add_styled(
        doc,
        "MDPI_3.1_text",
        "On the held-out PGB test set (n = 3,402), AgroNT reached 86.8% accuracy (AUROC 0.939, AUPRC "
        "0.935), compared with 75.4% accuracy (AUROC 0.842, AUPRC 0.835) for the CNN (Table 1; Figure 1). "
        "The AUROC difference of 0.097 had a bootstrap 95% CI of 0.086–0.108. McNemar's test on paired "
        "classifications was highly significant (p = 6.7×10⁻⁵⁴): 532 genes were classified correctly by "
        "AgroNT and incorrectly by the CNN, versus 142 the reverse (net +390). AgroNT LoRA was trained "
        "for up to three epochs with early stopping on "
        "validation AUROC; the retained checkpoint is the epoch-1 model (validation AUROC 0.937). Across "
        "three CNN seeds, test AUROC was 0.844±0.003 and accuracy 76.3±0.9% (seeds 42/43/44), so the "
        "primary CNN result is not a single-run outlier. These results show that, under identical "
        "splits/metrics and the reported training budgets (CNN ≤30 epochs; AgroNT LoRA ≤3 epochs), "
        "plant-genome pretraining plus LoRA fine-tuning yields a large and statistically supported gain "
        "over this DeepSEA-style CNN configuration.",
    )
    add_table(
        doc,
        ["Model", "Trainable Params", "Accuracy", "AUROC", "AUPRC"],
        [
            ["1D-CNN (DeepSEA-style)", "2.7 M", "75.4%", "0.842", "0.835"],
            ["AgroNT-1B (LoRA)", "16.2 M (of 985 M)", "86.8%", "0.939", "0.935"],
        ],
        "Table 1. Unified benchmark of the 1D-CNN and AgroNT on the PGB Arabidopsis held-out test set "
        "(primary seed 42). Bootstrap 95% CIs for AUROC: CNN 0.829–0.855; AgroNT 0.931–0.947.",
    )
    add_figure(
        doc,
        FIG / "fig3_benchmark.png",
        "Figure 1. Head-to-head comparison of the 1D-CNN baseline and AgroNT on the PGB Arabidopsis "
        "test set under locked splits, labels, and budgets. (a) Metric-wise gain from CNN to AgroNT "
        "for Accuracy, AUROC, and AUPRC shown as a dumbbell plot (exact values in Table 1). (b) AUROC "
        "versus trainable-parameter count (log scale), annotated with single-GPU training time. AgroNT "
        "is plotted at 16.2 M trainable parameters (985 M total).",
    )

    add_styled(doc, "MDPI_2.2_heading2", "3.2. Tissue-Averaged Labels Capture a Shared Abundance Axis")
    add_styled(
        doc,
        "MDPI_3.1_text",
        "Across 32,534 genes and 56 tissues/samples, pairwise tissue–tissue Spearman correlations were "
        "high (median 0.851; IQR 0.803–0.898), indicating substantial shared structure among expression "
        "profiles (Figure 2). Single-tissue high/low labels agreed with the tissue-averaged label for a "
        "mean of 92.1% of genes (range 78.7–95.4%). The median cross-tissue CV was 0.340, so "
        "tissue-dependent dispersion remains. High concordance is expected when tissues are strongly "
        "correlated; we treat it as support for using an averaged binary label in this benchmark, not as "
        "proof of a biological “expression potential.”",
    )
    add_figure(
        doc,
        FIG / "fig7_tissue_heterogeneity.png",
        "Figure 2. Tissue heterogeneity of PGB Arabidopsis expression labels across 32,534 genes and 56 "
        "tissues/samples. (a) Distribution of the per-gene cross-tissue coefficient of variation (CV); "
        "the red dashed line marks the 75th percentile (Q75 = 1.37), above which gene expression is "
        "strongly tissue-dependent. (b) Concordance of each tissue's own median-based high/low label with "
        "the tissue-averaged label, sorted ascending; the red dashed line marks the mean agreement "
        "(0.921). (c) Pairwise Spearman correlation among the 56 tissues/samples.",
        width_in=6.2,
    )

    add_styled(
        doc,
        "MDPI_2.2_heading2",
        "3.3. Proximal Flanks Carry Predictive Signal on the Structured Representation",
    )
    add_styled(
        doc,
        "MDPI_3.1_text",
        "On the rebuilt structured flanking set (train-median threshold; chromosomal split), the CNN "
        "reached test AUROC 0.805 (accuracy 69.6%, AUPRC 0.770). Matched sliding-window mutagenesis "
        "ranked mean importance per base as 5′UTR (0.073) > terminator (0.066) > promoter (0.060) > "
        "3′UTR (0.057) for the CNN and promoter (0.558) > 3′UTR (0.545) > 5′UTR (0.501) > terminator "
        "(0.483) for AgroNT (Table 2; Figure 3). Absolute mutagenesis scales are not comparable across "
        "architectures; within each model, proximal annotated flanks dominate the artificial gap, and "
        "AgroNT scores are relatively even across promoter/UTRs. DeepLift attributions used for motif "
        "discovery ranked 5′UTR (0.0088) > 3′UTR (0.0081) > promoter (0.0048) > terminator (0.0038), "
        "consistent in direction with prior reports that UTR sequence contributes heavily to plant "
        "expression prediction [8,9]. TF-MoDISco recovered four positive and four negative "
        "expression-associated patterns; TOMTOM against JASPAR 2024 mapped hits that include DOF-family "
        "factors (DOF1.5, DOF3.4/3.6, DOF5.1/5.8, CDF5), SPL8, NAC005, and AGL1. Because short DOF-like "
        "cores also appear among negative patterns, we treat motif annotation as provisional rather than "
        "as discovery of expression-driving CREs.",
    )
    add_table(
        doc,
        ["Region", "CNN", "AgroNT"],
        [
            ["Promoter", "0.060", "0.558"],
            ["5′UTR", "0.073", "0.501"],
            ["3′UTR", "0.057", "0.545"],
            ["Terminator", "0.066", "0.483"],
        ],
        "Table 2. Matched sliding-window mutagenesis importance per base on the structured flanking representation (gap excluded).",
    )
    add_figure(
        doc,
        FIG / "fig4_region_importance.png",
        "Figure 3. Region-level interpretability on the structured flanking representation. (a) Mean "
        "absolute DeepLift attribution per base by region for the CNN, the model used for "
        "TF-MoDISco/TOMTOM motif discovery. (b) Matched sliding-window mutagenesis importance per base "
        "(gap excluded) for both models, normalized to each model's maximum; raw per-base values are "
        "shown beside each point (exact values in Table 2).",
        width_in=6.0,
    )

    add_styled(doc, "MDPI_2.2_heading2", "3.4. Predicted Variant Effects Are Region-Structured")
    add_styled(
        doc,
        "MDPI_3.1_text",
        "On 250 genes of the structured flanking set, CNN scoring of 27,662 natural flanking-region SNVs "
        "mapped median |Δ| as 5′UTR (0.00269; n=2,542) > 3′UTR (0.00112; n=4,148) > terminator (0.00058; "
        "n=13,573) > promoter (0.00043; n=7,399). Pooled UTR variants exceeded terminator variants "
        "(Mann–Whitney p = 4.3×10⁻⁹⁸) and promoter variants (p = 1.4×10⁻¹⁰⁶); 5′UTR exceeded 3′UTR "
        "(p = 1.2×10⁻⁶¹). AgroNT median |Δ| on 26,518 overlapping non-gap sites ranked terminator (0.00202; "
        "n=12,958) > 3′UTR (0.00118; n=3,997) > 5′UTR (0.00085; n=2,474) > promoter (0.00054; n=7,089); "
        "terminator exceeded 3′UTR (p = 3.4×10⁻¹²²) and 5′UTR (p = 1.9×10⁻²²¹) (Table 3; Figure 4). "
        "Promoter was the least sensitive compartment in both variant maps. Absolute median |Δ| values "
        "remain small (probability units) and are not commensurate across the exact-allele CNN map and "
        "the three-alt AgroNT mean, so p-values are read as within-model regional sensitivity, not "
        "validated regulatory effect sizes. CNN promoter-versus-terminator order disagrees with CNN "
        "DeepLift; AgroNT terminator-led |Δ| also does not match AgroNT mutagenesis (promoter-led). We "
        "therefore claim only the shared promoter-lowest pattern on the variant maps plus "
        "architecture-specific UTR/terminator ranks. An exploratory overlap of 17 reported eQTL variants "
        "was higher than the CNN background (0.00205 vs. 0.00065; Mann–Whitney p = 0.041). Exact-allele "
        "AgroNT |Δ| on the same 17 sites (median 0.00082) was not higher than the AgroNT three-alt "
        "site-mean background (median 0.00113; one-sided Mann–Whitney p = 0.95); because the allele "
        "protocols differ, this is a descriptive check rather than a matched enrichment test.",
    )
    add_table(
        doc,
        ["Region", "n (CNN)", "CNN median", "n (AgroNT)", "AgroNT median"],
        [
            ["5′UTR", "2,542", "0.00269", "2,474", "0.00085"],
            ["3′UTR", "4,148", "0.00112", "3,997", "0.00118"],
            ["Terminator", "13,573", "0.00058", "12,958", "0.00202"],
            ["Promoter", "7,399", "0.00043", "7,089", "0.00054"],
        ],
        "Table 3. Predicted |Δ| in high-expression probability by flanking region, on 250 genes. CNN: mapped alternate allele. AgroNT: mean over three non-reference bases.",
    )
    add_figure(
        doc,
        FIG / "fig5b_variant_by_region.png",
        "Figure 4. Distribution of predicted |Δ| in high-expression probability by flanking region "
        "(boxplots omit outliers; Mann–Whitney tests in the main text). (a) CNN, mapped alternate "
        "allele. (b) AgroNT, mean |Δ| over the three non-reference bases at the same sites.",
        width_in=6.2,
    )

    add_styled(doc, "MDPI_2.2_heading2", "3.5. Cross-Species Transfer")
    add_styled(
        doc,
        "MDPI_3.1_text",
        "Without further fine-tuning, the Arabidopsis-trained CNN predicted high-versus-low expression "
        "with AUROC 0.716 (rice; n=3,702), 0.685 (maize; n=4,483), 0.763 (tomato; n=3,827), and 0.759 "
        "(soybean; n=4,803) under each species' training-set median labels. The same protocol applied to "
        "the Arabidopsis AgroNT LoRA checkpoint yielded AUROC 0.815, 0.894, 0.838, and 0.878 (Figure 5; "
        "Table 4). AgroNT exceeded the CNN on every crop; the largest gap was maize (+0.209 AUROC), "
        "where CNN transfer was weakest. CNN dicots still exceeded CNN monocots, whereas AgroNT was "
        "comparatively even across clades, consistent with pretraining already covering these genomes. "
        "Crop AUROC remained below Arabidopsis self-performance (CNN 0.842; AgroNT 0.939), so "
        "species-specific fine-tuning would still be expected to help. These scores support transferable "
        "sequence correlates of tissue-averaged abundance class; they do not establish a conserved "
        "causal cis-regulatory signature across the monocot–dicot divide.",
    )
    add_table(
        doc,
        ["Species", "n_test", "CNN Acc.", "CNN AUROC", "CNN AUPRC", "AgroNT Acc.", "AgroNT AUROC", "AgroNT AUPRC"],
        [
            ["Rice", "3,702", "0.647", "0.716", "0.699", "0.718", "0.815", "0.821"],
            ["Maize", "4,483", "0.564", "0.685", "0.698", "0.726", "0.894", "0.907"],
            ["Tomato", "3,827", "0.662", "0.763", "0.787", "0.695", "0.838", "0.858"],
            ["Soybean", "4,803", "0.655", "0.759", "0.762", "0.774", "0.878", "0.884"],
        ],
        "Table 4. Transfer of Arabidopsis-trained checkpoints to PGB crop test sets (species training-split median labels; no crop fine-tuning).",
    )
    add_figure(
        doc,
        FIG / "fig6_cross_species.png",
        "Figure 5. Transfer of Arabidopsis-trained checkpoints to rice, maize, tomato, and soybean "
        "under species training-set median labels. (a) CNN; (b) AgroNT LoRA. Cells are metric×species "
        "scores grouped by clade (exact values in Table 4). AgroNT was pretrained on genomes that "
        "include these crops, so panel (b) is adapter transfer on a multi-species backbone rather than "
        "species-unseen evaluation.",
        width_in=6.2,
    )

    add_styled(doc, "MDPI_2.2_heading2", "3.6. Input-Length Ablation")
    add_styled(
        doc,
        "MDPI_3.1_text",
        "Retraining after midpoint-cropping the published 6 kb windows to 3 kb or 1.5 kb reduced CNN "
        "test AUROC to 0.542 and 0.545 (versus 0.842 at full 6 kb) and AgroNT test AUROC to 0.625 and "
        "0.580 (versus 0.939; Table 5). Under the TSS-anchored 5 kb+1 kb "
        "geometry, midpoint cropping removes most TSS-proximal sequence, so near-chance AUROC is the "
        "expected outcome of that shortening rather than a test of gene-body centering. AgroNT remained "
        "slightly above the CNN at both truncated lengths, but both collapses show that the 6 kb gain is "
        "tied to this window geometry for both architectures. The structured "
        "3,020-bp flanking representation, which keeps complete annotated proximal flanks, reached CNN "
        "test AUROC 0.805 on its chromosomal split. Absolute AUROC from these two constructions is not "
        "pooled into one length narrative because split and sequence layout differ.",
    )
    add_table(
        doc,
        ["Length", "CNN Acc.", "CNN AUROC", "CNN AUPRC", "AgroNT Acc.", "AgroNT AUROC", "AgroNT AUPRC"],
        [
            ["6 kb", "0.754", "0.842", "0.835", "0.868", "0.939", "0.935"],
            ["3 kb center", "0.521", "0.542", "0.547", "0.593", "0.625", "0.611"],
            ["1.5 kb center", "0.538", "0.545", "0.542", "0.553", "0.580", "0.578"],
        ],
        "Table 5. Test metrics after midpoint-cropping the published 6 kb TSS-anchored PGB window. Both models were retrained at each length (seed 42).",
    )

    # --- 4–5 ---
    add_styled(doc, "MDPI_2.1_heading1", "4. Discussion")
    add_styled(
        doc,
        "MDPI_3.1_text",
        "Architecture choice under a shared plant benchmark. Head-to-head evaluation on this derived PGB "
        "Arabidopsis binary task isolates the gain of LoRA-tuned AgroNT over this DeepSEA-style CNN "
        "(+0.097 AUROC; bootstrap 95% CI 0.086–0.108; McNemar p = 6.7×10⁻⁵⁴; net +390 discordant pairs) "
        "under locked splits, metrics, and reported budgets. "
        "The direction and rough magnitude agree with AgroNT's task-level advantage across the full PGB "
        "suite [14], but the present design adds paired statistics, multi-seed CNN replication "
        "(AUROC 0.844±0.003), and an explicit cost account: LoRA confines training to 1.6% of AgroNT "
        "parameters within 12 GB of GPU memory. For genome-scale annotation pipelines where a ~0.1 AUROC "
        "improvement on this binary task is material, fine-tuned AgroNT is preferable on comparable "
        "hardware; where compute is limiting, the 2.7 M-parameter CNN remains a competitive compact "
        "baseline. We do not claim this CNN is optimally tuned relative to plant-specific CNNs that "
        "exceed 80% accuracy [8]; the comparison quantifies the DeepSEA-style configuration used here.",
    )
    add_styled(
        doc,
        "MDPI_3.1_text",
        "What the models learn (predictive associations). On the structured representation, matched "
        "mutagenesis and DeepLift both place predictive importance in proximal annotated flanks, with "
        "CNN DeepLift and CNN mutagenesis agreeing that the 5′UTR is among the strongest "
        "compartments—consistent in direction with prior reports that UTR sequence contributes heavily "
        "to plant expression prediction [8,9]. This is a predictive association with tissue-averaged "
        "mRNA-abundance class: UTR sequence may capture transcriptional and/or post-transcriptional "
        "correlates, and the present labels cannot separate those mechanisms. TOMTOM hits including "
        "DOF-, SPL-, and NAC-family motifs are provisional, since short DOF-like cores also appear among "
        "negative patterns. Region-resolved variant |Δ| ranks UTR > terminator > promoter for this CNN and "
        "terminator > UTR > promoter for AgroNT; promoter is the shared least-sensitive compartment on "
        "those maps. Attribution, mutagenesis, and variant maps are not required to agree beyond that, "
        "and none is a causal CRE map.",
    )
    add_styled(
        doc,
        "MDPI_3.1_text",
        "Tissue-averaged scope. High tissue–tissue correlation (median Spearman 0.851) and high "
        "concordance between single-tissue and averaged high/low labels (mean 92.1%) justify the averaged "
        "binary benchmark for architecture comparison. Residual disagreement (agreement as low as 78.7%) "
        "marks the boundary of that justification: organ-specific programs remain better addressed with "
        "tissue-conditioned labels and distal or epigenomic context [8], for which large Arabidopsis "
        "regulatory atlases now provide suitable features [23].",
    )
    add_styled(
        doc,
        "MDPI_3.1_text",
        "Cross-species transfer. CNN AUROC of 0.685–0.763 and AgroNT AUROC of 0.815–0.894 across four "
        "crops, under each species' training-set median labels, shows above-chance task transfer. AgroNT "
        "is higher on every crop, with the largest gain in maize, but the two evaluations are not "
        "interchangeable: the CNN never trained on crop genomes, whereas AgroNT's backbone already saw "
        "those species during pretraining. Transfer remains below within-species Arabidopsis "
        "performance, so crop-specific fine-tuning would still be expected to improve accuracy.",
    )
    add_styled(
        doc,
        "MDPI_3.1_text",
        "Input length. Midpoint cropping of the published 6 kb window to 3 kb or 1.5 kb drops CNN AUROC "
        "to 0.542–0.545 and AgroNT AUROC to 0.625–0.580. The same geometry therefore limits both "
        "architectures; the 6 kb gain is not a CNN-only artifact of local convolution.",
    )
    add_styled(
        doc,
        "MDPI_3.1_text",
        "Limitations. AgroNT was fine-tuned for few epochs, reported from a single seed (42), and compared "
        "under asymmetric budgets; DeepLift motif discovery was performed on the CNN, while "
        "architecture-level region ranks use matched mutagenesis (absolute scales remain "
        "non-commensurate across models); AgroNT mutagenesis applied the PGB LoRA checkpoint to "
        "concatenated flanks without retraining on that set; variant |Δ| used mapped alleles for the CNN "
        "and three-alt means for AgroNT on 250 genes; eQTL overlap was exploratory (n=17), with CNN "
        "scores exceeding background (p = 0.041) and AgroNT exact-allele scores not exceeding a "
        "three-alt background under unmatched allele protocols; structured UTR lengths are fixed at "
        "500 nt; and primary training was confined to one model species. Extending the "
        "benchmark to tissue-of-expression prediction, integrating accessibility or methylation features "
        "[23,24], expanding eQTL validation, and evaluating longer-context and zero-shot DNA language "
        "models [11–13] are natural next steps.",
    )

    add_styled(doc, "MDPI_2.1_heading1", "5. Conclusions")
    add_styled(
        doc,
        "MDPI_3.1_text",
        "On a derived Arabidopsis PGB task with tissue-averaged labels, AgroNT fine-tuned with LoRA "
        "outperforms this DeepSEA-style 1D-CNN by a large and statistically supported margin under the "
        "reported budgets, while multi-seed CNN replicates confirm a stable compact baseline for "
        "compute-limited settings. Predictive importance on a structured flanking representation "
        "concentrates in proximal annotated flanks. CNN variant |Δ| is led by the 5′UTR and AgroNT "
        "variant |Δ| by the terminator; both variant maps leave the promoter lowest. Motif hits remain "
        "non-causal. Tissue-averaged labels align with "
        "most single-tissue binary axes yet leave a measurable organ-specific residue. Midpoint cropping "
        "of the 6 kb window collapses both models toward chance. Both Arabidopsis "
        "checkpoints transfer to rice, maize, tomato, and soybean without crop fine-tuning at "
        "above-chance AUROC, with AgroNT remaining ahead on every crop. Together these results "
        "quantify this CNN–AgroNT pair under a locked protocol and tissue-averaged labels.",
    )

    # --- Back matter ---
    add_back(
        doc,
        "Author Contributions",
        "Conceptualization, F.L. and S.L.; methodology, F.L.; software, F.L.; validation, F.L.; formal "
        "analysis, F.L.; writing—original draft preparation, F.L.; writing—review and editing, S.L.; All "
        "authors have read and agreed to the published version of the manuscript.",
    )
    add_back(doc, "Funding", "This research received no external funding.")
    add_back(doc, "Institutional Review Board Statement", "Not applicable.")
    add_back(doc, "Informed Consent Statement", "Not applicable.")
    add_back(
        doc,
        "Data Availability Statement",
        "All primary data are public: PGB [15], TAIR10/Araport11, Ensembl Plants variation, and JASPAR. "
        "Analysis code, configs, and trained checkpoints (CNN weights; AgroNT LoRA adapters) are available "
        "from the corresponding author.",
    )
    add_back(doc, "Conflicts of Interest", "The authors declare no conflicts of interest.")

    add_styled(doc, "MDPI_2.1_heading1", "Abbreviations")
    add_styled(
        doc,
        "MDPI_3.2_text_no_indent",
        "The following abbreviations are used in this manuscript:",
    )
    abbr_rows = [
        ("AUROC", "Area under the receiver operating characteristic curve"),
        ("AUPRC", "Area under the precision–recall curve"),
        ("CNN", "Convolutional neural network"),
        ("CRE", "Cis-regulatory element"),
        ("CV", "Coefficient of variation"),
        ("eQTL", "Expression quantitative trait locus"),
        ("LoRA", "Low-rank adaptation"),
        ("PGB", "Plants Genomic Benchmark"),
        ("UTR", "Untranslated region"),
    ]
    abbr = doc.add_table(rows=len(abbr_rows), cols=2)
    try:
        abbr.style = "MDPI_table"
    except KeyError:
        abbr.style = "Table Grid"
    for i, (a, b) in enumerate(abbr_rows):
        abbr.rows[i].cells[0].text = a
        abbr.rows[i].cells[1].text = b

    add_styled(doc, "MDPI_2.1_heading1", "References")
    refs = [
        "1. Marand, A.P.; Eveland, A.L.; Kaufmann, K.; Springer, N.M. cis-Regulatory Elements in Plant Development, Adaptation, and Evolution. Annu. Rev. Plant Biol. 2023, 74, 111–137. https://doi.org/10.1146/annurev-arplant-070122-030236",
        "2. Hu, X.; Fernie, A.R.; Yan, J. Deep learning in regulatory genomics: from identification to design. Curr. Opin. Biotechnol. 2023, 79, 102887. https://doi.org/10.1016/j.copbio.2022.102887",
        "3. Vaishnav, E.D.; de Boer, C.G.; Molinet, J.; et al. The evolution, evolvability and engineering of gene regulatory DNA. Nature 2022, 603, 455–463. https://doi.org/10.1038/s41586-022-04506-6",
        "4. Eraslan, G.; Avsec, Ž.; Gagneur, J.; Theis, F.J. Deep learning: new computational modelling techniques for genomics. Nat. Rev. Genet. 2019, 20, 389–403. https://doi.org/10.1038/s41576-019-0122-6",
        "5. Zhou, J.; Troyanskaya, O.G. Predicting effects of noncoding variants with deep learning–based sequence model. Nat. Methods 2015, 12, 931–934. https://doi.org/10.1038/nmeth.3547",
        "6. Kelley, D.R.; Snoek, J.; Rinn, J.L. Basset: Learning the regulatory code of the accessible genome with deep convolutional neural networks. Genome Res. 2016, 26, 990–999. https://doi.org/10.1101/gr.200535.115",
        "7. Alipanahi, B.; Delong, A.; Weirauch, M.T.; Frey, B.J. Predicting the sequence specificities of DNA- and RNA-binding proteins by deep learning. Nat. Biotechnol. 2015, 33, 831–838. https://doi.org/10.1038/nbt.3300",
        "8. Peleke, F.F.; Zumkeller, S.M.; Gültas, M.; Schmitt, A.; Szymański, J. Deep learning the cis-regulatory code for gene expression in selected model plants. Nat. Commun. 2024, 15, 3488. https://doi.org/10.1038/s41467-024-47744-0",
        "9. Li, T.H.; Xu, H.; Teng, S.; et al. Modeling 0.6 million genes for the rational design of functional cis-regulatory variants and de novo design of cis-regulatory sequences. Proc. Natl. Acad. Sci. USA 2024, 121, e2319811121. https://doi.org/10.1073/pnas.2319811121",
        "10. Wu, Y.; Huang, J.; Ming, L.; Deng, P.; Wang, M.; Zhang, Z. DeepPlantCRE: A Transformer-CNN Hybrid Framework for Plant Gene Expression Modeling and Cross-Species Generalization. arXiv 2025, arXiv:2505.09883. https://doi.org/10.48550/arXiv.2505.09883",
        "11. Dalla-Torre, H.; Gonzalez, L.; Mendoza-Revilla, J.; et al. Nucleotide Transformer: Building and evaluating robust foundation models for human genomics. Nat. Methods 2025, 22, 287–297. https://doi.org/10.1038/s41592-024-02523-z",
        "12. Avsec, Ž.; Agarwal, V.; Visentin, D.; et al. Effective gene expression prediction from sequence by integrating long-range interactions. Nat. Methods 2021, 18, 1196–1203. https://doi.org/10.1038/s41592-021-01252-x",
        "13. Benegas, G.; Batra, S.S.; Song, Y.S. DNA language models are powerful predictors of genome-wide variant effects. Proc. Natl. Acad. Sci. USA 2023, 120, e2311219120. https://doi.org/10.1073/pnas.2311219120",
        "14. Mendoza-Revilla, J.; Trop, E.; Gonzalez, L.; et al. A foundational large language model for edible plant genomes. Commun. Biol. 2024, 7, 835. https://doi.org/10.1038/s42003-024-06465-2",
        "15. InstaDeepAI. Plants Genomic Benchmark (PGB). 2024. https://doi.org/10.57967/hf/2464",
        "16. Arabidopsis Genome Initiative. Analysis of the genome sequence of the flowering plant Arabidopsis thaliana. Nature 2000, 408, 796–815. https://doi.org/10.1038/35048692",
        "17. Lamesch, P.; Berardini, T.Z.; Li, D.; et al. The Arabidopsis Information Resource (TAIR): Improved gene annotation and new tools. Nucleic Acids Res. 2012, 40, D1202–D1210. https://doi.org/10.1093/nar/gkr1090",
        "18. Harrison, P.W.; Amode, M.R.; Austine-Orimoloye, O.; et al. Ensembl 2024. Nucleic Acids Res. 2024, 52, D891–D899. https://doi.org/10.1093/nar/gkad1049",
        "19. Hu, E.J.; Shen, Y.; Wallis, P.; et al. LoRA: Low-Rank Adaptation of Large Language Models. arXiv 2021, arXiv:2106.09685. https://doi.org/10.48550/arXiv.2106.09685",
        "20. Shrikumar, A.; Greenside, P.; Kundaje, A. Learning Important Features Through Propagating Activation Differences. In Proceedings of the 34th International Conference on Machine Learning (ICML); 2017; Vol. 70, pp. 3145–3153.",
        "21. Shrikumar, A.; Tian, K.; Shcherbina, A.; et al. TF-MoDISco v0.4.4.2-alpha: Technical Note. arXiv 2018, arXiv:1811.00416. https://doi.org/10.48550/arXiv.1811.00416",
        "22. Rauluseviciute, I.; Audoux, J.; Kaur, H.; et al. JASPAR 2024: 20th anniversary of the open-access database of transcription factor binding profiles. Nucleic Acids Res. 2024, 52, D174–D182. https://doi.org/10.1093/nar/gkad1059",
        "23. O'Malley, R.C.; Huang, S.-s.C.; Song, L.; et al. Cistrome and Epicistrome Features Shape the Regulatory DNA Landscape. Cell 2016, 165, 1280–1292. https://doi.org/10.1016/j.cell.2016.04.038",
        "24. Takou, M.; Bellis, E.S.; Lasky, J.R. Predicting Gene Expression Responses to Cold in Arabidopsis thaliana Using Natural Variation in DNA Sequence. Genes 2025, 16, 1108. https://doi.org/10.3390/genes16091108",
    ]
    for ref in refs:
        add_styled(doc, "MDPI_8.1_references", ref)

    add_styled(
        doc,
        "MDPI_6.3_notes",
        "Disclaimer/Publisher’s Note: The statements, opinions and data contained in all publications "
        "are solely those of the individual author(s) and contributor(s) and not of MDPI and/or the "
        "editor(s). MDPI and/or the editor(s) disclaim responsibility for any injury to people or "
        "property resulting from any ideas, methods, instructions or products referred to in the content.",
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(OUT))
    print(f">>> wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
