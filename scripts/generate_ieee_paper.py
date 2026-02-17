#!/usr/bin/env python3
"""
Generate `ASD_Detection_IEEE_Paper.docx` using the provided IEEE Word template.

This script intentionally edits the DOCX at the XML level to preserve the template's
styles/section properties (especially the two-column IEEE body layout), while replacing
template guidance text with project content, figure/table placeholders, and IEEE-style
references.
"""

from __future__ import annotations

import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

from lxml import etree


REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DOCX = REPO_ROOT / "conference-template-letter.docx"
OUTPUT_DOCX = REPO_ROOT / "ASD_Detection_IEEE_Paper.docx"
WORK_DIR = REPO_ROOT / ".tmp_paper_build"


NS = {
    "w": "http://purl.oclc.org/ooxml/wordprocessingml/main",
    "r": "http://purl.oclc.org/ooxml/officeDocument/relationships",
}


@dataclass(frozen=True)
class AuthorBlock:
    name: str
    dept: str
    org: str
    loc: str
    email: str


def _p_style(p: etree._Element) -> str | None:
    pPr = p.find("w:pPr", namespaces=NS)
    if pPr is None:
        return None
    ps = pPr.find("w:pStyle", namespaces=NS)
    if ps is None:
        return None
    return ps.get(f"{{{NS['w']}}}val")


def _p_text(p: etree._Element) -> str:
    return "".join(p.xpath(".//w:t/text()", namespaces=NS))


def _clear_runs(p: etree._Element) -> None:
    for child in list(p):
        if child.tag in {f"{{{NS['w']}}}r", f"{{{NS['w']}}}hyperlink"}:
            p.remove(child)


def _set_paragraph_text(p: etree._Element, text: str) -> None:
    _clear_runs(p)
    r_el = etree.SubElement(p, f"{{{NS['w']}}}r")
    t_el = etree.SubElement(r_el, f"{{{NS['w']}}}t")
    if text.startswith(" ") or text.endswith(" ") or "  " in text:
        t_el.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    t_el.text = text


def _set_paragraph_lines(p: etree._Element, lines: list[str]) -> None:
    """Replace runs with a single run containing lines separated by <w:br/>."""
    _clear_runs(p)
    r = etree.SubElement(p, f"{{{NS['w']}}}r")
    for i, line in enumerate(lines):
        t = etree.SubElement(r, f"{{{NS['w']}}}t")
        if line.startswith(" ") or line.endswith(" ") or "  " in line:
            t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        t.text = line
        if i != len(lines) - 1:
            etree.SubElement(r, f"{{{NS['w']}}}br")


def _make_p(style_val: str, text: str) -> etree._Element:
    p = etree.Element(f"{{{NS['w']}}}p")
    pPr = etree.SubElement(p, f"{{{NS['w']}}}pPr")
    ps = etree.SubElement(pPr, f"{{{NS['w']}}}pStyle")
    ps.set(f"{{{NS['w']}}}val", style_val)
    r_el = etree.SubElement(p, f"{{{NS['w']}}}r")
    t_el = etree.SubElement(r_el, f"{{{NS['w']}}}t")
    if text.startswith(" ") or text.endswith(" ") or "  " in text:
        t_el.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    t_el.text = text
    return p


def _write_docx_from_dir(src_dir: Path, out_docx: Path) -> None:
    if out_docx.exists():
        out_docx.unlink()
    with zipfile.ZipFile(out_docx, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for file in src_dir.rglob("*"):
            if file.is_file():
                z.write(file, file.relative_to(src_dir).as_posix())


def main() -> None:
    if not TEMPLATE_DOCX.exists():
        raise FileNotFoundError(f"Missing template: {TEMPLATE_DOCX}")

    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    WORK_DIR.mkdir(parents=True)

    with zipfile.ZipFile(TEMPLATE_DOCX, "r") as z:
        z.extractall(WORK_DIR)

    doc_xml_path = WORK_DIR / "word" / "document.xml"
    root = etree.fromstring(doc_xml_path.read_bytes(), etree.XMLParser(remove_blank_text=False))
    body = root.find("w:body", namespaces=NS)
    if body is None:
        raise RuntimeError("Invalid template: missing w:body")

    paras = body.findall("w:p", namespaces=NS)

    # Locate key template placeholders by style.
    title_p = next(p for p in paras if _p_style(p) == "papertitle")
    abstract_p = next(p for p in paras if _p_style(p) == "Abstract")
    keywords_p = next(p for p in paras if _p_style(p) == "Keywords")

    # Replace author instruction text with placeholders (4 students + co-supervisor + supervisor).
    student_authors = [
        AuthorBlock(
            name="Student 1 Given Name Surname",
            dept="Department of Computer Science and Engineering",
            org="Your University",
            loc="City, Country",
            email="student1@university.edu",
        ),
        AuthorBlock(
            name="Student 2 Given Name Surname",
            dept="Department of Computer Science and Engineering",
            org="Your University",
            loc="City, Country",
            email="student2@university.edu",
        ),
        AuthorBlock(
            name="Student 3 Given Name Surname",
            dept="Department of Computer Science and Engineering",
            org="Your University",
            loc="City, Country",
            email="student3@university.edu",
        ),
        AuthorBlock(
            name="Student 4 Given Name Surname",
            dept="Department of Computer Science and Engineering",
            org="Your University",
            loc="City, Country",
            email="student4@university.edu",
        ),
    ]
    co_supervisor = AuthorBlock(
        name="Co-Supervisor Given Name Surname",
        dept="Department of Computer Science and Engineering",
        org="Your University",
        loc="City, Country",
        email="cosupervisor@university.edu",
    )
    supervisor = AuthorBlock(
        name="Supervisor Given Name Surname",
        dept="Department of Computer Science and Engineering",
        org="Your University",
        loc="City, Country",
        email="supervisor@university.edu",
    )

    author_paras = [p for p in paras if _p_style(p) == "Author"]
    for p in author_paras:
        t = _p_text(p)
        # The template sometimes packs multiple authors into one paragraph with line breaks.
        if "1st Given Name" in t or ("line 1" in t and "2nd Given Name" in t and "3rd Given Name" in t):
            lines: list[str] = []
            for a in student_authors:
                lines += [a.name, a.dept, a.org, a.loc, a.email]
            _set_paragraph_lines(p, lines)
        elif "5th Given Name" in t:
            _set_paragraph_lines(p, [co_supervisor.name, co_supervisor.dept, co_supervisor.org, co_supervisor.loc, co_supervisor.email])
        elif "6th Given Name" in t:
            _set_paragraph_lines(p, [supervisor.name, supervisor.dept, supervisor.org, supervisor.loc, supervisor.email])
        elif "Sub-titles are not captured" in t:
            _set_paragraph_text(p, "")

    # Title / abstract / keywords.
    paper_title = (
        "A Modular Multimodal ASD Detection System Using Pragmatic- Conversational and "
        "Acoustic- Prosodic Speech Features with Explainable Predictions"
    )
    _set_paragraph_text(title_p, paper_title)

    abstract_text = (
        "This paper presents a modular multimodal system for autism spectrum disorder (ASD) screening from speech, "
        "supporting TalkBank CHAT transcripts and raw audio. The system integrates (i) pragmatic and conversational "
        "feature engineering aligned with turn taking, topic maintenance, pause and latency, and conversational repair; "
        "(ii) acoustic prosodic feature extraction from child only speech segments; and (iii) explainable outputs that "
        "combine annotated transcripts with SHAP explanations and counterfactual suggestions. Using ASDBank corpora [2]–[4] "
        "and component specific model selection, the pragmatic component achieves 0.859 accuracy and 0.851 weighted F1 on a held out test set "
        "after RFECV feature selection, while the acoustic component reaches 0.889 accuracy with a random forest baseline. "
        "The system is deployed as a FastAPI service with a training interface, a model registry, and reproducible feature extraction pipelines."
    )
    _set_paragraph_text(abstract_p, f"Abstract—{abstract_text}")
    _set_paragraph_text(
        keywords_p,
        "Keywords—autism spectrum disorder, speech analysis, pragmatic features, acoustic prosody, TalkBank, ASDBank, "
        "explainable AI, SHAP, counterfactual explanations",
    )

    # Remove all template guidance paragraphs after keywords, but keep the final sectPr.
    children = list(body)
    sectPr = body.find("w:sectPr", namespaces=NS)
    kw_idx = children.index(keywords_p)
    for child in children[kw_idx + 1 :]:
        if child.tag == f"{{{NS['w']}}}sectPr":
            continue
        body.remove(child)

    # Force IEEE two-column body.
    if sectPr is None:
        sectPr = etree.SubElement(body, f"{{{NS['w']}}}sectPr")
    cols = sectPr.find("w:cols", namespaces=NS)
    if cols is None:
        cols = etree.SubElement(sectPr, f"{{{NS['w']}}}cols")
    cols.set(f"{{{NS['w']}}}num", "2")
    cols.set(f"{{{NS['w']}}}space", "18pt")

    # Paper content (figure/table placeholders are explicit).
    P: list[tuple[str, str]] = []

    P += [
        ("Heading1", "I. INTRODUCTION"),
        (
            "BodyText",
            "Autism spectrum disorder (ASD) is a neurodevelopmental condition associated with differences in social communication and interaction "
            "and with restricted or repetitive behaviors. Clinical diagnosis relies on expert assessment and structured instruments, which are time "
            "intensive and not equally accessible across settings. Speech and conversation provide an ecologically valid behavioral signal that can be "
            "collected non invasively and repeatedly, including pragmatic competence, turn coordination, response relevance, and prosodic regulation. "
            "These signals motivate computational screening tools that are accurate, transparent, and deployable in telehealth and community contexts.",
        ),
        (
            "BodyText",
            "The TalkBank ecosystem provides standardized transcription formats (CHAT) and open corpora to support reproducible study of language behavior [1]. "
            "ASDBank includes CHAT formatted child adult interactions across multiple studies, including Eigsti [2], Nadig [3], and Rollins [4]. However, "
            "research prototypes often remain disconnected from deployable systems. Bridging this gap requires robust input handling for transcripts and audio, "
            "leakage free preprocessing, team friendly modular development, and human interpretable explanations.",
        ),
        (
            "BodyText",
            "We present Artistic, a multimodal ASD screening system that accepts CHAT files, raw text, or audio. Audio is transcribed using Whisper compatible backends [6] "
            "and converted into a CHAT like structure with segment timestamps, enabling unified feature extraction. The system implements three feature components: "
            "pragmatic and conversational (implemented), acoustic and prosodic (implemented), and syntactic and semantic (placeholder). Models are trained per component "
            "with explicit constraints on model families and stored in a registry with metadata. The API returns predictions together with transcript annotations, "
            "SHAP explanations [7], and counterfactual what if suggestions [8].",
        ),
        (
            "BodyText",
            "Contributions: 1) a pragmatic conversational feature set aligned with turn taking, topic coherence, pause and latency, and repair strategies, with explicit definitions "
            "and extraction metadata; 2) child only acoustic prosodic extraction using transcript or diarization timing; 3) component specific training with RFECV feature selection "
            "and stratified hyperparameter search; and 4) integrated explainability for both developers and end users.",
        ),
        ("Heading1", "II. RELATED WORK"),
        (
            "BodyText",
            "Computational ASD screening from speech has been studied via acoustic prosody, lexical and syntactic markers, and pragmatic discourse phenomena. Feature engineered methods "
            "are attractive because they can be aligned with observable behaviors, audited, and compared across corpora. Our work adopts this approach but integrates modern explainability "
            "so that both global patterns and individual predictions can be inspected.",
        ),
        (
            "BodyText",
            "Semantic similarity and coherence measures have been proposed as markers of atypical discourse, including approaches that compare semantic similarity in child language samples [18]. "
            "Turn timing and conversational coordination differences have also been empirically reported in autism related interactions [17]. Our pragmatic modules operationalize these phenomena "
            "in a way that is directly tied to extractable events in CHAT transcripts and timestamped transcriptions.",
        ),
        (
            "BodyText",
            "TalkBank and the CHAT standard provide a foundation for interoperable transcripts and metadata [1], while PyLangAcq provides Python tooling for CHAT parsing [5]. For semantic "
            "similarity and token level processing, our topic coherence and repair modules leverage spaCy vectors and similarity methods [21]. For waveform features, librosa provides robust "
            "acoustic descriptors [12]. Modeling builds on scikit-learn [11] and standard classifiers including SVMs [13] and random forests [14], as well as gradient boosting implementations "
            "such as XGBoost [9] and LightGBM [10]. For interpretability, SHAP provides a unified additive explanation framework [7], while counterfactual explanation methods such as DiCE formalize "
            "actionable perturbations [8].",
        ),
        ("Heading1", "III. SYSTEM ARCHITECTURE"),
        (
            "BodyText",
            "Artistic is organized as five layers: (1) input handling, (2) preprocessing, (3) feature extraction, (4) modeling and fusion, and (5) explanation and visualization. "
            "The system is exposed through a FastAPI backend and an optional web frontend. The codebase is modular: each component implements independent feature extraction and model "
            "training while adhering to shared data structures for transcripts, features, predictions, and metadata.",
        ),
        (
            "BodyText",
            "FIGURE 1 PLACEHOLDER: Overall architecture. Show the flow from inputs (audio, CHAT, text) through transcription or parsing, feature extraction modules, component models, optional fusion, "
            "model registry, and explainability outputs. Indicate saved artifacts such as output CSV files, models/*/metadata.json, assets/shap/* plots, and counterfactual autoencoders.",
        ),
        ("figurecaption", "Fig. 1. End to end architecture of Artistic with three components, component specific modeling, optional fusion, and explainability outputs."),
        ("Heading2", "A. Input Handling and Audio Transcription"),
        (
            "BodyText",
            "The input handler determines whether the input is a CHAT file, text, or audio. CHAT transcripts are parsed with pylangacq [5] to extract utterances, participant metadata, and "
            "optional morphological and grammatical tiers. For audio, transcription is performed using Whisper [6]. To improve robustness across platforms, multiple backends are supported and "
            "selected dynamically; on macOS, faster-whisper is preferred when available to avoid PyTorch instability. Transcription segments are converted into utterances with start and end times "
            "and can be saved as CHAT like files for reproducible downstream processing.",
        ),
        (
            "BodyText",
            "Utterance validity is determined by minimal length constraints and by filtering of non linguistic markers. This yields consistent feature computation across transcripts with varying annotation density. "
            "Extraction metadata, including the number of valid child utterances and whether timing is available for more than half the utterances, is propagated to downstream modules to contextualize model outputs.",
        ),
        ("Heading2", "B. Speaker Tagging and Child Only Audio Extraction"),
        (
            "BodyText",
            "Adult caregiver speech can contaminate child targeted acoustic metrics. The audio processor can tag speakers using a pitch based heuristic, motivated by typical differences in F0 between children and adults. "
            "The acoustic extractor further provides a child audio extractor that concatenates child segments based on transcript timestamps or diarization labels, producing a temporary child only waveform for acoustic feature extraction. "
            "When child segmentation fails, the system falls back to full audio and records the fallback in metadata.",
        ),
        ("Heading2", "C. Feature Extraction Orchestration"),
        (
            "BodyText",
            "Feature extraction is orchestrated by a single controller that instantiates active modules and merges the resulting feature dictionaries. Each module exposes an explicit feature name list, enabling stable column ordering and consistent scaling. "
            "Extraction can be run over directories of CHAT files or over audio sets, in which case transcription and extraction are performed sequentially and cached to CSV.",
        ),
        ("Heading2", "D. Model Registry and Reproducibility"),
        (
            "BodyText",
            "Trained models are stored in a registry with structured metadata: feature lists, sample counts, hyperparameter settings, evaluation metrics, and confusion matrices. Registry entries allow comparisons across model versions and support later fusion and stacking. "
            "Optional cloud synchronization via HuggingFace Hub enables team workflows and reproducible backups.",
        ),
        ("Heading2", "E. Model Fusion"),
        (
            "BodyText",
            "Fusion methods include voting, averaging, weighted averaging, maximum confidence selection, and stacking. Weighted fusion computes a final ASD probability as a convex combination of component probabilities:",
        ),
        (
            "BodyText",
            "Equation (1) PLACEHOLDER: p_ASD = (sum_c w_c p_c) / (sum_c w_c). Use default weights w_prag=0.5, w_acoustic=0.25, w_syntactic=0.25, and describe tuning and calibration as future work.",
        ),
        ("Heading2", "F. Deployment Interface"),
        (
            "BodyText",
            "The backend exposes endpoints for training and inference and returns structured JSON containing predictions, component probabilities, and explanation artifacts. A typical request includes either a transcript file or an audio file. On audio inputs, transcription is executed first "
            "and the resulting timestamps enable both timing features and transcript annotation. Training jobs persist model artifacts and SHAP plots to disk, and the registry enables selecting the active model per component for deployment.",
        ),
        ("Heading1", "IV. DATASET AND PREPROCESSING"),
        (
            "BodyText",
            "We use ASDBank corpora distributed through TalkBank [1], including datasets associated with Eigsti et al. [2], Nadig et al. [3], and Rollins [4]. These corpora provide transcripts of child adult interactions in CHAT format with diagnostic grouping. "
            "The system supports audio aligned corpora and audio only datasets by generating CHAT like transcripts from transcriptions.",
        ),
        (
            "BodyText",
            "Diagnosis labels are normalized by combining transcript header information and directory structure. Labels such as TYP are mapped to TD, and ambiguous labels are handled conservatively. The system currently targets binary ASD vs TD classification; additional diagnostic classes "
            "can be incorporated by extending the mapper.",
        ),
        (
            "BodyText",
            "Preprocessing is implemented as an explicit pipeline to minimize leakage. After identifying numeric feature columns, the pipeline performs a stratified train test split. Cleaning includes replacing infinite values, imputing missing values (median), and outlier handling (clipping). "
            "Feature selection and scaling are fitted on training data only and then applied to test data. Acoustic preprocessing additionally applies variance thresholding and standardization.",
        ),
        (
            "BodyText",
            "FIGURE 2 PLACEHOLDER: Dataset and split summary. Include a table listing per component sample counts (pragmatic 247, acoustic 108, syntactic placeholder 47) and class distributions. Include a bar plot of ASD vs TD counts and per corpus counts.",
        ),
        ("figurecaption", "Fig. 2. Dataset composition and stratified splitting protocol across components."),
        ("Heading1", "V. FEATURE ENGINEERING"),
        (
            "BodyText",
            "Feature extraction is implemented as composable modules with explicit feature name lists. The pragmatic conversational component implements four methodology aligned modules and two supporting modules, with 214 total features. The acoustic component computes a waveform based feature set "
            "of 52 or more descriptors. All extractors return both features and metadata describing extraction assumptions such as timing availability and utterance counts.",
        ),
        (
            "BodyText",
            "A key design principle is traceability: each feature is tied to a concrete extraction rule (pattern, threshold, or statistical summary). This supports debugging and helps avoid silent failures when transcripts are missing tiers or timing information.",
        ),
        ("Heading2", "A. Turn Taking (Section 3.3.1)"),
        (
            "BodyText",
            "Turn taking features quantify participation balance and temporal coordination. Basic statistics include counts of child and adult turns, turns per minute, and initiation ratios. Turn length features summarize word counts per utterance; variability is captured via standard deviation and "
            "coefficient of variation. When timestamps exist, duration based features are computed from utterance start and end times.",
        ),
        (
            "BodyText",
            "Inter turn gap features measure response latency between consecutive turns. Overlap and interruption features quantify simultaneous speech and rapid cut ins, using thresholds to avoid counting micro overlaps. These features are motivated by prior work on turn timing and coordination "
            "differences in autism related conversations [17].",
        ),
        ("Heading2", "B. Topic Coherence and Maintenance (Section 3.3.2)"),
        (
            "BodyText",
            "Topic coherence is computed via a hybrid semantic and topic modeling approach. When spaCy vectors are available [21], semantic similarity between consecutive utterances is computed using cosine similarity of document vectors. This yields a coherence score, standard deviation, and extrema. "
            "Child response relevance compares each child utterance to the immediately preceding adult turn.",
        ),
        (
            "BodyText",
            "Topic shift detection uses a sliding window and a similarity threshold of 0.3. LDA topic modeling [15] produces topic diversity, entropy, dominant topic ratios, and child topic consistency. Lexical overlap and novel word ratios quantify continuity and divergence. This module is conceptually "
            "aligned with semantic similarity analyses in ASD language samples [18].",
        ),
        ("BodyText", "Equation (2) PLACEHOLDER: cosine similarity sim(a,b) = (v_a · v_b)/(||v_a|| ||v_b||). Document how vectors are obtained from spaCy and how the system falls back when vectors are absent."),
        ("Heading2", "C. Pause and Latency (Section 3.3.3)"),
        (
            "BodyText",
            "Pause and latency features combine between turn response time and within utterance hesitations. Filled pauses are detected using lexical patterns and CHAT encodings. Unfilled pauses are inferred from CHAT pause markers and from timing gaps. Statistics include percentiles and interquartile ranges, "
            "as well as derived indicators of delayed responses.",
        ),
        (
            "BodyText",
            "A novelty is threshold selection derived from Gaussian mixture clustering over ASDBank response times, yielding interpretable clusters (rapid, processing, disengaged) and thresholds used to compute immediate and very delayed response ratios. This provides a data driven alternative to choosing arbitrary latency cutoffs.",
        ),
        ("BodyText", "Equation (3) PLACEHOLDER: define delayed response indicators I(g_i > tau_long) and I(g_i > tau_very_long) with tau_long=2.00 s and tau_very_long=4.32 s. Include a short explanation of the clustering and threshold intersections."),
        ("Heading2", "D. Conversational Repair (Section 3.3.4)"),
        (
            "BodyText",
            "Repair features capture how speakers manage breakdowns. Lexical markers (e.g., I mean) and CHAT retrace codes such as [/], [//], and [///] are used to detect repair events. The extractor reports repair attempt rates, success rates, sequence lengths, and strategy diversity.",
        ),
        (
            "BodyText",
            "Repair effectiveness is approximated using semantic similarity between a repair utterance and its triggering context when spaCy models are available [21]. While approximate, this provides a quantitative proxy for whether repairs increase relevance and coherence.",
        ),
        ("Heading2", "E. Pragmatic Linguistic Markers (Supporting)"),
        (
            "BodyText",
            "Supporting pragmatic linguistic features include mean length of utterance (MLU in words and morphemes), lexical diversity (TTR and corrected TTR), lexical density, echolalia indicators, pronoun usage proxies, question ratios, social phrase rates, discourse markers, and non verbal behavioral markers. "
            "These features complement the primary methodology modules and help capture broader pragmatic competence.",
        ),
        ("Heading2", "F. Pragmatic Audio Timing (Supporting)"),
        (
            "BodyText",
            "When audio is available, pragmatic timing features are computed from transcription segments and optional waveform analysis. The extractor estimates pause distributions, speaking rates, articulation rates, and segment duration variability, and stores detected pauses for transcript annotation. These audio derived measures "
            "reduce dependence on transcript pause markers and improve temporal resolution.",
        ),
        ("Heading2", "G. Acoustic Prosody (Waveform Based)"),
        (
            "BodyText",
            "Acoustic features are extracted using librosa [12] and include pitch statistics and dynamics, MFCCs and deltas, chroma and tonnetz features, spectral centroid and rolloff, energy and intensity statistics, and voice quality proxies. Child only extraction concatenates child segments to reduce adult speech contamination "
            "before feature computation.",
        ),
        ("BodyText", "TABLE I PLACEHOLDER: Feature set summary by component. Include counts (pragmatic 214, RFECV selected 44, acoustic 52+, syntactic placeholder 10), representative features, and modality (transcript, timing, waveform)."),
        ("Heading1", "VI. MODEL TRAINING"),
        (
            "BodyText",
            "Artistic enforces component specific model families to match feature characteristics and control overfitting. The pragmatic component uses SVM [13] and logistic regression with class_weight balanced. The acoustic component uses random forest [14] and XGBoost [9]. Syntactic semantic models are configured for LightGBM [10] "
            "and gradient boosting but currently operate on placeholder features.",
        ),
        (
            "BodyText",
            "This component specific restriction is intended to prevent misuse of overly expressive models on small datasets and to encourage interpretable baselines. For example, SVM and logistic regression provide stable decision boundaries and coefficients that can be analyzed with SHAP and with feature importance proxies.",
        ),
        ("Heading2", "A. Feature Selection and Hyperparameter Search"),
        (
            "BodyText",
            "Pragmatic training applies RFECV when feature dimensionality exceeds 30, selecting a subset that maximizes weighted F1 under stratified folds. Hyperparameters are tuned via RandomizedSearchCV with 20 iterations and 3 fold stratified cross validation. Search spaces include C and gamma for SVM and C and solver for logistic regression, "
            "with class_weight fixed to balanced.",
        ),
        ("BodyText", "TABLE II PLACEHOLDER: Hyperparameter search spaces and selected parameters for pragmatic and acoustic models. Include n_iter, cv folds, scoring metric, and final best parameters."),
        ("Heading2", "B. Evaluation Metrics"),
        (
            "BodyText",
            "We report accuracy and weighted precision, recall, and F1 to account for class imbalance. For models with probability outputs we also report ROC AUC. Matthews correlation coefficient and confusion matrices are stored to provide an error sensitive view of performance.",
        ),
        ("Heading1", "VII. EXPLAINABILITY"),
        (
            "BodyText",
            "Explainability is implemented at three levels. First, transcript annotation highlights where features were extracted, mapping patterns such as filled pauses, repairs, topic shifts, and discourse markers to spans. Second, SHAP explanations [7] provide global and local additive attributions for model predictions. Third, counterfactual generation produces a minimally "
            "modified feature vector that flips the predicted label, regularized by an autoencoder to keep the counterfactual realistic.",
        ),
        (
            "BodyText",
            "The SHAP implementation is integrated into the training and inference workflows. For tree models, TreeExplainer is used, while linear and kernel explainers are used for logistic regression and SVM as needed. Global explanations are saved as beeswarm and bar plots, and local explanations are saved as waterfall plots. Counterfactual generation uses gradient based optimization over "
            "the input feature vector with an additional realism term computed from an autoencoder reconstruction loss.",
        ),
        ("BodyText", "FIGURE 3 PLACEHOLDER: Annotated transcript excerpt from a prediction response, including a legend of markers and color codes."),
        ("figurecaption", "Fig. 3. Example annotated transcript highlighting pragmatic and conversational markers used by the feature extractors."),
        ("BodyText", "FIGURE 4 PLACEHOLDER: SHAP global beeswarm and bar plots for a best component model. Use assets/shap/acoustic_prosodic_random_forest/global_beeswarm.png and global_bar.png or corresponding pragmatic plots."),
        ("figurecaption", "Fig. 4. Global SHAP explanations for a trained component model."),
        ("BodyText", "FIGURE 5 PLACEHOLDER: SHAP local waterfall for one instance, using assets/shap/local/*/waterfall.png. Pair with annotated transcript excerpt."),
        ("figurecaption", "Fig. 5. Local SHAP waterfall explanation for a single instance."),
        ("Heading1", "VIII. RESULTS"),
        (
            "BodyText",
            "Results were obtained from stored training artifacts and registry metadata. The pragmatic component was trained on 247 samples with 214 features. RFECV selected 44 features and improved generalization for the SVM model. The acoustic component was trained on 108 samples using child only waveform segments and an acoustic feature set. The syntactic semantic component uses placeholder features and serves as a baseline for future integration.",
        ),
        (
            "BodyText",
            "Pragmatic SVM with RFECV achieved accuracy 0.859 and weighted F1 0.851. Pragmatic logistic regression achieved accuracy 0.808 and weighted F1 0.809 under the same configuration. Acoustic random forest achieved accuracy 0.889 and weighted F1 0.888. The placeholder syntactic semantic model achieved accuracy 0.500, consistent with an uninformative baseline and highlighting the need for full implementation.",
        ),
        ("BodyText", "TABLE III PLACEHOLDER: Component performance summary including accuracy, weighted F1, precision, recall, ROC AUC, and Matthews correlation. Include confusion matrices: acoustic random forest [[15,1],[2,9]] and pragmatic models as stored."),
        ("BodyText", "FIGURE 6 PLACEHOLDER: Confusion matrices and ROC curves for best pragmatic and acoustic models."),
        ("figurecaption", "Fig. 6. Confusion matrices and ROC curves for component models."),
        ("Heading2", "A. Feature Importance"),
        (
            "BodyText",
            "The selected pragmatic subset emphasizes interactional timing and conversational contingency. Selected features include long pause ratios, child initiation ratios, semantic coherence variability, topic drift, lexical overlap metrics, filled pause density, and repair success indicators. These features align with reported differences in turn timing and semantic similarity measures in autism related speech [17], [18].",
        ),
        ("Heading2", "B. Ablation and Robustness Considerations"),
        (
            "BodyText",
            "Ablation is enabled by the modular pipeline. Removing RFECV increases overfitting risk due to the high feature to sample ratio. Similarly, acoustic performance depends on consistent child only segmentation; adult speech contamination can shift pitch and energy distributions. The system records whether child segmentation was successful to enable stratified evaluation of this factor.",
        ),
        ("Heading2", "C. Error Analysis"),
        (
            "BodyText",
            "Misclassifications frequently occur in short sessions with limited child utterances, where estimates of topic drift and repair patterns are unstable. In such cases, extraction metadata such as utterance counts and timing availability can be used to flag low reliability inputs. Future work will integrate explicit uncertainty estimation and calibrated fusion to reduce overconfident errors.",
        ),
        ("Heading2", "D. Runtime and Engineering Observations"),
        (
            "BodyText",
            "From an engineering standpoint, transcript based feature extraction is fast and deterministic, enabling batch processing of corpora for experimentation. Audio based extraction is dominated by transcription time, which varies with backend choice and hardware. To support reproducibility, the system persists intermediate transcriptions, extracted feature CSV files, and model metadata in a standardized layout. This enables re running evaluation, regenerating SHAP plots, and comparing model versions without re extracting features unless the extractor changes.",
        ),
        ("Heading1", "IX. LIMITATIONS AND FUTURE WORK"),
        (
            "BodyText",
            "The system provides strong pragmatic and acoustic baselines and an end to end framework, but limitations remain. The syntactic semantic component is not yet implemented; completing it with robust parsing based features is necessary before evaluating full multimodal fusion. Dataset heterogeneity and limited sample sizes constrain generalization, and demographic fairness has not yet been evaluated. Future work includes cross corpus validation, calibrated fusion and stacking, improved diarization, and prospective clinical evaluation.",
        ),
        ("Heading1", "X. CONCLUSION"),
        (
            "BodyText",
            "We presented Artistic, a modular multimodal ASD screening system integrating pragmatic conversational and acoustic prosodic feature engineering with explainable outputs. The pipeline supports CHAT transcripts and audio, enforces component specific models, and stores artifacts with metadata for reproducibility. On ASDBank corpora, the pragmatic component achieved 0.859 accuracy and 0.851 weighted F1 after RFECV, and the acoustic component achieved 0.889 accuracy with a random forest baseline. The modular architecture enables continued development toward complete multimodal fusion and clinically meaningful evaluation.",
        ),
        ("Heading5", "Acknowledgment"),
        (
            "BodyText",
            "We thank the supervisor and co supervisor for guidance on research design, methodology alignment, and evaluation. We also acknowledge TalkBank for providing open access corpora and standardized tooling that enabled reproducible analysis [1].",
        ),
    ]

    refs = [
        "[1] B. MacWhinney, The CHILDES Project: Tools for Analyzing Talk, 3rd ed. Mahwah, NJ, USA: Lawrence Erlbaum Associates, 2000.",
        "[2] I.-M. Eigsti, L. Bennetto, and M. Dadlani, “Beyond pragmatics: Morphosyntactic development in autism,” Journal of Autism and Developmental Disorders, vol. 37, pp. 1007–1023, 2007, doi: 10.1007/s10803-006-0259-2.",
        "[3] J. Bang and A. Nadig, “Language learning in autism: Maternal linguistic input contributes to later vocabulary,” Autism Research, vol. 8, no. 2, pp. 214–233, 2015.",
        "[4] P. R. Rollins, “Pragmatic accomplishments and vocabulary development in pre-school children with autism,” American Journal of Speech-Language Pathology, vol. 8, pp. 85–94, 1999.",
        "[5] J. L. Lee, R. Burkholder, G. B. Flinn, and E. R. Coppess, “Working with CHAT transcripts in Python,” Tech. Rep. TR-2016-02, Dept. Computer Science, University of Chicago, 2016. [Online]. Available: https://pylangacq.org/",
        "[6] A. Radford et al., “Robust speech recognition via large-scale weak supervision,” OpenAI, 2022. [Online]. Available: https://cdn.openai.com/papers/whisper.pdf",
        "[7] S. M. Lundberg and S.-I. Lee, “A unified approach to interpreting model predictions,” in Proc. 31st Int. Conf. Neural Information Processing Systems (NeurIPS), 2017, pp. 4766–4777.",
        "[8] R. K. Mothilal, A. Sharma, and C. Tan, “Explaining machine learning classifiers through diverse counterfactual explanations,” in Proc. Conf. Fairness, Accountability, and Transparency (FAT*), 2020, pp. 607–617, doi: 10.1145/3351095.3372850.",
        "[9] T. Chen and C. Guestrin, “XGBoost: A scalable tree boosting system,” in Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining, 2016, pp. 785–794, doi: 10.1145/2939672.2939785.",
        "[10] G. Ke et al., “LightGBM: A highly efficient gradient boosting decision tree,” in Advances in Neural Information Processing Systems 30, 2017.",
        "[11] F. Pedregosa et al., “Scikit-learn: Machine learning in Python,” Journal of Machine Learning Research, vol. 12, pp. 2825–2830, 2011.",
        "[12] B. McFee et al., “librosa: Audio and music signal analysis in Python,” in Proc. SciPy, 2015.",
        "[13] C. Cortes and V. Vapnik, “Support-vector networks,” Machine Learning, vol. 20, pp. 273–297, 1995.",
        "[14] L. Breiman, “Random forests,” Machine Learning, vol. 45, no. 1, pp. 5–32, 2001.",
        "[15] D. M. Blei, A. Y. Ng, and M. I. Jordan, “Latent Dirichlet allocation,” Journal of Machine Learning Research, vol. 3, pp. 993–1022, 2003.",
        "[16] T. Mikolov, K. Chen, G. Corrado, and J. Dean, “Efficient estimation of word representations in vector space,” 2013, arXiv:1301.3781. [Online]. Available: https://arxiv.org/abs/1301.3781",
        "[17] S. Wehrle et al., “Turn-timing in conversations between autistic adults: Typical short-gap transitions are preferred, but not achieved instantly,” PLOS ONE, vol. 18, no. 4, 2023, doi: 10.1371/journal.pone.0284029.",
        "[18] R. J. Adams et al., “A pseudo-value approach to analyze the semantic similarity of the speech of children with and without autism spectrum disorder,” Frontiers in Psychology, vol. 12, 2021, doi: 10.3389/fpsyg.2021.668344.",
        "[19] D. P. Kingma and M. Welling, “Auto-encoding variational Bayes,” 2013, arXiv:1312.6114. [Online]. Available: https://arxiv.org/abs/1312.6114",
        "[20] P. Povey et al., “The Kaldi speech recognition toolkit,” in IEEE Workshop on Automatic Speech Recognition and Understanding, 2011.",
        "[21] M. Honnibal, I. Montani, S. Van Landeghem, and A. Boyd, “spaCy: Industrial-strength Natural Language Processing in Python,” 2020. [Online]. Available: https://doi.org/10.5281/zenodo.1212303",
    ]

    P.append(("Heading5", "References"))
    for r in refs:
        P.append(("references", r))

    nodes = [_make_p(style, text) for style, text in P]
    for node in nodes:
        body.insert(body.index(sectPr), node)

    doc_xml_path.write_bytes(etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone="yes"))
    _write_docx_from_dir(WORK_DIR, OUTPUT_DOCX)
    print(f"Wrote {OUTPUT_DOCX}")


if __name__ == "__main__":
    main()

