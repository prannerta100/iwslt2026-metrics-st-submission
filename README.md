# iwslt2026-metrics-st-submission


## Speech Translation Metrics track

### Description
Speech translation has been a core focus of IWSLT for years, yet its evaluation remains underexplored. Most existing evaluations assume gold segmentation, an unrealistic scenario for real-world systems. When defering to automatically segmented speech, conventional text-to-text metrics become less reliable. Despite this, current evaluation practices still rely heavily on these metrics, highlighting the need for more robust and realistic assessment approaches.

This shared task focuses on Quality Estimation for speech translation, a reference-free evaluation of speech translation quality. Participants will assess the quality of translations produced in other IWSLT shared tasks, and system outputs will be evaluated based on their correlation with human judgments.

We consider two angles for speech translation quality estimation:

Speech sample + system translation → score
ASR transcript + system translation → score
To evaluate the submissions, we compute correlations with human judgmnets of quality. We encourage participation in both scenarios or exploring other approaches. The translation segments occur in documents which can also provide additional context to the quality estimation.

We look forward to your submissions!

### Data
Train and development data is stored on Hugging Face: maikezu/iwslt2026-metrics-shared-train-dev

The train set includes human annotations from IWSLT 2023, WMT 2024 and WMT 2025. The dev set consists of the human annotaions from IWSLT 2025 (ACL Talks). More details can be found on Hugging Face.

The test set will consist of human annotations of IWLST 2026 and include the language pairs English→German and English→Chinese.

As an example input, consider the following audio:

and the corresponding testset entry:

{
    "audio": "sample.wav"
    "src_text": "Plans are well underway for races to Mars and the Moon in 1992, by solar sail. The race to Mars is to commemorate Columbus's journey to the New World 500 years ago, and the one to the Moon is to promote the use of solar sails in space exploration.",
    "tgt_text": "Pläne sind gut im Wege für Rennen nach Mars und der Mond in 1992, mit Sonnensegel. Das Rennen zum Mars ist zu Kolumbus' Reise in die Neue Welt vor 500 Jahren zu gedenken, und der eine zum Mond ist den Gebrauch von Sonnensegeln in Weltraum Exploration zu fördern.",
    "score": 71.5,
}
The goal is to predict the human score score given the source speech audio, the ASR transription src_text, and tgt_text. The human score is not 100 because the style of the automatic translation is awkward at places.

### Baselines
We consider the following quality estimation baselines, which are available on GitHub:

ASR-based COMETKiwi-22
ASR-based COMET-partial
SpeechQE
More baseline to be announced

### Submission
More details on the submission process and timeline will be released soon.

As part of the submission, we require a system description paper to be submitted to IWLST to be reviewed.

### Evaluation
Quality Estimation models will be evaluated by measuring their correlation with human judgments similar to WMT Metrics Shared Task. For each language pair, we compute:

Kendall’s Tau: segment-level measure, akin to Pearson correlation groupped by item. This measures the ability of metrics to select the best translation given a single source.
Soft Pairwise Accuracy: system-level measure. This reveals how good the metric is at ranking the participating systems.
The scripts for meta-evaluation of metrics are available publicly on GitHub.

### Important Dates
The preliminary timeline is below and may be subject to minor changes.

Jan 1, 2026	Release of shared task training and dev data
Apr 21-30, 2026	Evaluation period (only for Metrics Task; postponed)
May 10, 2026	System paper submission deadline (postponed)
May 15, 2026	Notification of acceptance
June 1, 2026	Camera ready deadline (all papers)
July 3-4, 2026	IWSLT conference

### Organizers
Maike Züfle, Karlsruhe Institute of Technology, maike.zuefle@kit.edu
Vilém Zouhar, ETH Zurich, vzouhar@ethz.ch
Brian Thompson
Dominik Macháček, Charles University
Mattias Sperber, Apple
Marine Carpuat, University of Maryland
HyoJung Han, University of Maryland
Marco Turchi, Zoom
Matteo Negri, FBK
ContactPermalink
Chair(s): Maike Züfle maike.zuefle@kit.edu; Vilém Zouhar vzouhar@ethz.ch

Discussion: iwslt-evaluation-campaign@googlegroups.com