# ECG Arrhythmia Classification on Unseen Patients — and Knowing When Not to Trust the Result

Heartbeat classification (AAMI classes **N, S, V, F**) evaluated the way it would be used in practice:
**the models are tested only on patients they have never seen.** Training uses 18 MIT-BIH patients; testing uses
22 other MIT-BIH patients and two external databases recorded elsewhere. No test set is used for training, tuning or
model selection.

The project then asks a second question that matters just as much in clinical use: **can the system tell, without any
labels, for which recordings its own result should not be trusted?**

**Highlights**
- **Three test sets, no tuning on any of them:** 22 unseen MIT-BIH patients (DS2), the MIT-BIH Supraventricular Arrhythmia
  Database (78 recordings) and the St Petersburg INCART database (75 recordings from 32 patients, different hospital and leads).
- **Patient-relative features** lift S-beat sensitivity from 6–8% to 92% with a plain logistic regression.
- **Most errors were false fusion alarms, and they have a physiological explanation.** AAMI's "N" class merges different
  conduction patterns. In record 212, whose usual beat is right bundle branch block, 917 of its
  922 normally conducted beats were called fusion. A recording-level **fusion gate** (no ventricular
  beats → no fusion is possible) cuts false F alarms on DS2 from 4,740 to 1,962 and raises
  accuracy from 84.8% to 90.1% (macro-F1 0.608 → 0.631).
- **A label-free reliability flag.** How consistent a patient's beats are with their own median beat predicts which
  recordings the model will get wrong. With a threshold chosen on training patients only, recordings that are *not*
  flagged reach 94.1% (SVDB) and 91.9% (INCART) beat accuracy, flagged ones
  62.0% and 65.6%.
- **An AF-aware rule** (a rhythm detector trained on MIT-BIH training patients only) removes 41% of false S alarms on DS2
  (3,479 → 2,045).
- **Fully automatic pipeline:** 99.2% QRS detection, with classification on detected beats at the level of expert-annotated beats.
- **An honest result:** no single model wins everywhere. On DS2 the main pipeline (ensemble + AF rule + fusion gate) has the best
  macro-F1 (0.631); on the two external databases the two-lead CNN is clearly better
  (0.528 and 0.674 vs 0.510 and 0.461), but it finds far fewer
  S beats (27% and 61% vs 64% and 96%).

![Model comparison](results/model_comparison.png)

## Why I built this
I'm a software engineering student at Sichuan University, and I work as an intern on a hospital
teleconsultation platform. My own research looks at how unreliable or incomplete clinical data affects
AI-assisted clinical decisions. One question keeps coming back in that work: *does a model still hold up
when it meets a patient it has never seen?*

ECG classification is a good place to test that honestly. MIT-BIH is public and well documented, and many
published results look near-perfect only because the same patients appear in training and testing. I wanted a
project that uses the strict unseen-patient evaluation from the start, finds out *why* models fail on new
patients, and writes down what I learned, including the parts that did not work.

Because my research is about data reliability, the last part of the project turns the question around: instead of
only asking how accurate the model is, it asks whether the system can recognise the recordings where its own output
should not be trusted, before anyone acts on it.

## Why inter-patient matters
Many MIT-BIH papers shuffle all beats and split them randomly, so beats from the same patient appear in
both training and test sets. Accuracy then looks close to 99% but mostly measures how well the model
recognises *people*, not arrhythmias. This project uses the standard inter-patient split of
de Chazal et al. (2004):

| Set | Records | Use |
|---|---|---|
| DS1 (18 patients) | 101, 106, 109, 112, 114, 115, 116, 119, 122, 124, 201, 203, 205, 208, 209, 215, 220, 230 | training |
| DS1 hold-out (4 patients) | 108, 118, 207, 223 | early stopping and ensemble weight only |
| DS2 (22 patients) | 100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234 | final test, used once |

Paced records (102, 104, 107, 217) are excluded and Q beats are dropped, following AAMI EC57.

## Method
**Beats.** Lead A (MLII on MIT-BIH, ECG1 on SVDB, II on INCART), resampled to 360 Hz, 0.5–40 Hz band-pass filter,
256-sample window around each R peak (0.25 s before, 0.46 s after), z-normalised per beat.

**Patient-relative features.** The key design choice: every beat is described *relative to its own patient*.
- 7 RR-interval features as log ratios to the *local* rhythm (previous 10 beats, surrounding 50 beats, previous RR).
- 4 template features: correlation and RMS distance between the beat and the patient's median beat,
  for the whole beat and for the QRS complex only. The template is computed from the unlabeled signal.

**Models.**
- **Logistic regression** on the 11 features.
- **1D CNN** (4 conv blocks) whose input is the beat **and** the beat minus the patient template (2 channels),
  fused with the 11 features. Class-weighted loss, augmentation (amplitude scaling, baseline wander, noise),
  early stopping on the held-out DS1 patients, 3 seeds averaged.
- **Ensemble:** weighted average of the two models' probabilities. The weight (0.9 logistic regression / 0.1 CNN)
  is chosen on the DS1 hold-out patients, never on test data.
- **AF-aware rule:** a gradient-boosting detector recognises atrial fibrillation / flutter from rhythm context alone
  (RR irregularity over ±15 beats and P-wave consistency). Inside detected AF, S predictions become N, because by
  definition there are no supraventricular ectopic beats during AF. Model and threshold (0.75) are chosen by
  leave-patient-out cross-validation on DS1. On DS2 the detector reaches AF sensitivity 82.6%, +P 92.8%,
  specificity 98.6%.
- **Fusion gate** (`src/reliability.py`): a fusion beat is a normal and a ventricular beat firing together, so it cannot occur
  in a recording without ventricular ectopy. If the model predicts V for fewer than 1% of a recording's beats, its F
  predictions are replaced by the best of N / S / V. The 1% threshold is fixed in advance, not tuned.
- **Reliability flag** (`analysis.py`): *beat consistency* = the median correlation between each beat and the patient's
  median beat (lead A). It needs no labels and no model. A recording is flagged when consistency < 0.975, a threshold chosen
  on DS1 patients only (out-of-fold predictions, Youden's J for "recording accuracy below 90%").
- **Two-lead experiment:** the same models with a second lead (V1 on most MIT-BIH records, ECG2 on SVDB, V1 on INCART).

**Three test sets.**
- **DS2:** 22 unseen MIT-BIH patients (49,661 beats).
- **SVDB:** MIT-BIH Supraventricular Arrhythmia Database, 78 recordings at 128 Hz (184,347 beats, 12,194 S beats).
- **INCART:** St Petersburg INCART 12-lead database, 75 recordings from 32 patients at 257 Hz (175,718 beats).

## Results
Se = sensitivity, +P = positive predictivity, in %. Macro-F1 is the mean F1 over N, S, V and F.
The CNN rows use the average of 3 seeds' probabilities. Full tables (all models, including the two-lead
logistic regression and ensemble) are in [`results/final_results.md`](results/final_results.md).

### DS2 — 22 unseen MIT-BIH patients
| Model | Accuracy | Macro-F1 | N Se / +P | S Se / +P | V Se / +P | F Se / +P |
|---|---|---|---|---|---|---|
| Logistic regression (features) | 81.2% | 0.573 | 80.9 / **99.1** | **92.0** / 31.0 | 82.6 / 92.0 | **48.4** / 3.6 |
| CNN (waveform + features) | 94.5% | 0.612 | 97.7 / 96.7 | 35.3 / 39.4 | 94.2 / 92.6 | 10.3 / **48.8** |
| Ensemble + AF-aware rule | 84.8% | 0.608 | 85.2 / 98.6 | 85.0 / 43.3 | 84.1 / 92.2 | 45.1 / 3.6 |
| **Ensemble + AF rule + fusion gate** | 90.1% | **0.631** | 91.1 / 98.7 | 85.4 / 42.5 | 84.1 / 90.4 | 45.1 / 8.2 |
| Two-lead CNN | **95.5%** | 0.616 | **98.3** / 97.0 | 43.4 / **61.6** | **98.6** / **95.5** | 0.5 / 0.9 |

**Seed variance (single CNNs, mean ± std over 3 seeds on DS2):** macro-F1 0.614 ± 0.057 (one lead) and
0.601 ± 0.044 (two leads); S sensitivity 38.5 ± 14.3% and 42.3 ± 20.5% (`results/cnn*_seeds.json`).

### External test sets (never used for training or tuning)
| Model | SVDB Acc | SVDB Macro-F1 | SVDB S Se / +P | INCART Acc | INCART Macro-F1 | INCART S Se / +P |
|---|---|---|---|---|---|---|
| Logistic regression | 67.6% | 0.498 | 68.6 / 47.3 | 68.6% | 0.449 | 95.7 / 10.9 |
| CNN | 91.9% | 0.525 | 23.1 / 55.7 | 95.5% | 0.626 | 83.2 / 40.8 |
| **Ensemble + AF rule + fusion gate** | 72.4% | 0.510 | 64.4 / 50.8 | 72.9% | 0.461 | 95.6 / 11.4 |
| Two-lead CNN | 92.2% | 0.528 | 27.1 / 59.1 | 97.8% | 0.674 | 61.4 / 73.6 |

SVDB contains only 23 F beats and INCART 219, so N, V and F columns are in `results/final_results.md`.

### Fully automatic pipeline (no expert beat positions)
R peaks found by the XQRS detector (wfdb) instead of the annotations; labels come from the nearest annotation within 150 ms.

| | Result |
|---|---|
| QRS detection, 44 MIT-BIH records (100,733 beats) | Se **99.17%**, +P **99.29%** |
| Logistic regression on detected beats, DS2 | macro-F1 **0.587** · S 87.5 / 30.3 · V 86.7 / 95.5 |

The classifier performs at the same level on automatically detected beats as on expert-annotated ones.

## Where the errors come from

### S errors: atrial fibrillation
![Per-patient errors](results/per_patient_errors.png)

Before the AF-aware rule, most false S alarms on DS2 come from the 5 test patients with atrial fibrillation or flutter
(records 202, 210, 219, 221, 222): during AF the rhythm is irregular, so normal beats often arrive "early", which is
exactly what an S beat looks like to an RR-based model. The rule removes 1,434 false S alarms at the cost of 111 missed
S beats. The per-patient table is in [`results/final_results.md`](results/final_results.md).

### F errors: one "normal" class, several shapes
Before the fusion gate, 4,740 of the 7,553 errors on DS2 were beats wrongly called fusion (F).
They are not random: AAMI's N class merges normal beats with bundle branch block and escape beats, and a model that
compares every beat with *the patient's own usual beat* sees the minority pattern as "on time, but a different shape",
which is exactly the signature of a fusion beat.

![False fusion alarms](results/false_fusion.png)

- **Record 212** has 1,824 right bundle branch block beats and 922 normally conducted beats.
  917 of the normally conducted beats were called F.
- **Record 231** has the same mix: all 314 of its normally conducted beats were called F.
- Beat by beat, these alarms look like real fusion (record 213, right). What separates them is context: real fusion needs
  ventricular beats, and records 117, 212 and 231 have almost none.

The fusion gate uses exactly that context. On DS2, record 117 goes from 44.0% to 98.1% accuracy and record 212
from 65.3% to 98.7%, without losing a single true F beat:

| Test set | False F alarms | Accuracy | Macro-F1 | F Se | F +P |
|---|---|---|---|---|---|
| DS2 | 4,740 → 1,962 | 84.8% → 90.1% | 0.608 → 0.631 | 45.1% → 45.1% | 3.6% → 8.2% |
| SVDB | 42,572 → 37,762 | 70.0% → 72.4% | 0.507 → 0.510 | 47.8% → 4.3% | 0.0% → 0.0% |
| INCART | 35,378 → 31,828 | 71.1% → 72.9% | 0.459 → 0.461 | 47.9% → 47.5% | 0.3% → 0.3% |

On the external databases the gate helps much less: most of their false F alarms happen in recordings where the model
also predicts some V beats (correctly or not), so the gate does not switch on. On SVDB it also removes most of the
23 real F beats.

## When not to trust a result
A clinician reading a remote recording needs to know more than "the model is 90% accurate on average": they need to
know whether *this* recording's result is usable. Beat consistency is a simple, label-free check. In recordings where
beats keep changing shape (noise, electrode problems, or a patient switching between conduction patterns), the
"patient's usual beat" is not well defined, and every patient-relative feature becomes unreliable.

![Reliability](results/reliability.png)

Threshold 0.975 chosen on DS1 patients only; a recording "fails" when its beat accuracy is below 90%.

| Test set | Recordings | Flagged | Failing | Failing and flagged | Flagged but fine | Accuracy, not flagged | Accuracy, flagged | AUROC |
|---|---|---|---|---|---|---|---|---|
| DS2 | 22 | 8 | 10 | 5 | 3 | 92.0% | 87.1% | 0.80 |
| SVDB | 78 | 50 | 50 | 45 | 5 | 94.1% | 62.0% | 0.94 |
| INCART | 75 | 51 | 54 | 44 | 7 | 91.9% | 65.6% | 0.86 |

On DS2 the flag is weaker, partly because the fusion gate already repaired two of the flagged recordings (117 and 212).
Most SVDB and INCART recordings are flagged: the model, trained on 18 MIT-BIH patients, is simply less sure of itself
on data from other hospitals, and the flag says so instead of hiding it.

**Recording-level burden.** What a reader usually wants first is the ectopic burden (% of beats). The predicted
V burden falls in the right category (<1%, 1–10%, ≥10%) for 77% of DS2 recordings
(86% of the recordings that are not flagged); S burden is much less reliable (32%), because false S
alarms inflate it. Per-recording summaries for DS2 are in [`results/reliability.md`](results/reliability.md).

## What I learned
1. **Normalisation decides S detection.** In the first version, RR intervals were divided by each patient's
   median RR, and both models (random forest and CNN) found only 6–8% of S beats. The cause is record 232: it contains
   1,381 of the 1,836 S beats in DS2, and S beats are the *majority* of its beats, so the "patient median" was itself an
   S rhythm. Local rhythm ratios plus template features raised S sensitivity above 90% with a simple logistic regression.
2. **Patient-relative features cut both ways.** Comparing each beat with the patient's own usual beat is what makes
   the model work on new patients, and it is also what creates most of its errors: when a patient has two conduction
   patterns (record 212), the "usual beat" is a mix, and the minority pattern looks like fusion.
3. **Clinical context beats calibration.** Rescaling the F probability with a factor chosen on DS1 looked good on DS1 but
   did not transfer: it removed every F prediction on all three test sets (F sensitivity
   0% / 0% / 0%). The fusion gate, which encodes one physiological fact and has no tuned
   parameter, kept every true F beat on DS2 and still removed 59% of the false ones.
4. **A model should know when it is on unfamiliar ground.** A one-number consistency check, computed without labels,
   separates recordings the pipeline handles well from recordings it does not, on two databases it never saw.
5. **AF needs more than a rule.** The AF rule helps on DS2, but on SVDB and INCART the gain is small.
6. **One training run is not a result.** Three CNN seeds trained the same way give DS2 macro-F1 between 0.54 and 0.68 and
   S sensitivity between 21% and 56%. Single-run CNN numbers should be read with that spread in mind.
7. **Accuracy is misleading.** On every test set the most accurate model is a CNN, and on every test set it finds
   fewer S beats than the logistic regression.

## Limitations
- The second lead is not the same lead everywhere (V1 on 38 of the 44 MIT-BIH records, V2, V4 or V5 on the others; ECG2 on SVDB),
  and SVDB's first channel (ECG1) is not necessarily MLII. SVDB and INCART are resampled to 360 Hz.
- F (fusion) beats remain hard: 398 training beats, mostly from one patient. The fusion gate only removes false F alarms
  in recordings without predicted V beats, and on SVDB it removes most of the 23 real F beats.
- The reliability threshold is chosen on 22 DS1 recordings, and consistency mixes several causes (noise, conduction
  changes, frequent ectopy) that a clinician would want to see separately.
- The S class in DS2 is dominated by one patient (232), so DS2 S metrics depend heavily on that record.
- INCART's 75 recordings come from 32 patients, so its recordings are not independent.
- AF is handled by a post-hoc rule, not by a joint rhythm-and-beat model. The AF detector is scored on DS2 only,
  because rhythm labels are not used for SVDB and INCART.
- Features use the whole recording (patient template, RR context after the beat), so this is an offline (Holter-style)
  analysis, not a real-time detector.

## Next steps
- [x] Ensemble: logistic regression for S detection + CNN for N/V precision
- [x] Per-patient error analysis (which records fail, and why)
- [x] External validation on a second database (MIT-BIH Supraventricular Arrhythmia Database)
- [x] Automatic R-peak detection instead of annotated peaks
- [x] AF-aware rule: detect the rhythm first, then correct S predictions
- [x] Two-lead input and a third database with a different lead setup (St Petersburg INCART)
- [x] Explain and reduce false F alarms (fusion gate)
- [x] Label-free reliability flag per recording
- [ ] Several templates per patient (one per conduction pattern) instead of a single median beat
- [ ] Separate the reliability flag into causes (noise vs conduction changes) so the warning is actionable
- [ ] Joint rhythm-and-beat model instead of a post-hoc AF rule
- [ ] More seeds and seed ensembling to reduce CNN variance

## Reproduce
```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/download_mitdb.command      # MIT-BIH Arrhythmia Database (~100 MB)
bash scripts/download_svdb.command       # MIT-BIH Supraventricular Arrhythmia Database (~55 MB)
bash scripts/download_incart.command     # St Petersburg INCART (~800 MB, optional)
bash scripts/run_all.command             # features, baselines, detection, CNNs, AF detector, evaluation, analysis, plots
bash scripts/update_results.command      # re-run evaluation + analysis only, with the trained CNNs (1-2 minutes)
```
Downloads are resumable. If INCART is not downloaded, `evaluate.py` skips it. Runs on Apple Silicon (MPS), CUDA or CPU;
training the CNNs (3 seeds each, one and two leads) takes about 5 minutes on an Apple M-series laptop.
The `.command` scripts are plain bash scripts and run on macOS and Linux.

## Project structure
```
src/data.py          beat extraction, RR + template + rhythm features, R-peak detection, DS1/DS2 split, SVDB/INCART loaders
src/model.py         BeatNet (1D CNN over beat + beat-minus-template, fused with the features)
src/af.py            AF detector (leave-patient-out CV on DS1) and the AF-aware rule
src/reliability.py   fusion gate, beat consistency, per-recording summaries
src/metrics.py       AAMI per-class metrics and confusion matrices
baseline.py          logistic regression and gradient boosting (features only)
detection.py         QRS detector evaluation + fully automatic pipeline
train.py             CNN training (3 seeds; --leads 1 or 2)
evaluate.py          ensembles, AF rule, fusion gate, evaluation on DS2 / SVDB / INCART, per-patient analysis
analysis.py          recording-level analysis: false fusion alarms, reliability flag, ectopic burden
scripts/             data downloads, full pipeline, evaluation-only update, comparison plot
results/             metrics, plots, final_results.md and reliability.md (see results/README.md for an index)
```

## Data and references
- Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. *IEEE Eng Med Biol* 20(3):45–50, 2001.
- Greenwald SD. *Improved detection and classification of arrhythmias in noise-corrupted electrocardiograms using
  contextual information.* Thesis, Harvard-MIT Division of Health Sciences and Technology, 1990. (SVDB)
- St Petersburg INCART 12-lead Arrhythmia Database, PhysioNet. doi:10.13026/C2V88N
- Goldberger AL et al. PhysioBank, PhysioToolkit, and PhysioNet. *Circulation* 101(23):e215–e220, 2000.
- Pollard T et al. PhysioNet as a global platform for biomedical research. *Nature Health*, 2026. doi:10.1038/s44360-026-00096-z
- de Chazal P, O'Dwyer M, Reilly RB. Automatic classification of heartbeats using ECG morphology and heartbeat
  interval features. *IEEE Trans Biomed Eng* 51(7):1196–1206, 2004.

## License
Code: MIT (see `LICENSE`). The ECG data are not included; they are distributed by PhysioNet under their own licenses.
This is a research/portfolio project and is not a medical device.

## Author
**Amina El Hasnaouy** · Software Engineering, Sichuan University · [GitHub](https://github.com/AMINA-ELHASNAOUY)
