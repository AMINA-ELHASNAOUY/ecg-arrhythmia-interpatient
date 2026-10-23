# ECG Arrhythmia Classification — Inter-Patient Evaluation on MIT-BIH

Heartbeat classification (AAMI classes **N, S, V, F**) evaluated the way it would be used in practice:
**the models are tested only on patients they have never seen.** Training uses 18 MIT-BIH patients; testing uses
22 other MIT-BIH patients and two external databases recorded elsewhere. No test set is used for training, tuning or
model selection.

**Highlights**
- **Three test sets, no tuning on any of them:** 22 unseen MIT-BIH patients (DS2), the MIT-BIH Supraventricular Arrhythmia
  Database (78 recordings) and the St Petersburg INCART database (75 recordings from 32 patients, different hospital and leads).
- **Patient-relative features** lift S-beat sensitivity from 6–8% to 92% with a plain logistic regression.
- **An AF-aware rule** (a rhythm detector trained on MIT-BIH training patients only) removes 41% of false S alarms on DS2
  (3,479 → 2,045) and raises macro-F1 from 0.581 to 0.608.
- **Fully automatic pipeline:** 99.2% QRS detection, with classification on detected beats at the level of expert-annotated beats.
- **An honest result:** no single model wins. The logistic regression finds the most S beats; the CNNs have the best accuracy and
  macro-F1 but miss most S beats, and their S sensitivity ranges from 21% to 56% between training seeds.

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
  leave-patient-out cross-validation on DS1. On DS2 the detector reaches AF sensitivity 82.6%, +P 92.8%, specificity 98.6%.
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
| Ensemble | 82.3% | 0.581 | 82.1 / **99.1** | 91.1 / 32.5 | 84.1 / 92.2 | 45.1 / 3.6 |
| **Ensemble + AF-aware rule** | 84.8% | 0.608 | 85.2 / 98.6 | 85.0 / 43.3 | 84.1 / 92.2 | 45.1 / 3.6 |
| Two-lead CNN | **95.5%** | **0.616** | **98.3** / 97.0 | 43.4 / **61.6** | **98.6** / **95.5** | 0.5 / 0.9 |

**Seed variance (single CNNs, mean ± std over 3 seeds on DS2):** macro-F1 0.614 ± 0.057 (one lead) and
0.601 ± 0.044 (two leads); S sensitivity 38.5 ± 14.3% and 42.3 ± 20.5% (`results/cnn*_seeds.json`).

### External test sets (never used for training or tuning)
| Model | SVDB Acc | SVDB Macro-F1 | SVDB S Se / +P | INCART Acc | INCART Macro-F1 | INCART S Se / +P |
|---|---|---|---|---|---|---|
| Logistic regression | 67.6% | 0.498 | **68.6** / 47.3 | 68.6% | 0.449 | **95.7** / 10.9 |
| CNN | 91.9% | 0.525 | 23.1 / 55.7 | 95.5% | 0.626 | 83.2 / 40.8 |
| Ensemble + AF-aware rule | 70.0% | 0.507 | 64.3 / 50.9 | 71.1% | 0.459 | 95.6 / 11.6 |
| Two-lead CNN | **92.2%** | **0.528** | 27.1 / **59.1** | **97.8%** | **0.674** | 61.4 / **73.6** |

SVDB contains only 23 F beats and INCART 219, so N, V and F columns are in `results/final_results.md`.

### Fully automatic pipeline (no expert beat positions)
R peaks found by the XQRS detector (wfdb) instead of the annotations; labels come from the nearest annotation within 150 ms.

| | Result |
|---|---|
| QRS detection, 44 MIT-BIH records (100,733 beats) | Se **99.17%**, +P **99.29%** |
| Logistic regression on detected beats, DS2 | macro-F1 **0.587** · S 87.5 / 30.3 · V 86.7 / 95.5 |

The classifier performs at the same level on automatically detected beats as on expert-annotated ones.

### Where the S errors come from
![Per-patient errors](results/per_patient_errors.png)

Before the AF-aware rule, most false S alarms on DS2 come from the 5 test patients with atrial fibrillation or flutter
(records 202, 210, 219, 221, 222): during AF the rhythm is irregular, so normal beats often arrive "early", which is
exactly what an S beat looks like to an RR-based model. The rule removes 1,434 false S alarms at the cost of 111 missed
S beats. The per-patient table is in [`results/final_results.md`](results/final_results.md).

## What I learned
1. **Normalisation decides S detection.** In the first version, RR intervals were divided by each patient's
   median RR, and both models (random forest and CNN) found only 6–8% of S beats. The cause is record 232: it contains
   1,381 of the 1,836 S beats in DS2, and S beats are the *majority* of its beats, so the "patient median" was itself an
   S rhythm. Local rhythm ratios plus template features raised S sensitivity above 90% with a simple logistic regression.
2. **Different models fail in different ways.** The logistic regression finds S beats (92% on DS2, 96% on INCART) but
   raises many false alarms; the CNN is precise on N and V but misses most S beats. The ensemble weight chosen on the
   hold-out patients (0.9 on the logistic regression) keeps the ensemble close to the logistic regression.
3. **Most remaining errors of the feature models are false fusion (F) alarms.** For the ensemble + AF rule, 63% of all
   errors on DS2, 77% on SVDB and 70% on INCART are beats wrongly called F. Records 117 and 212 contain no F beats at all,
   yet false F alarms account for 857 of 859 and 927 of 952 of their errors. With only 398 F training beats, class-balanced
   weights push the logistic regression towards F, and the ensemble weight is chosen on N/S/V only, so these errors are
   never penalised during model selection. This is the clearest thing to fix next.
4. **AF needs more than a rule.** The rule helps on DS2 (macro-F1 0.581 → 0.608), but after it the AF patients still cause
   44% of false S alarms (895 of 2,045). On SVDB and INCART the gain is small.
5. **One training run is not a result.** Three CNN seeds trained the same way give DS2 macro-F1 between 0.54 and 0.68 and
   S sensitivity between 21% and 56%. Single-run CNN numbers should be read with that spread in mind.
6. **A second lead helps on new data, not on DS2.** On DS2 the two-lead CNN is within seed noise of the one-lead CNN;
   on INCART it raises macro-F1 from 0.626 to 0.674.
7. **Accuracy is misleading.** On every test set the most accurate model is a CNN, and on every test set it finds
   fewer S beats than the logistic regression.

## Limitations
- The second lead is not the same lead everywhere (V1 on 38 of the 44 MIT-BIH records, V2, V4 or V5 on the others; ECG2 on SVDB),
  and SVDB's first channel (ECG1) is not necessarily MLII. SVDB and INCART are resampled to 360 Hz.
- F (fusion) beats remain unsolved (398 training beats, mostly from one patient), and false F alarms dominate the errors
  of the feature models (see above).
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
- [ ] Reduce false F alarms: include F in model selection, or calibrate the F class
- [ ] Joint rhythm-and-beat model instead of a post-hoc AF rule
- [ ] More seeds and seed ensembling to reduce CNN variance

## Reproduce
```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/download_mitdb.command      # MIT-BIH Arrhythmia Database (~100 MB)
bash scripts/download_svdb.command       # MIT-BIH Supraventricular Arrhythmia Database (~55 MB)
bash scripts/download_incart.command     # St Petersburg INCART (~800 MB, optional)
bash scripts/run_all.command             # features, baselines, detection, CNNs, AF detector, evaluation, plots
```
Downloads are resumable. If INCART is not downloaded, `evaluate.py` skips it. Runs on Apple Silicon (MPS), CUDA or CPU;
training the CNNs (3 seeds each, one and two leads) takes about 5 minutes on an Apple M-series laptop.
The `.command` scripts are plain bash scripts and run on macOS and Linux.

## Project structure
```
src/data.py          beat extraction, RR + template + rhythm features, R-peak detection, DS1/DS2 split, SVDB/INCART loaders
src/model.py         BeatNet (1D CNN over beat + beat-minus-template, fused with the features)
src/af.py            AF detector (leave-patient-out CV on DS1) and the AF-aware rule
src/metrics.py       AAMI per-class metrics and confusion matrices
baseline.py          logistic regression and gradient boosting (features only)
detection.py         QRS detector evaluation + fully automatic pipeline
train.py             CNN training (3 seeds; --leads 1 or 2)
evaluate.py          ensembles, AF rule, evaluation on DS2 / SVDB / INCART, per-patient analysis
scripts/             data downloads, full pipeline, comparison plot
results/             metrics, plots and final_results.md (see results/README.md for an index)
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
