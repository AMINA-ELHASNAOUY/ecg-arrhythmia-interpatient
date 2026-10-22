# ECG Arrhythmia Classification — Inter-Patient Evaluation on MIT-BIH

Heartbeat classification (AAMI classes **N, S, V, F**) evaluated the way it would be used in practice:
**the model is tested only on patients it has never seen**, first on 22 held-out MIT-BIH patients and then on
78 patients from a second database. Training and test beats never come from the same person.

**Highlights:** patient-relative features lift S-beat sensitivity from 6–8% to 92% · a logistic-regression + CNN ensemble
gives the best macro-F1 on both test sets · a fully automatic pipeline (99.2% QRS detection) matches expert-annotated
beats · 71% of the remaining false S alarms trace back to atrial fibrillation.

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
| DS1 hold-out (4 patients) | 108, 118, 207, 223 | early stopping only |
| DS2 (22 patients) | 100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234 | final test, used once |

Paced records (102, 104, 107, 217) are excluded and Q beats are dropped, following AAMI EC57.

## Method
**Beats.** One ECG lead (MLII on MIT-BIH), resampled to 360 Hz, 0.5–40 Hz band-pass filter,
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
- **Ensemble:** weighted average of the two models' probabilities. The weight (0.75 logistic regression /
  0.25 CNN) is chosen on the DS1 validation patients, never on test data.

**Two test sets.**
- **DS2:** 22 unseen MIT-BIH patients (49,661 beats).
- **SVDB:** the MIT-BIH Supraventricular Arrhythmia Database, 78 patients recorded separately at 128 Hz
  (184,347 beats, 12,194 S beats). Nothing from SVDB is used for training or tuning.

## Results

### DS2 — 22 unseen MIT-BIH patients
| Model | Accuracy | Macro-F1 | N Se / +P | S Se / +P | V Se / +P | F Se / +P |
|---|---|---|---|---|---|---|
| Logistic regression (features) | 81.2% | 0.574 | 80.9 / **99.1** | **92.0** / 31.0 | 82.8 / 92.1 | **48.4** / 3.6 |
| CNN (waveform + features) | **94.3%** | 0.562 | **98.0** / 96.3 | 18.5 / 30.9 | **97.3** / 89.6 | 7.0 / **30.7** |
| **Ensemble** | 84.8% | **0.596** | 85.0 / 98.8 | 87.0 / **35.8** | 87.2 / **92.3** | 38.9 / 3.6 |

### SVDB — 78 patients from a second database (external test)
| Model | Accuracy | Macro-F1 | N Se / +P | S Se / +P | V Se / +P |
|---|---|---|---|---|---|
| Logistic regression (features) | 67.6% | 0.499 | 68.0 / **98.9** | **68.6** / 47.4 | 60.4 / 65.3 |
| CNN (waveform + features) | **90.3%** | 0.499 | **96.2** / 94.3 | 17.6 / **54.5** | **84.0** / **71.9** |
| **Ensemble** | 73.7% | **0.519** | 74.9 / 98.1 | 64.4 / 50.5 | 65.5 / 66.4 |

Se = sensitivity, +P = positive predictivity, in %. SVDB contains only 23 F beats, so F is left out of that table
(it is in `results/svdb_*.json`).

### Fully automatic pipeline (no expert beat positions)
R peaks found by the XQRS detector (wfdb) instead of the annotations; labels come from the nearest annotation within 150 ms.

| | Result |
|---|---|
| QRS detection, 44 MIT-BIH records (100,733 beats) | Se **99.17%**, +P **99.29%** |
| Logistic regression on detected beats, DS2 | macro-F1 **0.587** · S 87.5 / 30.3 · V 86.7 / 95.5 |

The classifier performs at the same level on automatically detected beats as on expert-annotated ones.

### Where the errors come from
![Per-patient errors](results/per_patient_errors.png)

**71% of all false S alarms on DS2 (2,030 of 2,866) come from the 5 test patients with atrial fibrillation or
flutter** (records 202, 210, 219, 221, 222). During AF the rhythm is irregular, so normal beats often arrive
"early" — exactly what an S beat looks like to an RR-based model. The full per-patient table is in
[`results/final_results.md`](results/final_results.md).

## What I learned
1. **Normalisation decides S detection.** In the first version, RR intervals were divided by each patient's
   median RR, and both models (random forest and CNN) found only 6–8% of S beats. The cause is record 232: it contains 1,381 of the 1,836 S beats in DS2,
   and S beats are the *majority* of its beats, so the "patient median" was itself an S rhythm. Local rhythm ratios
   plus template features raised S sensitivity above 90% with a simple logistic regression.
2. **Model capacity is not the bottleneck.** A logistic regression on 11 features finds S beats far better than
   the CNN. The CNN is better at N and V. Combining them gives the best macro-F1 on both test sets.
3. **The next problem is atrial fibrillation, not S beats.** Most false S alarms happen during AF, where irregular
   rhythm breaks the "early beat" assumption. Recognising the rhythm first (AF or not) is the clear next step.
4. **A second database is humbling.** On SVDB every model drops (ensemble macro-F1 0.596 → 0.519), and the CNN's
   90% accuracy hides that it misses 82% of S beats. Testing on data from a different source is the only way to see this.
5. **Accuracy is misleading.** The model with the highest accuracy is the worst at finding S beats on both test sets.

## Limitations
- Single lead. SVDB uses its first channel (ECG1), which is not necessarily MLII, and is resampled from 128 Hz.
- F (fusion) beats remain unsolved: 398 training beats, mostly from one patient.
- The S class in DS2 is dominated by one patient (232), so DS2 S metrics depend heavily on that record.
- No rhythm-level (AF) modelling yet.

## Next steps
- [x] Ensemble: logistic regression for S detection + CNN for N/V precision
- [x] Per-patient error analysis (which records fail, and why)
- [x] External validation on a second database (MIT-BIH Supraventricular Arrhythmia Database)
- [x] Automatic R-peak detection instead of annotated peaks
- [ ] AF-aware model: detect the rhythm first, then classify beats
- [ ] Multi-lead input and a database with a different lead setup (e.g. St Petersburg INCART)

## Reproduce
```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/download_mitdb.command      # MIT-BIH Arrhythmia Database (~100 MB), resumable
bash scripts/download_svdb.command       # MIT-BIH Supraventricular Arrhythmia Database (~55 MB), resumable
bash scripts/run_all.command             # features, baselines, automatic detection, CNN, ensemble, plots
```
Runs on Apple Silicon (MPS), CUDA or CPU. Training the CNN (3 seeds) takes about 3.5 minutes on an Apple M-series laptop.

## Project structure
```
src/data.py          beat extraction, RR + template features, R-peak detection, DS1/DS2 split, SVDB loader
src/model.py         BeatNet (2-channel 1D CNN + feature branch)
src/metrics.py       AAMI per-class metrics and confusion matrices
baseline.py          logistic regression and gradient boosting (features only)
detection.py         QRS detector evaluation + fully automatic pipeline
train.py             CNN training (3 seeds)
evaluate.py          ensemble, DS2 + SVDB evaluation, per-patient analysis
scripts/             data downloads, full pipeline, comparison plot
results/             metrics (JSON), confusion matrices, plots, final_results.md
```

## Data and reference
- Greenwald SD. *Improved detection and classification of arrhythmias in noise-corrupted electrocardiograms using
  contextual information.* Thesis, Harvard-MIT Division of Health Sciences and Technology, 1990. (SVDB)
- Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. *IEEE Eng Med Biol* 20(3):45–50, 2001.
- Goldberger AL et al. PhysioBank, PhysioToolkit, and PhysioNet. *Circulation* 101(23):e215–e220, 2000.
- de Chazal P, O'Dwyer M, Reilly RB. Automatic classification of heartbeats using ECG morphology and heartbeat
  interval features. *IEEE Trans Biomed Eng* 51(7):1196–1206, 2004.

## Author
**Amina El Hasnaouy** · Software Engineering, Sichuan University · [GitHub](https://github.com/AMINA-ELHASNAOUY)
