# Citation Map — FORMA Report Claims → Bibliography Keys

Each row below pairs a load-bearing FORMA claim to the bibliography keys
that back it up. When writing a paragraph in the report, scan the
relevant section here to pick up the canonical citations.

BibTeX keys resolve to entries in `bibliography.bib`. Annotations live in
the corresponding topic Markdown files (`01_…` to `11_…`).

---

## Chapter: Introduction / Motivation

| Claim | Keys |
|---|---|
| WHO and ACSM recommend 2-3 resistance-training sessions per week | bull2020who, garber2011acsm, acsm2021guidelines |
| Poor form is a leading cause of resistance-training injury | keogh2017epi, aasa2017injuries, siewe2011powerlifting, bengtsson2018powerlifting |
| Up to 36% of resistance-training injuries involve the shoulder | kolber2010shoulder, kolber2014recreational, haupt2001upper |
| Supervised training outperforms unsupervised home training | lacroix2017supervised, dishman1982adherence |
| Home fitness rose prominently in consumer-fitness trends | thompson2020trends |
| Fitness apps can produce modest but significant PA gains | schoeppe2016apps, romeo2019smartphoneapps, direito2014bct |
| YouMove is a precedent CV movement-coach system | anderson2013youmove |
| Pose Trainer is the closest direct prior work to FORMA | chenposetrainer2020 |

## Chapter: Literature Review — Pose Estimation

| Claim | Keys |
|---|---|
| MediaPipe BlazePose — 33-landmark topology, real-time on mobile | bazarevsky2020blazepose |
| BlazeFace used as the detector stage of BlazePose's two-stage pipeline | bazarevsky2019blazeface |
| MobileNetV2 backbone enables lightweight inference | sandler2018mobilenetv2 |
| OpenPose — multi-person alternative via Part Affinity Fields | cao2019openpose |
| HRNet — high-resolution representation learning | sun2019hrnet |
| AlphaPose / RMPE — top-down with Symmetric Spatial Transformer | fang2017rmpe |
| Google's earlier top-down PoseNet / MoveNet lineage | papandreou2017accurate |
| DeepPose — first deep-learning pose paper | toshev2014deeppose |
| Stacked Hourglass + CPM — seminal heatmap regression | newell2016hourglass, wei2016cpm |
| COCO 17-keypoint topology + OKS metric | lin2014coco |
| Human3.6M benchmark for MPJPE evaluation | ionescu2014human36m |
| MPII benchmark + PCKh | andriluka2014mpii, yang2013articulated |
| OKS error-diagnosis taxonomy | ronchi2017benchmarking |

## Chapter: Method — Feature Extraction + Joint-Angle Rules

### Squat detector (`src/classification/squat_detector.py`)
| Claim | Keys |
|---|---|
| Knee biomechanics + parallel-depth thresholds | escamilla2001knee, caterisano2002squatdepth, hartmann2012squat |
| High-bar vs low-bar torso-angle ranges | hales2009kinematic |
| Functional-deficit movement-screen template | myer2014backsquat, schoenfeld2010squat |
| Ankle dorsiflexion ROM predicts squat depth | kim2015ankle |
| Knee valgus (FPPA) as ACL risk | hewett2005valgus, herrington2010valgus |

### Deadlift detector
| Claim | Keys |
|---|---|
| Deadlift kinematics — upright torso + hip hinge | mcguigan1996deadlift, hales2009kinematic |
| Lumbar compressive loads up to 17 kN | cholewicki1991lumbar |
| Deadlift-specific lower-back / meniscal injury pattern | bengtsson2018powerlifting |

### Push-up / pull-up / plank / side-plank / crunch / lateral-raise / curl
| Claim | Keys |
|---|---|
| Push-up hand-position EMG → shoulder-flare rule | cogley2005pushup |
| Pull-up elbow-angle + chin-above-bar rule | youdas2010pullup, dickie2017pullup |
| Plank posterior-pelvic-tilt + hip-sag rules | mcgill2010core, schoenfeld2014plank |
| Side-plank hip-lift + body-line rule | ekstrom2007core, boren2011gluteal |
| Crunch trunk-flexion thresholds | juker1998abdominal, escamilla2006abdominal |
| Lateral-raise shoulder-abduction + "no above-shoulder" | botton2013deltoid, coratella2020lateralraise, kolber2014recreational |
| Bicep curl anti-swing (shoulder-position EMG) | oliveira2009bicepscurl |
| Bench-press sticking region + shoulder-flare (retired exercise) | elliott1989benchpress, green2007bench |

### Rep counting + symmetry
| Claim | Keys |
|---|---|
| Rep counting from time-series peaks (vs IMU / DTW) | morris2014recofit, pernek2013rep |
| Left-right symmetry matters for performance | bishop2018asymmetry |

## Chapter: Method — ML Classifiers

| Claim | Keys |
|---|---|
| Random Forest baseline | breiman2001random |
| SVM baseline (kernel + margin) | cortes1995support, burges1998tutorial |
| LSTM + BiLSTM recurrent backbones | hochreiter1997lstm, schuster1997bidirectional |
| CNN + time-series hybrid (Conv1D front-end) | karim2017lstm, wang2015imaging |
| DeepConvLSTM — direct architectural template | ordonez2016deep |
| CNN vs LSTM vs BiLSTM benchmarks for HAR | hammerla2016deep |
| Deep-learning HAR taxonomy | bulling2014tutorial, wang2019deeplearning |
| Additive + multiplicative attention | bahdanau2015neural, luong2015effective |
| Self-attention / Transformer | vaswani2017attention |
| Skeleton-based action recognition — ST-GCN and PoseConv3D | yan2018stgcn, duan2022poseconv3d |
| Action quality assessment — continuous form scoring | parmar2017learning, parmar2019what |
| Interpretable models preferred for high-stakes feedback | rudin2019stop, doshi2017towards |
| Rule-based detector as primary path (explanability) | rudin2019stop |

## Chapter: Method — Training Pipeline

| Claim | Keys |
|---|---|
| Video-level splits prevent data leakage | kaufman2012leakage, kapoor2023leakage |
| Class imbalance: Focal Loss, SMOTE, undersampling | lin2017focal, chawla2002smote, kubat1997addressing, he2009learning |
| Label smoothing (ε = 0.1) | szegedy2016rethinking, muller2019when |
| Mixup augmentation | zhang2018mixup |
| Stochastic Weight Averaging | izmailov2018swa |
| Cosine annealing with warm restarts + linear warmup | loshchilov2017sgdr, goyal2017accurate, he2019bag |
| Adam / AdamW optimisers | kingma2015adam, loshchilov2019decoupled |
| Dropout + BatchNorm | srivastava2014dropout, ioffe2015batch |
| Mixed-precision training (AMP) | micikevicius2018mixed |
| Gradient clipping for RNNs | pascanu2013difficulty |
| Early stopping on val F1 | prechelt2012early |
| Pose-landmark augmentations | rao2021augmented, gong2021poseaug |
| Cross-exercise pretraining | carreira2017i3d |
| F1, precision, recall under imbalance | powers2011evaluation, saito2015precision |
| F1-maximising threshold search on validation | lipton2014optimal |

## Chapter: System Design — Real-Time Transport

| Claim | Keys |
|---|---|
| WebSocket protocol (FORMA uses Socket.IO) | fette2011websocket, pimentel2012websocket |
| WebRTC reference for comparison | alvestrand2021rfc8825, jansen2018webrtc |
| WebAssembly + WebGL for browser ML inference | haas2017webassembly, smilkov2019tensorflowjs |
| MediaPipe framework (dataflow graph, calculator abstraction) | lugaresi2019mediapipe |
| H.264/AVC — codec comparison for session recording | wiegand2003h264 |
| Mobile inference performance ceiling | ignatov2019aibenchmark |
| Web-networking latency budgets | grigorik2013hpbn |
| 0.1 s / 1 s / 10 s response-time rules | nielsen1993usability, miller1968response |
| Model Human Processor — perceptual fusion threshold | card1983psychology |
| Fitts' information-theoretic motor bound | fitts1954information, seow2005information |

## Chapter: UX / HCI — Feedback Design

| Claim | Keys |
|---|---|
| Knowledge of Results / Knowledge of Performance | schmidt2019motor, salmoni1984knowledge |
| Guidance hypothesis — frequent KR degrades retention | salmoni1984knowledge, winstein1990reduced, schmidt1991frequent |
| Bandwidth feedback — silent-when-correct | lee1990bandwidth, schmidt1991frequent |
| Continuous concurrent feedback degrades learning | schmidt1997continuous |
| Caveats for very beginners | wulf2004understanding |
| Multimodal (voice + visual) feedback | sigrist2013augmented |
| Cognitive Load Theory — one error at a time | sweller1988cognitive, vanmerrienboer2005cognitive, mayer2003nine |
| Qualitative Activity Recognition (QAR) | velloso2013qualitative |
| AR-mirror movement training precedent | anderson2013youmove |
| Exertion interfaces framing | mueller2003exertion, mueller2011designing |
| Pre-attentive colour coding (red/green) | healey2012attention, ware2020information |
| Gamification motivation | deterding2011gamification, hamari2014does, wouters2013metaanalysis |
| Self-determination theory (badges respect autonomy) | ryan2000sdt, deci1985intrinsic |
| Gamification-for-health efficacy | cugelman2013gamification, johnson2016gamification |
| Motivational interviewing — "patient trainer" tone | miller2012motivational |
| Forward-model / Bayesian motor control | wolpert2000computational |

## Chapter: Web Product Layer

| Claim | Keys |
|---|---|
| JWT auth + signed tokens | jones2015jwt, jones2015jws, jones2015jwe |
| Cookie HttpOnly / Secure / SameSite | barth2011cookies, west2020samesite, owasp2024cheatsheet |
| LLM tool use / function calling | schick2023toolformer, yao2023react |
| Retrieval-Augmented Generation | lewis2020rag, borgeaud2022retro, guu2020realm, izacard2023atlas |
| Server-Sent Events streaming | hickson2015sse |
| Rate limiting via leaky / token bucket | turner1986leakybucket, shenker1995fundamental |
| Progressive overload (plan engine) | kraemer2004fundamentals, schoenfeld2017doseresponse, schoenfeld2019volume |
| RPE / RIR autoregulation | helms2016rpe, zourdos2016rpe |
| Prompt-injection defence | perez2022ignore, greshake2023indirect |

## Chapter: UI / Design

| Claim | Keys |
|---|---|
| WCAG 2.1 contrast + dark-mode audit | wcag21 |
| Dark-mode HCI evidence | pedersen2020darkmode, erickson2020dark |
| `prefers-reduced-motion` respect | mediaqueries5, ahmadian2023vestibular |
| Typography + grid | bringhurst2005, muller1981grid, lupton2010thinking |
| Micro-interactions + scrollytelling | saffer2013microinteractions, seyser2018scrollytelling |
| Animation for attention, not decoration | harley2014animation |
| Atomic Design component hierarchy | frost2016atomic |
| Progressive disclosure | nielsen2006progressive |
| Perception thresholds for skeleton-loader timing | nielsen1993usability, card1991info |
| "Overview first, zoom and filter, details on demand" | shneiderman1996eyes |
| Emotional / hedonic design | norman2004emotional, hassenzahl2006hedonic |
| System Usability Scale | brooke1996sus |

## Chapter: Ethics / SE

| Claim | Keys |
|---|---|
| Algorithmic bias in commercial face + object systems | buolamwini2018gendershades, raji2019actionableauditing, wilson2019predictiveinequity |
| Fairness benchmark FACET | gustafson2023facet |
| Model Cards + Datasheets | mitchell2019modelcards, gebru2021datasheets |
| FORMA sits on the general-wellness side of FDA SaMD | imdrf2013samdkeydefs, fda2022mma, fda2019precert |
| GDPR — data-minimisation + Article 9 biometrics | eu2016gdpr, voigt2017gdprpracticalguide, ico2024biometric |
| Critical-biometrics framing | gates2011biometricfuture |
| ACM Code of Ethics + IEEE EAD | acm2018ethics, ieee2019ead |
| ISO/IEC 25010 quality model + ISO/IEC/IEEE 12207 lifecycle | isoiec25010_2011, isoiecieee12207_2017 |
| Health-app engagement / trust | torous2018mentalhealthapps, lupton2014digitalhealth |
| Bioethics four-principles + Nuffield consent | beauchamp2019biomedicalethics, nuffield2015biologicaldata |
| Digital technology + wellbeing effect size is small | orben2019adolescentwellbeing |
| Edge computing — on-device inference preference | shi2016edgecomputing, mach2017mecsurvey |

## Chapter: Limitations (honest)

| Claim | Keys |
|---|---|
| Single-person detection only (BlazePose by design) | bazarevsky2020blazepose |
| Multi-person alternative (OpenPose) is heavier | cao2019openpose |
| Monocular depth ambiguity — irreducible for out-of-plane motion | martinez2017simple3dbaseline, liu2022depthambiguity |
| Multi-view would boost accuracy | iskakov2019learnabletriangulation |
| Occlusion is the key failure mode | cheng2019occlusionaware |
| Temporal jitter / flicker requires smoothing | pavllo2019videopose3d, dabral2018structureandmotion, kanazawa2019humandynamics |
| Lighting sensitivity of MediaPipe | zhang2020mediapipehands, mlrpose2024posebench |
| Consumer webcams lack calibrated intrinsics | bradski2008learningopencv |
| Mobile GPU thermal throttling | halpern2016mobilecpu |
| Consumer pose estimators ≠ marker-based gold standard | needham2021markerless |
| Inter-rater reliability of visual squat assessment is moderate — caps supervised-label ceiling | ressman2019slsmeta, bonazza2017fmsreliability |

---

## Gaps — to investigate in a later session

The research agents flagged the following claims as under-sourced or
where a specific citation from the original brief could not be verified.
These should be revisited before final submission:

- **Token-budgeting for LLM serving.** No verified peer-reviewed paper. Candidate: **Chen et al. 2023 "FrugalGPT"** (arXiv:2305.05176) — verify before citing.
- **LLM streaming UX.** No peer-reviewed source located. Fall back to NNG perceived-performance practitioner articles (`harley2014animation`, `nielsen2006progressive`).
- **Hero video in product marketing.** No peer-reviewed source. Emotional-design (`norman2004emotional`) + hedonic quality (`hassenzahl2006hedonic`) cover the broader argument.
- **Logistic regression + pose classification.** No canonical paper. Suggested fall-back: Hosmer, Lemeshow & Sturdivant (2013) *Applied Logistic Regression* textbook.
- **Lehman 2005 tricep-dip EMG and Signorile 2002 biceps curl EMG.** Could not verify; dropped to avoid misattribution.
- **Phone camera FOV / autofocus drift specifically.** No dedicated peer-reviewed paper; `bradski2008learningopencv` + `halpern2016mobilecpu` together cover the technical envelope.
- **Push-up-specific injury epidemiology.** No dedicated study; implicit in `keogh2017epi`. Consider Kerr, Collins & Comstock 2010 (AJSM, DOI 10.1177/0363546509351560).
- **Novice-vs-experienced kinematic differences.** `myer2014backsquat` supplies the framework; no empirical comparison paper verified.
- **Gross et al. 1993 and Ogata 2019 CV-coach.** Could not be verified — dropped.
- **Froomkin & Arencibia 2020 Algorithmic Accountability Act.** Not located — drop or cite the bill directly.
- **Pandiyan ISPASS / Gutierrez & Lopez 2014 mobile thermal.** `halpern2016mobilecpu` is the verified substitute.

Next session, these can be closed by (a) substituting verified replacements from the bibliography, or (b) re-running targeted Scholar searches with more specific queries (e.g. author + exact phrase).
