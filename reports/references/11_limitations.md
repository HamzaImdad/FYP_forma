# Topic 11 — Limitations (to cite honestly)

Sources for FORMA's limitations chapter: single-person detection,
monocular depth ambiguity, occlusion, jitter, frame-rate, lighting,
mobile thermal, and the upper bound set by human inter-rater reliability
on visual movement assessment.

---

## 11.1  Single-person detection limit

### bazarevsky2020blazepose
**Harvard.** Bazarevsky, V. *et al.* (2020) 'BlazePose: On-device Real-time Body Pose Tracking', *arXiv preprint* arXiv:2006.10204.
**Annotation.** Authoritative source for FORMA's 33-keypoint topology, single-person constraint, and mobile-first design philosophy. Cite as the acknowledged limitation: BlazePose is optimised for one person per frame.

### cao2021openpose — Multi-person alternative
**Harvard.** Cao, Z. *et al.* (2021) 'OpenPose: Realtime Multi-Person 2D Pose Estimation Using Part Affinity Fields', *IEEE TPAMI*, 43(1), pp. 172-186.
**DOI.** 10.1109/TPAMI.2019.2929257.
**Annotation.** The main multi-person 2D-pose alternative. OpenPose scales to N people but at significantly higher compute, infeasible in-browser. FORMA's single-person choice is a deliberate tradeoff.

---

## 11.2  Monocular depth ambiguity

### martinez2017simple3dbaseline
**Harvard.** Martinez, J., Hossain, R., Romero, J. and Little, J.J. (2017) 'A simple yet effective baseline for 3D human pose estimation', in *IEEE ICCV*, pp. 2640-2649.
**Annotation.** Even a simple 2D-to-3D lifter carries large residual error, and much of modern 3D-pose error is in the 2D front-end. Monocular 3D pose is inherently approximate — FORMA's angles inherit this error budget.

### liu2022depthambiguity
**Harvard.** Liu, R., Shen, J., Wang, H., Chen, C., Cheung, S.-C. and Asari, V. (2022) 'A Survey on Depth Ambiguity of 3D Human Pose Estimation', *Applied Sciences*, 12(20), 10591.
**DOI.** 10.3390/app122010591.
**Annotation.** Directly surveys the monocular depth-ambiguity problem. Anchors the claim that multiple 3D poses project to the same 2D observation — FORMA's joint angles carry irreducible ambiguity for out-of-plane motion.

### pavllo2019videopose3d
**Harvard.** Pavllo, D., Feichtenhofer, C., Grangier, D. and Auli, M. (2019) '3D Human Pose Estimation in Video with Temporal Convolutions and Semi-Supervised Training', in *IEEE/CVF CVPR*, pp. 7753-7762.
**Annotation.** Temporal context across video frames substantially reduces 3D-pose error vs per-frame. Motivation for FORMA's TemporalSmoother — and a limitation, since FORMA does not run a full temporal-convolutional lifter.

### iskakov2019learnabletriangulation
**Harvard.** Iskakov, K., Burkov, E., Lempitsky, V. and Malkov, Y. (2019) 'Learnable Triangulation of Human Pose', in *IEEE/CVF ICCV*, pp. 7718-7727.
**Annotation.** Multi-view triangulation dramatically improves 3D pose accuracy. The counterfactual — FORMA's single-camera setup forgoes the accuracy achievable with multi-view rigs.

---

## 11.3  Occlusion + visibility

### cheng2019occlusionaware
**Harvard.** Cheng, Y., Yang, B., Wang, B., Yan, W. and Tan, R.T. (2019) 'Occlusion-Aware Networks for 3D Human Pose Estimation in Video', in *IEEE/CVF ICCV*, pp. 723-732.
**Annotation.** Occlusion is the key 3D-pose problem in monocular video. Supports FORMA's known failure cases: lunge (rear leg occluded), plank side-view (far arm hidden) — MediaPipe visibility falls and form score becomes unreliable.

---

## 11.4  Temporal jitter / flicker

### dabral2018structureandmotion
**Harvard.** Dabral, R. *et al.* (2018) 'Learning 3D Human Pose from Structure and Motion', in *ECCV 2018*.
**Annotation.** Adds an anatomically-inspired temporal smoothing network. Supports the claim that raw per-frame pose output is inherently jittery and a temporal smoother is necessary — exactly what FORMA's TemporalSmoother does.

### kanazawa2019humandynamics
**Harvard.** Kanazawa, A., Zhang, J.Y., Felsen, P. and Malik, J. (2019) 'Learning 3D Human Dynamics From Video', in *IEEE/CVF CVPR*, pp. 5614-5623.
**Annotation.** Smooth 3D mesh prediction from video. FORMA's simpler EWMA smoother leaves residual flicker in low-visibility frames.

---

## 11.5  Lighting sensitivity + camera characteristics

### zhang2020mediapipehands
**Harvard.** Zhang, F. *et al.* (2020) 'MediaPipe Hands: On-device Real-time Hand Tracking', *arXiv preprint* arXiv:2006.10214.
**Annotation.** Companion MediaPipe technical note describing on-device design tradeoffs (single-person, lighting sensitivity, crop reliance on previous-frame detection). Google's own documentation of MediaPipe's operating envelope.

### bradski2008learningopencv
**Harvard.** Bradski, G. and Kaehler, A. (2008) *Learning OpenCV: Computer Vision with the OpenCV Library*. Sebastopol, CA: O'Reilly Media.
**Annotation.** Reference text for camera-intrinsics / calibration. Consumer webcams and phone cameras lack calibrated intrinsics, so any 3D reconstruction carries geometric error that FORMA does not correct.

### mlrpose2024posebench
**Harvard.** Liu, S. *et al.* (2024) 'PoseBench: Benchmarking the Robustness of Pose Estimation Models under Corruptions', *arXiv preprint* arXiv:2406.14367.
**Annotation.** Benchmark of 60 pose models under blur, noise, compression, colour loss, lighting, and occlusion corruptions. Quantifies lighting sensitivity and illustrates the type of robustness evaluation FORMA has not yet performed.

---

## 11.6  Mobile GPU thermal throttling

### halpern2016mobilecpu
**Harvard.** Halpern, M., Zhu, Y. and Reddi, V.J. (2016) 'Mobile CPU's rise to power: Quantifying the impact of generational mobile CPU design trends on performance, energy, and user satisfaction', in *IEEE HPCA*, pp. 64-76.
**Annotation.** Quantifies the thermal/power envelope of mobile SoCs across seven generations. Establishes that sustained compute is fundamentally limited on phones — grounds FORMA's acknowledged mobile-GPU thermal-throttling limitation during long sessions.

---

## 11.7  Consumer pose estimation vs marker-based gold standard

### needham2021markerless
**Harvard.** Needham, L. *et al.* (2021) 'The accuracy of several pose estimation methods for 3D joint centre localisation', *Scientific Reports*, 11, 20673.
**Annotation.** Head-to-head vs marker-based gold standard: OpenPose/AlphaPose joint-centre error 16-48 mm for lower limb, max 80 mm+ during running. **Consumer pose estimators cannot match optical motion capture** — so FORMA cannot claim biomechanical-grade measurement.

---

## 11.8  Inter-rater reliability ceiling on visual movement assessment

### bonazza2017fmsreliability
**Harvard.** Bonazza, N.A. *et al.* (2017) 'Reliability, Validity, and Injury Predictive Value of the Functional Movement Screen: A Systematic Review and Meta-analysis', *Am. J. Sports Medicine*, 45(3), pp. 725-732.
**DOI.** 10.1177/0363546516641937.
**Annotation.** Meta-analysis of FMS inter-rater and intra-rater reliability showing "excellent" pooled agreement. The positive benchmark: human experts using a standardised screen agree strongly.

### ressman2019slsmeta — The ceiling
**Harvard.** Ressman, J., Grooten, W.J.A. and Rasmussen Barr, E. (2019) 'Visual assessment of movement quality in the single leg squat test: a review and meta-analysis of inter-rater and intrarater reliability', *BMJ Open Sport & Exercise Medicine*, 5(1), e000541.
**Annotation.** Human inter-rater reliability for visual squat assessment is **only "moderate" (pooled 0.58, range 0.00-0.95)**. The crucial honest-limitation context: FORMA cannot exceed human inter-rater agreement as an upper bound — that's the ceiling on the supervised "ground truth" available for form.
**Suggested quote.** "Pooled results showed a 'moderate' agreement for inter-rater reliability (0.58, 95% CI 0.50 to 0.65)."

---

**Gaps in this section** — Han 2018 frame-rate pose paper, Chen 2017 monocular depth paper, Sarkar 2020 / Kyrollos 2021 pose-BMI-fairness papers, and Pandiyan / Gutierrez mobile thermal papers could not be verified. PoseBench 2024 (above) covers frame-rate robustness; Liu 2022 (above) covers depth ambiguity.
