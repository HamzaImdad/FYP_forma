# Topic 2 — Joint-Angle Form Evaluation (Biomechanics)

Sources for FORMA's exercise-specific form rules. Many of these are already
cited as in-code comments in `src/classification/*_detector.py` (e.g.
Escamilla 2001, Herrington 2010, Hewett 2005). This file is the formal
Harvard record for the report.

---

## 2.1  Squat

### escamilla2001knee
**Harvard.** Escamilla, R.F. (2001) 'Knee biomechanics of the dynamic squat exercise', *Medicine & Science in Sports & Exercise*, 33(1), pp. 127-141.
**Annotation.** Authoritative review of tibiofemoral shear/compressive forces, patellofemoral forces, and knee muscle activity across knee flexion angles during the squat. Directly justifies FORMA's SquatDetector knee-angle thresholds.
**Suggested quote.** "Low to moderate posterior shear forces, restrained primarily by the posterior cruciate ligament (PCL), were generated throughout the squat for all knee flexion angles."

### caterisano2002squatdepth
**Harvard.** Caterisano, A. *et al.* (2002) 'The effect of back squat depth on the EMG activity of 4 superficial hip and thigh muscles', *J. Strength Cond. Res.*, 16(3), pp. 428-432.
**Annotation.** Gluteus maximus EMG activation increases with squat depth (partial < parallel < full). Justifies FORMA's depth-as-quality rule.

### hartmann2012squat
**Harvard.** Hartmann, H. *et al.* (2012) 'Influence of squatting depth on jumping performance', *J. Strength Cond. Res.*, 26(12), pp. 3243-3261.
**Annotation.** Deep front and back squats produce greater jump-performance transfer than quarter squats. Supports depth-based form scoring.

### hales2009kinematic — Squat vs deadlift kinematics
**Harvard.** Hales, M.E., Johnson, B.F. and Johnson, J.T. (2009) 'Kinematic analysis of the powerlifting style squat and the conventional deadlift during competition', *J. Strength Cond. Res.*, 23(9), pp. 2574-2580.
**Annotation.** 3D cinematography of hip/knee/ankle angles during squat vs deadlift at national competition. Justifies FORMA's separate detectors for squat (knee-dominant) and deadlift (hip-dominant hinge).

### myer2014backsquat — Functional-deficit assessment
**Harvard.** Myer, G.D. *et al.* (2014) 'The back squat: a proposed assessment of functional deficits and technical factors that limit performance', *Strength and Conditioning Journal*, 36(6), pp. 4-27.
**DOI.** 10.1519/SSC.0000000000000103.
**Annotation.** Presents a dynamic movement screen for the back squat — breaks technique into observable deficits (knee valgus, forward torso lean, heel rise, asymmetric depth). Directly parallels FORMA's per-joint feedback taxonomy.

### schoenfeld2010squat — Squat kinetics/kinematics
**Harvard.** Schoenfeld, B.J. (2010) 'Squatting kinematics and kinetics and their application to exercise performance', *J. Strength Cond. Res.*, 24(12), pp. 3497-3506.
**DOI.** 10.1519/JSC.0b013e3181bac2d7.
**Annotation.** Ankle/knee/hip/spine biomechanics during the squat with guidance on depth, knee tracking, and torso angle. Primary citation behind FORMA's SquatDetector checks.

### kim2015ankle — Ankle dorsiflexion and squat depth
**Harvard.** Kim, S.-H. *et al.* (2015) 'Lower extremity strength and the range of motion in relation to squat depth', *J. Human Kinetics*, 45(1), pp. 59-69.
**Annotation.** Ankle dorsiflexion ROM was a significant predictor of squat depth. Justifies FORMA's "ankles tight" heuristic and explains why depth may be limited by mobility rather than effort.

---

## 2.2  Knee valgus / ACL risk

### hewett2005valgus
**Harvard.** Hewett, T.E. *et al.* (2005) 'Biomechanical measures of neuromuscular control and valgus loading of the knee predict ACL injury risk in female athletes: a prospective study', *Am. J. Sports Medicine*, 33(4), pp. 492-501.
**Annotation.** Prospective study: knee valgus loading predicts ACL injury (73% specificity, 78% sensitivity) in 205 female athletes. Gold-standard citation for FORMA's knee-valgus warning in SquatDetector and LungeDetector.

### herrington2010valgus — Normative FPPA values
**Harvard.** Herrington, L. and Munro, A. (2010) 'Drop jump landing knee valgus angle; normative data in a physically active population', *Physical Therapy in Sport*, 11(2), pp. 56-59.
**DOI.** 10.1016/j.ptsp.2009.11.004.
**Annotation.** Provides normative knee valgus angles: 5-12° for females, 1-9° for males during 2D drop-jump landing. Empirical basis for FORMA's valgus-threshold choice (>13° flagged incorrect).

---

## 2.3  Deadlift

### mcguigan1996deadlift
**Harvard.** McGuigan, M.R.M. and Wilson, B.D. (1996) 'Biomechanical analysis of the deadlift', *J. Strength Cond. Res.*, 10(4), pp. 250-255.
**Annotation.** Kinematic comparison of sumo vs conventional deadlift from competition footage. Documents the upright-torso posture and shortened bar path. Cite for FORMA's DeadliftDetector hip-hinge and back-angle rules.

### cholewicki1991lumbar
**Harvard.** Cholewicki, J., McGill, S.M. and Norman, R.W. (1991) 'Lumbar spine loads during the lifting of extremely heavy weights', *Med. Sci. Sports Exerc.*, 23(10), pp. 1179-1186.
**Annotation.** Estimates L4/L5 compressive loads up to ~17 kN during competitive deadlifts. Canonical citation for the injury-prevention rationale behind FORMA's back-straightness check.

---

## 2.4  Push-up

### cogley2005pushup
**Harvard.** Cogley, R.M. *et al.* (2005) 'Comparison of muscle activation using various hand positions during the push-up exercise', *J. Strength Cond. Res.*, 19(3), pp. 628-633.
**Annotation.** Narrow-base push-ups elicit greater pectoralis and triceps EMG than wide-base. Supports FORMA's PushUpDetector shoulder-flare rule.

---

## 2.5  Pull-up

### youdas2010pullup
**Harvard.** Youdas, J.W. *et al.* (2010) 'Surface electromyographic activation patterns and elbow joint motion during a pull-up, chin-up, or Perfect-Pullup rotational exercise', *J. Strength Cond. Res.*, 24(12), pp. 3404-3414.
**Annotation.** Latissimus dorsi EMG ~117-130% MVIC during pull-ups, biceps 78-96%. Biomechanical basis for FORMA's PullUpDetector elbow-angle range and chin-above-bar rule.

### dickie2017pullup
**Harvard.** Dickie, J.A. *et al.* (2017) 'Electromyographic analysis of muscle activation during pull-up variations', *J. Electromyogr. Kinesiol.*, 32, pp. 30-36.
**Annotation.** Compares grip variations. Concentric-phase peaks exceed eccentric for forearm and biceps — supports FORMA's rep-phase state machine (GOING_UP concentric, GOING_DOWN eccentric).

---

## 2.6  Plank + side-plank + core

### mcgill2010core
**Harvard.** McGill, S.M. (2010) 'Core training: evidence translating to better performance and injury prevention', *Strength Cond. J.*, 32(3), pp. 33-46.
**Annotation.** McGill's canonical argument for neutral-spine core exercises (front plank, side plank, bird-dog) and the "Big 3". Biomechanical foundation for FORMA's PlankDetector and SidePlankDetector.

### schoenfeld2014plank
**Harvard.** Schoenfeld, B.J. *et al.* (2014) 'An electromyographic comparison of a modified plank with long lever and posterior tilt versus the traditional plank', *Sports Biomech.*, 13(3), pp. 296-306.
**Annotation.** Quantifies rectus abdominis and external oblique EMG across plank variations. Cite for FORMA's PlankDetector posterior-pelvic-tilt rationale and hip-sag penalty.

### boren2011gluteal — Side-plank gluteal activation
**Harvard.** Boren, K. *et al.* (2011) 'Electromyographic analysis of gluteus medius and gluteus maximus during rehabilitation exercises', *Int. J. Sports Phys. Ther.*, 6(3), pp. 206-223.
**Annotation.** Side-plank-abduction achieved ~103% MVIC of gluteus medius — the highest of 18 exercises. Reinforces side-plank as a meaningful core exercise in FORMA's curriculum.

### ekstrom2007core
**Harvard.** Ekstrom, R.A., Donatelli, R.A. and Carp, K.C. (2007) 'Electromyographic analysis of core trunk, hip, and thigh muscles during 9 rehabilitation exercises', *J. Orthop. Sports Phys. Ther.*, 37(12), pp. 754-762.
**Annotation.** Side-bridge (side plank) activates gluteus medius and external oblique most effectively among 9 exercises. Supports FORMA's SidePlankDetector hip-lift and body-line rules.

---

## 2.7  Crunch / abdominal exercises

### juker1998abdominal
**Harvard.** Juker, D., McGill, S., Kropf, P. and Steffen, T. (1998) 'Quantitative intramuscular myoelectric activity of lumbar portions of psoas and the abdominal wall during a wide variety of tasks', *Med. Sci. Sports Exerc.*, 30(2), pp. 301-310.
**Annotation.** Intramuscular EMG during crunches. Cite as the biomechanical basis for FORMA's CrunchDetector trunk-flexion angle thresholds and hip-flexor-dominance flag.

### escamilla2006abdominal
**Harvard.** Escamilla, R.F. *et al.* (2006) 'Electromyographic analysis of traditional and nontraditional abdominal exercises: implications for rehabilitation and training', *Physical Therapy*, 86(5), pp. 656-671.
**Annotation.** Compares 13 abdominal exercises; informs which variations maximise rectus abdominis activation while minimising hip-flexor involvement.

---

## 2.8  Bicep curl

### oliveira2009bicepscurl
**Harvard.** Oliveira, L.F. *et al.* (2009) 'Effect of the shoulder position on the biceps brachii EMG in different dumbbell curls', *J. Sports Sci. Med.*, 8(1), pp. 24-29.
**Annotation.** Biceps EMG depends strongly on shoulder position due to length-tension. Cite as evidence for FORMA's BicepCurlDetector upper-arm-stability (anti-cheat-swing) rule.

---

## 2.9  Lateral raise + shoulder

### botton2013deltoid
**Harvard.** Botton, C.E. *et al.* (2013) 'Electromyographical analysis of the deltoid muscle between different strength training exercises', *Medicina Sportiva*, 17(2), pp. 67-71.
**Annotation.** Lateral raise elicited highest medial-deltoid EMG among tested exercises. Supports FORMA's LateralRaiseDetector shoulder-abduction-angle rule.

### coratella2020lateralraise
**Harvard.** Coratella, G. *et al.* (2020) 'An electromyographic analysis of lateral raise variations and frontal raise in competitive bodybuilders', *Int. J. Environ. Res. Public Health*, 17(17), p. 6015.
**Annotation.** Raises above shoulder height shift load to upper trapezius and increase subacromial impingement risk. Directly supports FORMA's LateralRaiseDetector "do not raise above shoulder line" rule.

---

## 2.10  Bench press (retained for literature background)

### elliott1989benchpress
**Harvard.** Elliott, B.C., Wilson, G.J. and Kerr, G.K. (1989) 'A biomechanical analysis of the sticking region in the bench press', *Med. Sci. Sports Exerc.*, 21(4), pp. 450-462.
**Annotation.** 3D cinematography + EMG of elite powerlifters; identifies the sticking region as mechanical-disadvantage-driven rather than activation-driven. Historical reference for FORMA's (now dropped) bench-press detector.

### green2007benchgrip
**Harvard.** Green, C.M. and Comfort, P. (2007) 'The affect of grip width on bench press performance and risk of injury', *Strength Cond. J.*, 29(5), pp. 10-14.
**Annotation.** Reviews shoulder-injury risk associated with wide-grip bench press (>1.5 biacromial width). Part of the rationale for why heavy pressing was excluded from FORMA's home-friendly 10.

---

## 2.11  Rep counting via peak detection

### morris2014recofit
**Harvard.** Morris, D., Saponas, T.S., Guillory, A. and Kelner, I. (2014) 'RecoFit: using a wearable sensor to find, recognize, and count repetitive exercises', in *Proceedings of ACM CHI*, pp. 3225-3234.
**DOI.** 10.1145/2556288.2557116.
**Annotation.** Arm-worn IMU achieving ±1 rep accuracy 93% of the time. Cite as prior art for rep counting; justifies FORMA's peak-detection approach on angle time-series rather than IMU.

### pernek2013rep
**Harvard.** Pernek, I., Hummel, K.A. and Kokol, P. (2013) 'Exercise repetition detection for resistance training based on smartphones', *Pers. Ubiquit. Comput.*, 17(4), pp. 771-782.
**Annotation.** DTW rep-detection on smartphone accelerometer reaches 99% accuracy on 8 exercises. Cite when comparing FORMA's angle-peak-detection rep counter to IMU+DTW alternatives.

---

## 2.12  Symmetry / bilateral asymmetry

### bishop2018asymmetry
**Harvard.** Bishop, C., Turner, A. and Read, P. (2018) 'Effects of inter-limb asymmetries on physical and sports performance: a systematic review', *J. Sports Sci.*, 36(10), pp. 1135-1144.
**Annotation.** Systematic review of 18 studies showing inter-limb asymmetries >10-15% are detrimental to jump, sprint, and change-of-direction performance. Justifies FORMA's left-right symmetry checks in PushUpDetector, LateralRaiseDetector and BicepCurlDetector.

---

## 2.13  Modern CV + biomechanics intersection

### chiquier2023muscles
**Harvard.** Chiquier, M. and Vondrick, C. (2023) 'Muscles in Action', in *IEEE/CVF ICCV*, pp. 22091-22101.
**Annotation.** Bridges pose and muscle activation via a 12.5-hour video+sEMG dataset across exercises. Modern support for the research direction FORMA operates in — inferring form quality from video-based kinematics.

---

**Gaps in this section** — Signorile 2002 studied lat pull-down not biceps curl (misattribution risk); Lehman 2005 tricep-dip-specific paper could not be verified. Khurana 2020 video rep counting replaced by Morris 2014 + Pernek 2013.
