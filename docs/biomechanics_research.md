# Biomechanical joint angles for AI-powered exercise form detection

**Every joint angle your MediaPipe system needs to detect correct form, flag errors, and prevent injuries across 10 exercises — drawn from peer-reviewed biomechanics research.** The data below synthesizes findings from Escamilla, Schoenfeld, Contreras, McGill, McKenzie, and others, mapped to MediaPipe's 33-landmark pose estimation system. Two critical conventions first: MediaPipe calculates **included angles** (180° = straight limb, decreasing with flexion), and all angles below use this convention unless stated otherwise. Individual anthropometry can shift "correct" angles by 10–20°, so the system should calibrate against each user's baseline.

---

## 1. Barbell back squat: depth, lean, and valgus thresholds

**Knee angle (HIP → KNEE → ANKLE):** At parallel depth (femur horizontal), expect an included angle of **70–90°**. Below-parallel/powerlifting depth reaches **55–70°**, and a full ATG squat drops to **40–55°**. Standing lockout should read **170–180°**. Escamilla (2001) defined parallel as ~100° of anatomical flexion. Caterisano et al. (2002) showed glute activation jumps from **16.9% MVIC** at partial depth to **35.5%** at full depth, making depth tracking functionally important.

**Hip angle (SHOULDER → HIP → KNEE):** Standing reads ~170–180°. At parallel, expect **70–90°**. Deep squats reach **50–65°**. Kotiuk et al. (2023, n=103) measured **128.21°** of hip flexion at parallel using anatomical convention. A barbell adds ~4.56° more flexion than bodyweight.

**Trunk angle (SHOULDER-HIP line vs. vertical):** High-bar squats produce **20–35°** forward lean at the bottom; low-bar shifts this to **30–45°**. Hales et al. (2009) measured trunk angles of ~60° from horizontal for high-bar and ~55° for low-bar. The critical distinction is not the lean itself but *how* it's achieved — forward lean via hip flexion with a neutral spine is safe, while lean via **lumbar flexion** increases compressive and shear forces dramatically (Straub & Powers, 2024). Flag trunk angles exceeding **45° for high-bar** or **55° for low-bar** as potential errors.

**Knee valgus (ANKLE → KNEE → HIP, frontal view):** A straight 180° line is ideal. Herrington et al. (2014) found healthy controls average **8.4° ± 5.1°** of frontal plane projection angle (FPPA). Patellofemoral pain patients averaged **16.8°**. Hewett et al. (2005) showed ACL-injured athletes exhibited **>8° more valgus** than uninjured athletes during landing, predicting injury with 73% sensitivity. For your system: **0–8° = acceptable**, **8–15° = warning**, **>15° = error flag**. Context matters — valgus at deep flexion under controlled load is far less dangerous than valgus at 0–30° flexion during dynamic movements.

**Ankle dorsiflexion (KNEE → ANKLE → FOOT_INDEX):** Standing neutral is ~90°. Parallel depth requires **25–35°** of dorsiflexion (ankle angle dropping to ~55–65°). Hemmerich et al. (2006) measured **38.5° ± 5.9°** for deep squats. Limited dorsiflexion is the strongest predictor of squat depth (Kim et al., 2015) and correlates with increased knee valgus (Macrum et al., 2012).

**Body proportion effects:** A long-femur/short-torso individual may need **50–60°** forward lean (vs. 15–25° for the reverse) to keep the bar over midfoot. Wider stances compensate for long femurs by reducing both forward lean and dorsiflexion demands. The system should use a calibration frame to establish each user's baseline rather than applying universal thresholds.

**Key form errors and thresholds:**

| Error | Detection method | Threshold |
|---|---|---|
| Insufficient depth | HIP-KNEE-ANKLE angle | >100° at bottom |
| Excessive forward lean | SHOULDER-HIP vs. vertical | >45° high-bar, >55° low-bar |
| Knee valgus | ANKLE-KNEE-HIP frontal plane | >10° deviation |
| "Good morning" squat | Trunk angle increases while hips extend | >10° trunk angle increase from bottom during ascent |
| Butt wink | Sudden hip angle change near bottom | Rapid angle acceleration at end-range |
| Heel rise | HEEL y vs. TOE y | Heel rises above toe plane |

---

## 2. Forward lunge: knee tracking, step length, and pelvic control

**Front knee angle (HIP → KNEE → ANKLE):** At the bottom position, the included angle should be **80–100°**, with the thigh approximately parallel to the ground. Kotiuk et al. (2023) measured **98.04°** of knee flexion (bodyweight), decreasing to ~93.89° with a barbell.

**Knee-over-toe position:** The long-held prohibition against the knee passing the toes has been effectively debunked. Fry, Smith & Schilling (2003) demonstrated that restricting forward knee travel **decreased knee torque by 22% but increased hip torque by nearly 1000%**, shifting dangerous stress to the lumbar spine. Modern consensus holds that knee travel past the toes is biomechanically necessary for proper mechanics. For your system, monitor heel contact instead — if the heel rises, that indicates problematic forward shift.

**Rear knee angle (HIP → KNEE → ANKLE, trailing leg):** Should reach approximately **90–110°** at the bottom, with the knee hovering just above the ground. The rear hip should be in slight extension.

**Torso angle (SHOULDER-HIP vs. vertical):** Standard lunge targets **0–20°** forward lean. Hip-focused variations allow **20–40°**. Flag **>45°** as excessive unless intentional. Farrokhi et al. (2008) confirmed that forward trunk lean shifts demand to hip extensors while upright positioning loads the knee extensors.

**Lateral pelvic tilt (LEFT_HIP vs. RIGHT_HIP y-coordinates):** Acceptable range is **0–5°**. Flag **>8°** as indicative of gluteus medius weakness or poor hip stability. Greater than **10°** represents a Trendelenburg-like pattern requiring correction.

**Step length effects:** Song et al. (2023) tested four step lengths and found that longer steps increase hip flexion, decrease knee flexion, decrease ankle dorsiflexion, and increase gluteus maximus activation. Optimal step length equals approximately **100% of individual lower extremity length** (hip-to-ankle distance). The system can estimate this by measuring the ankle-to-ankle distance normalized against hip-to-ankle length from a standing calibration frame.

---

## 3. Conventional deadlift: spinal loading, lockout, and hip shoot detection

**Hip angle at liftoff (SHOULDER → HIP → KNEE):** Expect an included angle of **65–80°**, representing 100–115° of hip flexion. Escamilla et al. (2000) and Swinton et al. (2011) consistently measured ~100–110° of hip flexion at liftoff for conventional pulls. Sumo deadlifts show ~10° less flexion with a more upright trunk.

**Knee angle at liftoff (HIP → KNEE → ANKLE):** The included angle should be **110–135°** (~45–70° flexion). The majority of knee extension occurs early in the lift. Flag starting positions below **95°** (squatting the deadlift) or above **150°** (stiff-legged pattern). At lockout, both hips and knees should reach **170–180°**.

**Trunk angle (SHOULDER-HIP vs. vertical):** At liftoff, expect **40–65°** forward lean from vertical. Conventional deadlifts produce 5–9° more horizontal trunk angles than sumo (Escamilla et al., 2000). The trunk angle should become progressively more vertical throughout the lift — if it becomes *more* horizontal after liftoff, that indicates a hip shoot error.

**Spinal flexion thresholds:** This is where McGill's research provides critical context. There is **no single "magic number"** for dangerous spinal flexion — the injury mechanism is cumulative, driven by repeated flexion-extension cycles under load. Cholewicki & McGill (1992) found national-class powerlifters completed deadlifts with only **1.5–13°** less lumbar flexion than their terminal range. Howe & Lehman (2021) found that heavy squats and deadlifts typically involve **50–80%** of maximum lumbar flexion capacity. Approaching **>80%** of maximum ROM recruits passive spinal structures and compromises spine strength by **23–47%**. For the AI system, flag lumbar flexion that *changes* by **>20°** from the starting position during the lift — this dynamic flexion under load represents the highest injury risk.

**Lockout:** The IPF requires full hip and knee extension ("locked"). Flag any backward lean past vertical (**>5°**) — hyperextension at lockout compresses facet joints and provides no mechanical benefit. Proper lockout means an erect standing position with shoulders directly over hips.

**Hip shoot detection:** No published ratio exists in peer-reviewed literature, but the practical guideline is clear: chest and hips should rise at the same rate. Track the vertical displacement rate of hip landmarks vs. shoulder landmarks. **Flag if hip vertical velocity exceeds shoulder vertical velocity by >1.5×** during the liftoff-to-knee-passing phase. Also flag if the trunk angle becomes more horizontal after the bar breaks the floor.

**Shin angle (KNEE-ANKLE vs. vertical):** Rippetoe recommends approximately **7–8°** forward from vertical. Practical range is **5–15°**. Flag shins exceeding **20°** as excessive forward knee travel.

**Body proportion effects:** Longer femurs and shorter arms require more forward trunk inclination, increasing spinal extensor demands. Longer arms permit a more upright starting position. The system should use relative angle changes and hip-to-shoulder velocity ratios rather than fixed absolute thresholds. Swinton et al. (2011) found no significant difference in starting posture from 10–80% 1RM — self-organization changes emerge primarily at >90% loads (Gundersen et al., 2025).

---

## 4. Bench press: elbow flare zones and bar path tracking

**Elbow angle at bottom (SHOULDER → ELBOW → WRIST):** When the bar contacts the chest, expect **85–95°** of elbow flexion. Grip width modulates this — narrow grips produce more acute flexion, wide grips less.

**Elbow flare (shoulder abduction, frontal view):** This is the most injury-relevant angle in the bench press. Measure the angle between the HIP-SHOULDER vector and the SHOULDER-ELBOW vector in the frontal plane. The research-backed zones are:

- **45–75°: Safe zone** — consensus from NSCA, Rippetoe (~70°), and Zatsiorsky & Kraemer (65–90°)
- **75–85°: Caution zone** — increasing shoulder stress
- **≥85°: Danger zone** — shoulder impingement risk; humeral head compresses rotator cuff against the acromion (Green & Comfort, 2007; Fees et al., 1998)
- **<30°: Inefficiency zone** — excessive tuck shifts load to triceps and front delts, reducing chest activation

Noteboom et al. (2024) provided the first musculoskeletal modeling evidence across 21 technique variations, confirming that **abduction angle has the largest effect on glenohumeral and AC joint forces**, with 90° producing the highest loads. Scavenius & Iversen (1992) found **27% prevalence** of distal clavicle osteolysis in competitive bench pressers.

**Forearm verticality (ELBOW-WRIST line vs. vertical):** The forearm should be perpendicular to the floor in both frontal and sagittal views. Flag any deviation **>15°** — this indicates grip width mismatch. Measure using the angle of the elbow-to-wrist vector relative to the gravity line.

**Bar path (wrist trajectory across frames):** Elite lifters press in a characteristic **J-curve** pattern: the bar descends roughly linearly from over the shoulders to the lower sternum, then presses up *and back* toward the shoulders during the concentric phase, with **5–8 cm** of horizontal displacement (McLaughlin, 1984). Novice lifters tend to press straight up. Track wrist position frame-by-frame in the sagittal plane.

**Shoulder/scapular position:** NSCA guidelines (Graham, 2003) recommend maintaining continuous scapular retraction throughout the lift. Flag visible shoulder protraction (shoulder landmark rolling anterior) during the press. Sharma & Singh Vij (2005) found that **76% of bench pressers with shoulder pain** had tendinitis, with 56% involving the rotator cuff.

---

## 5. Overhead press: torso lean limits and the impingement transit zone

**Starting elbow angle (SHOULDER → ELBOW → WRIST):** Approximately **85–95°** of flexion with the bar at clavicle height. Forearms should be vertical, stacked directly under the bar.

**Bar path:** Should travel in a nearly straight vertical line from the front deltoids to directly overhead. The lifter creates clearance by pushing the hips slightly forward (not by pressing the bar around the head). After the bar passes the forehead, the head moves forward between the arms ("push head through"). At lockout, the bar should sit directly over **mid-foot, shoulders, hips, and ankles** — forming a plumb line over the C2 vertebra.

**Torso lean (SHOULDER-HIP line vs. vertical):** A slight **5–10°** backward lean is necessary and acceptable to clear the head for a vertical bar path. Flag **10–15°** as a caution zone. **>15°** transforms the movement into a standing incline press with significant lumbar hyperextension risk. Historical context: excessive backward lean in competitive overhead pressing caused so many spinal injuries that the lift was removed from Olympic weightlifting competition.

**Shoulder impingement considerations:** The **60–120°** shoulder elevation range constitutes the "painful arc" for subacromial impingement (Neer, 1972). However, this is a *transit zone*, not a position to avoid entirely. Pressing in the **scapular plane** (elbows 30–45° anterior to the frontal plane, at the "10 and 2 o'clock" positions) dramatically reduces impingement risk compared to pressing in the pure frontal plane (elbows at 90° abduction). ACE Fitness (2025) emphasizes the 2:1 scapulohumeral rhythm — for every 2° of humeral elevation, 1° of scapular upward rotation should occur. Active shoulder shrug at lockout pulls the acromion away from the humeral head, making impingement "anatomically impossible" at correct lockout (Rippetoe).

**Lockout:** Full elbow extension (~**175–180°**) with shoulders actively shrugged upward. Flag incomplete lockout (<165° elbow) or bar finishing anterior to the shoulder joints.

---

## 6. Pull-up: ROM standards and swing thresholds

**Elbow at bottom (SHOULDER → ELBOW → WRIST):** Full dead hang produces **170–180°**. A slight bend (**170–175°**) maintains scapular engagement and protects the joint capsule. Flag **<160°** as incomplete extension between reps.

**Elbow at top:** When the chin clears the bar, elbow angle reaches approximately **70–90°** (Youdas et al., 2010, using Vicon motion capture). Seedman recommends **~90°** as optimal for maximizing lat activation. For rep counting, set the minimum threshold at **≤100°** to validate a complete rep; flag **>110°** as an incomplete repetition.

**Shoulder angle (HIP → SHOULDER → ELBOW):** At the bottom, the shoulder is at ~170–180° elevation (arms overhead). At the top, it adducts/extends to approximately **10–30°** from the torso. Prinold & Bull (2016) provided the most detailed scapular kinematics: **wide-grip pull-ups** at ~90° abduction with ~45° external rotation showed increased subacromial impingement risk. Standard-grip pull-ups demonstrated the safest scapular kinematics.

**Body swing (SHOULDER-HIP line vs. vertical):** Strict pull-ups should maintain the body within **0–10°** of vertical. Flag **15–25°** as excessive swing. Beyond **25°** indicates kipping, which Snarr & Esco (2013) showed produces significantly lower upper-body EMG activation due to momentum assistance.

**Scapular position:** Cannot be directly measured by MediaPipe, but use shoulder-to-ear distance as a proxy. Decreasing distance indicates shoulder shrugging (loss of scapular depression). At the bottom of each rep, scapulae should be depressed and slightly retracted — not passively hanging.

---

## 7. Push-up: the 45° flare myth and body line detection

**Elbow at bottom (SHOULDER → ELBOW → WRIST):** The FITNESSGRAM standard uses **90°** as the reliable, standardized criterion. Full-ROM push-ups reach **60–70°** (chest near floor). San Juan et al. (2015) found vGRF peaks at 90° elbow flexion at **76% bodyweight**. For rep validation: flag elbow angles **>110°** as partial reps.

**Elbow flare (SHOULDER-ELBOW vs. SHOULDER-HIP vector):** Research supports **20–45°** as optimal — not the commonly cited 45° as a single target. Chou et al. (2010) found optimal leverage at 20–40°. Built With Science, citing multiple studies, identifies **>60°** as the threshold where shoulder impingement risk significantly increases. The 90° "T-position" dramatically increases anterior and superior glenohumeral shear forces. For the system: **20–45° = optimal**, **45–60° = acceptable**, **>60° = warning**, **>75° = error**.

**Hip alignment (SHOULDER → HIP → ANKLE):** The gold standard is a straight line reading **175–185°**. Contreras et al. (2012) noted this requires simultaneous stiffening of knee, hip, pelvis, and spine joints. L4/L5 compressive loading during standard push-ups reaches **~1,838 N** and increases with sagging. Thresholds: **170–190° = acceptable**, **<165° = hip sag warning**, **<160° = error**, **>195° = pike warning**.

**Head/neck position:** The nose landmark should align with the SHOULDER-HIP spine projection. Flag if the nose drops significantly below this line (cervical flexion) or rises more than **15°** above it (cervical extension). Ideal gaze is at the floor ~6 inches ahead of the fingertips.

**Tempo:** A push-up tempo study (24 male athletes) found that **2:0:2** (2-second eccentric, 2-second concentric at 30 bpm) produced the most effective ROM usage with the most stable joint kinematics. Faster tempos increased vGRF and joint instability. Flag total rep times **<1.5 seconds** as excessively fast.

---

## 8. Plank: sag detection, pike thresholds, and fatigue monitoring

**Hip angle (SHOULDER → HIP → ANKLE):** Target **175–185°** for a straight body line. Testing protocols from Strand et al. (2014, n=102 collegiate athletes) and the ACFT require "a straight line between shoulders, hips, knees, and ankles."

**Sag vs. pike thresholds:** These are the most important plank detection parameters:

| Zone | SHOULDER-HIP-ANKLE angle | Status |
|---|---|---|
| Perfect | 175–185° | Correct form |
| Acceptable | 170–190° | Minor deviation |
| Mild sag / mild pike | 165–170° / 190–195° | Warning — verbal cue |
| Excessive sag / excessive pike | <160° / >200° | Stop exercise |

**Shoulder alignment:** In a forearm plank, elbows should be at **90°** and positioned **directly below shoulders**. Verify that SHOULDER landmarks (11/12) are roughly vertically above ELBOW landmarks (13/14). Flag horizontal displacement exceeding **~15% of upper arm length**.

**Fatigue-driven form degradation:** Strand et al. (2014) found mean hold times of **124.5 ± 59.1 seconds** (males) and **83.3 ± 63.0 seconds** (females) before form collapse. The system should track **angle drift from the initial position**: warn at **>5° change** from starting angle, recommend stopping at **>10° drift**. Rate of change matters too — sudden drops indicate imminent failure. Abdominal fatigue is the most common termination reason and produces the shortest hold times, manifesting as progressive hip sag. McGill recommends **10-second holds for 3–4 sets** rather than extended holds to maintain form quality.

**Knee angle (HIP → KNEE → ANKLE):** Should be ~180° (straight legs). Flag **<170°** as knee bend.

---

## 9. Bicep curl: cheating detection through trunk and shoulder monitoring

**Elbow at top (SHOULDER → ELBOW → WRIST):** Full contraction produces an included angle of approximately **30–50°** (Oliveira et al., 2009). Maximum biceps torque occurs at **90°** (forearm parallel to floor), but full ROM should continue past this point. Flag **>60°** at the top as incomplete contraction; **>70°** as very incomplete.

**Elbow at bottom:** Target **165–175°** (slight bend maintained). Full lockout to 180° is acceptable but creates hyperextension stress on the biceps tendon at the radial tuberosity under heavy eccentric load. Flag **<150°** as excessively partial ROM.

**Shoulder movement (upper arm vs. torso angle):** This is the primary "cheating" indicator. Measure the angle of the SHOULDER-ELBOW vector relative to the SHOULDER-HIP vector in the sagittal plane. **0–10° forward drift = strict form**, **10–20° = mild cheating**, **>20° = excessive momentum**, **>30° = transitions into a front raise**. At >30°, anterior deltoid involvement replaces biceps as the prime mover.

**Trunk swing (HIP-SHOULDER line vs. vertical):** Track frame-to-frame changes in the torso angle. **<5° deviation = strict**, **5–10° = mild swing/warning**, **>10–15° = excessive momentum/error**. Rapid trunk angular velocity (>5°/frame at 30 fps) indicates momentum use regardless of absolute angle. Low back strain risk increases significantly beyond 15° of trunk extension under load.

**Wrist angle:** MediaPipe cannot directly measure forearm supination, but wrist flexion/extension can be estimated from the ELBOW → WRIST → INDEX_FINGER angle. Neutral alignment reads **170–180°**. Flag **<160°** as excessive wrist curl. Coratella et al. (2023) showed supinated grip produces **+19%** more biceps brachii activation versus pronated grip (effect size 2.60).

---

## 10. Tricep dip: the bench dip problem and impingement zones

**Elbow at bottom (SHOULDER → ELBOW → WRIST):** The safe target is **85–100°** (upper arms approximately parallel to the ground). McKenzie et al. (2022), the only peer-reviewed study using 3D motion capture across dip variations, found bar dips produce significantly greater elbow flexion than bench dips. Flag **<80°** as excessively deep — beyond this point, forced internal rotation at the shoulder increases impingement risk.

**Shoulder extension at bottom (ELBOW-SHOULDER vs. SHOULDER-HIP angle, sagittal plane):** This is the most critical safety metric for dips. McKenzie et al. (2022) produced a landmark finding: **bench dips force the shoulder to 101.35% of maximum passive extension ROM** (~89–90°), meaning the exercise exceeds the shoulder's structural limits. Bar dips reach **88.03%** (~78°), and ring dips reach **68.88%** (~62°). The Brookbush Institute identifies **60° of total shoulder complex extension** as the safe limit, beyond which anterior capsule strain and subacromial compression increase dramatically. For the system: **<50° = safe**, **50–65° = moderate risk**, **>65–70° = high impingement risk**.

**Torso lean (SHOULDER-HIP vs. vertical):** This determines whether the dip targets triceps or chest:

- **0–15°** forward lean = triceps emphasis
- **15–30°** = balanced loading
- **30–45°** = chest emphasis (as confirmed by Barnett et al., JSCR)
- **>45°** = excessive lean with increased spinal loading

McKenzie et al. (2022) found that anterior thoracic lean increases significantly from bench dip (most upright) to bar dip to ring dip (most forward), with a very large effect size (d=3.46).

**Elbow flare (frontal view, SHOULDER-ELBOW vs. SHOULDER-HIP):** For triceps-focused dips, keep elbows tucked at **0–15°**. Chest dips allow natural flare of **20–45°**. Flag **>50°** as excessive. McKenzie et al. found no significant difference in shoulder abduction across dip types (p=0.516).

**Impingement onset:** Flag shoulder extension exceeding **60°** (where anterior scapular tipping combines with humeral anterior/superior glide). Also monitor when the shoulder landmark drops significantly below elbow level — this is a reliable visual proxy for excessive shoulder extension. Cook & Purdam (2011) identified compressive biceps tendon loading that increases with shoulder extension as the tendon wraps around the humeral head.

**Bench dip caution:** Given McKenzie et al.'s finding that bench dips exceed maximum shoulder ROM, the AI system should display a persistent warning for bench dips and recommend bar or ring dips as alternatives, particularly for users with any shoulder injury history. Bar dips produce **higher muscle activation** (~1.04 mV vs. 0.83 mV triceps EMG) with **less shoulder extension demand** — a strictly better risk-to-reward ratio.

---

## Conclusion: calibration principles for the detection system

The single most important design decision is **rejecting universal fixed thresholds** in favor of individual calibration. Anthropometric variation — particularly femur-to-torso ratio, arm length, and shoulder structure — can shift "correct" angles by 10–20° across users. The system should capture a baseline calibration frame for each user and flag deviations *relative to their established pattern* rather than absolute angles.

Three tiers of detection priority emerge from the research. **Tier 1 (injury-critical):** knee valgus >10° during squats/lunges, shoulder extension >60° during dips, lumbar flexion change >20° during deadlifts, and elbow flare >75° during pressing. **Tier 2 (form quality):** depth thresholds, trunk lean ranges, hip sag/pike in planks and push-ups, and rep completeness. **Tier 3 (performance optimization):** bar path tracking, tempo monitoring, and movement symmetry.

The MediaPipe limitation to be most aware of is the inability to measure spinal curvature segmentally — the system can detect gross trunk angle changes but cannot differentiate thoracic from lumbar flexion. For deadlifts and squats, tracking the *rate of change* in the SHOULDER-HIP angle serves as the most reliable proxy for spinal flexion events. Finally, Swinton et al. (2011) demonstrated that lifting posture remains remarkably stable from 10–80% 1RM, meaning form thresholds calibrated at moderate loads remain valid across most training intensities.