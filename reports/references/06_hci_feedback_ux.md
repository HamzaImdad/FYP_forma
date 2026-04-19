# Topic 6 — HCI / Feedback UX

Sources for FORMA's "patient trainer" UX: one-error-at-a-time, words not
numbers, silence when correct, colour-coded skeleton, voice + visual
multimodal feedback, gamified motivation, and the motor-learning science
behind bandwidth / reduced-frequency feedback.

---

## 6.1  Motor learning: KR, KP, guidance hypothesis

### salmoni1984knowledge
**Harvard.** Salmoni, A.W., Schmidt, R.A. and Walter, C.B. (1984) 'Knowledge of Results and Motor Learning: A Review and Critical Reappraisal', *Psychological Bulletin*, 95(3), pp. 355-386.
**DOI.** 10.1037/0033-2909.95.3.355.
**Annotation.** Foundational review introducing the **guidance hypothesis** — over-frequent feedback makes learners dependent on it and degrades retention. Directly motivates FORMA's choice to go silent when movement is within tolerance.
**Suggested quote.** "KR also acts as guidance, enhancing performance when it is present but degrading learning if it is given too frequently."

### winstein1990reduced
**Harvard.** Winstein, C.J. and Schmidt, R.A. (1990) 'Reduced frequency of knowledge of results enhances motor skill learning', *J. Exp. Psychol.: LMC*, 16(4), pp. 677-691.
**DOI.** 10.1037/0278-7393.16.4.677.
**Annotation.** Three experiments showing fading KR frequency enhances long-term retention vs 100% KR. Central evidence that FORMA's "silence-when-correct + one-cue-at-a-time" strategy is pedagogically superior to continuous numeric scores.

### lee1990bandwidth
**Harvard.** Lee, T.D. and Carnahan, H. (1990) 'Bandwidth knowledge of results and motor learning: More than just a relative frequency effect', *QJEP: A*, 42(4), pp. 777-789.
**DOI.** 10.1080/14640749008401249.
**Annotation.** Bandwidth feedback (speak only when error exceeds a goal-centred tolerance) improves retention beyond the reduced-frequency effect. **Direct scientific grounding for FORMA's threshold-based colour-coded joint feedback** (green = silent, red = cue).

### schmidt1991frequent
**Harvard.** Schmidt, R.A. (1991) 'Frequent augmented feedback can degrade learning: Evidence and interpretations', in Requin, J. and Stelmach, G.E. (eds.) *Tutorials in Motor Neuroscience*. Dordrecht: Kluwer Academic, pp. 59-75.
**DOI.** 10.1007/978-94-011-3626-6_6.
**Annotation.** Schmidt's synthesis of the guidance hypothesis and the original formulation of bandwidth feedback. Canonical "silence when correct" citation.

### schmidt1997continuous
**Harvard.** Schmidt, R.A. and Wulf, G. (1997) 'Continuous concurrent feedback degrades skill learning: Implications for training and simulation', *Human Factors*, 39(4), pp. 509-525.
**DOI.** 10.1518/001872097778667979.
**Annotation.** Directly relevant: continuous on-screen feedback during practice interferes with motor-program formation on a no-feedback retention test. Key evidence against persistent numeric scores, favouring FORMA's "words-not-numbers, silence-when-correct" design.

### wulf2004understanding
**Harvard.** Wulf, G. and Shea, C.H. (2004) 'Understanding the role of augmented feedback: The good, the bad, and the ugly', in Williams, A.M. and Hodges, N.J. (eds.) *Skill Acquisition in Sport*. London: Routledge, pp. 121-144.
**Annotation.** Critical review arguing that for complex real-world skills some frequent feedback can help. Acknowledges limits of FORMA's silent-correct policy for absolute beginners.

### schmidt2019motor — Textbook
**Harvard.** Schmidt, R.A., Lee, T.D., Winstein, C.J., Wulf, G. and Zelaznik, H.N. (2019) *Motor Control and Learning: A Behavioral Emphasis*. 6th edn. Champaign, IL: Human Kinetics.
**Annotation.** The canonical motor-learning textbook. Used for definitions of KR vs KP, schedule effects, and the OPTIMAL theory that explains why brief, specific cues work better than a running score.

### wolpert2000computational
**Harvard.** Wolpert, D.M. and Ghahramani, Z. (2000) 'Computational principles of movement neuroscience', *Nature Neuroscience*, 3(S11), pp. 1212-1217.
**DOI.** 10.1038/81497.
**Annotation.** Unifies motor-learning into Bayesian estimation, forward models, internal simulation. Supports FORMA's claim that a visible skeleton overlay augments the learner's forward-model by externalising state estimate.

---

## 6.2  Multimodal feedback (voice + visual)

### sigrist2013augmented
**Harvard.** Sigrist, R., Rauter, G., Riener, R. and Wolf, P. (2013) 'Augmented visual, auditory, haptic, and multimodal feedback in motor learning: A review', *Psychonomic Bulletin & Review*, 20(1), pp. 21-53.
**DOI.** 10.3758/s13423-012-0333-8.
**Annotation.** Most comprehensive review of multimodal feedback in motor learning. Supports FORMA's use of synchronised voice coaching alongside the colour-coded skeleton, with caveats on cognitive-load limits.

---

## 6.3  Cognitive load

### sweller1988cognitive
**Harvard.** Sweller, J. (1988) 'Cognitive load during problem solving: Effects on learning', *Cognitive Science*, 12(2), pp. 257-285.
**DOI.** 10.1207/s15516709cog1202_4.
**Annotation.** Originating paper of Cognitive Load Theory. FORMA's "one error at a time" rule is operationalised from the intrinsic/extraneous load distinction — stacking multiple joint cues overflows working memory during a physically demanding squat.

### vanmerrienboer2005cognitive
**Harvard.** van Merriënboer, J.J.G. and Sweller, J. (2005) 'Cognitive load theory and complex learning: Recent developments and future directions', *Educational Psychology Review*, 17(2), pp. 147-177.
**DOI.** 10.1007/s10648-005-3951-0.
**Annotation.** Extends CLT to complex real-world tasks. Justifies FORMA's progressive disclosure — hide per-joint scores until requested, show only the most-violated cue.

### mayer2003nine
**Harvard.** Mayer, R.E. and Moreno, R. (2003) 'Nine ways to reduce cognitive load in multimedia learning', *Educational Psychologist*, 38(1), pp. 43-52.
**DOI.** 10.1207/S15326985EP3801_6.
**Annotation.** Nine concrete techniques (segmenting, signaling, weeding, modality) for managing cognitive load. Design-pattern source for FORMA's colour-coded overlays + brief voice cue.

---

## 6.4  Exertion interfaces + prior CV-coach systems

### velloso2013qualitative
**Harvard.** Velloso, E., Bulling, A., Gellersen, H., Ugulino, W. and Fuks, H. (2013) 'Qualitative activity recognition of weight lifting exercises', in *Proceedings of Augmented Human 2013*. New York: ACM, pp. 116-123.
**DOI.** 10.1145/2459236.2459256.
**Annotation.** The foundational CS paper on classifying not "what" but "how well" — coining **qualitative activity recognition (QAR)**. Closest direct antecedent of FORMA's form-scoring goal.

### anderson2013youmove
**Harvard.** Anderson, F., Grossman, T., Matejka, J. and Fitzmaurice, G. (2013) 'YouMove: Enhancing movement training with an augmented reality mirror', in *Proceedings of ACM UIST 2013*, pp. 311-320.
**DOI.** 10.1145/2501988.2502045.
**Annotation.** AR mirror teaching ballet/martial-arts postures via fading skeletal overlays — 2× short-term retention vs video. Primary prior-work reference for FORMA's mirrored-skeleton overlay and staged feedback.

### mueller2003exertion
**Harvard.** Mueller, F., Agamanolis, S. and Picard, R. (2003) 'Exertion interfaces: Sports over a distance for social bonding and fun', in *Proceedings of ACM CHI 2003*, pp. 561-568.
**DOI.** 10.1145/642611.642709.
**Annotation.** Paper that named "exertion interfaces" — UIs deliberately requiring physical effort. Sets the HCI context for FORMA as an exertion system.

### mueller2011designing
**Harvard.** Mueller, F. *et al.* (2011) 'Designing Sports: A Framework for Exertion Games', in *Proceedings of ACM CHI 2011*, pp. 2651-2660.
**DOI.** 10.1145/1978942.1979330.
**Annotation.** Four-lens framework (responding / moving / sensing / relating body) for designing exertion experiences. Gives FORMA a structured vocabulary for the embodied experience beyond form-correctness scores.

---

## 6.5  Colour-coded visual feedback + pre-attentive perception

### healey2012attention
**Harvard.** Healey, C.G. and Enns, J.T. (2012) 'Attention and visual memory in visualization and computer graphics', *IEEE TVCG*, 18(7), pp. 1170-1188.
**DOI.** 10.1109/TVCG.2011.127.
**Annotation.** Survey of pre-attentive visual features (colour, motion, orientation). Directly backs FORMA's use of saturated red vs green on the skeleton as a pre-attentive signal — the user sees the bad joint before parsing the overlay.
**Suggested quote.** "Certain visual features ... can be perceived within 200 ms, without the need for focused attention."

### ware2020information
**Harvard.** Ware, C. (2020) *Information Visualization: Perception for Design*. 4th edn. Cambridge, MA: Morgan Kaufmann.
**Annotation.** The standard reference on perception-driven visualisation. Justifies FORMA's colour scheme (lime-green default, red violation, saturated hues only where they matter).

---

## 6.6  Motivation + engagement

### wouters2013metaanalysis
**Harvard.** Wouters, P., van Nimwegen, C., van Oostendorp, H. and van der Spek, E.D. (2013) 'A meta-analysis of the cognitive and motivational effects of serious games', *J. Educational Psychology*, 105(2), pp. 249-265.
**DOI.** 10.1037/a0031311.
**Annotation.** Meta-analysis of 77 studies (N=5 547): serious games produce moderate learning (d=0.29) and retention (d=0.36) gains. Evidence for FORMA's gamified form-coaching as an empirically supported motivational layer.

### miller2012motivational
**Harvard.** Miller, W.R. and Rollnick, S. (2012) *Motivational Interviewing: Helping People Change*. 3rd edn. New York: Guilford Press.
**Annotation.** Foundational text of motivational interviewing. Frames FORMA's coach-voice tone as non-judgemental, autonomy-supporting and question-driven rather than directive — the "patient trainer" UX vision.

---

**Gaps in this section** — No peer-reviewed source specifically on "patient trainer" feedback framing as a named concept; Miller & Rollnick 2012 + Sigrist 2013 together substitute for it. No single canonical HCI paper on "traffic-light semantics"; Healey & Enns 2012 + Ware 2020 cover the perception side.
