# Topic 10 — Ethics / SE Considerations

Sources for FORMA's ethics chapter: FDA/SaMD boundary, GDPR / ICO
biometrics, ML algorithmic bias (Gender Shades, pose fairness), model
cards, datasheets, IEEE / ACM ethics codes, ISO SE standards, and
on-device-vs-cloud privacy trade-offs.

---

## 10.1  Algorithmic bias + fairness in computer vision

### buolamwini2018gendershades
**Harvard.** Buolamwini, J. and Gebru, T. (2018) 'Gender Shades: Intersectional Accuracy Disparities in Commercial Gender Classification', in *Proceedings of FAT* 2018*, PMLR 81, pp. 77-91.
**Annotation.** Canonical algorithmic-bias audit showing face-classifiers performed far worse on darker-skinned women than lighter-skinned men. Cite to justify FORMA's honest limitation that MediaPipe BlazePose's form-coaching accuracy has not been independently audited across skin tones or body types.
**Suggested quote.** "Darker-skinned females are the most misclassified group (with error rates of up to 34.7%)."

### raji2019actionableauditing
**Harvard.** Raji, I.D. and Buolamwini, J. (2019) 'Actionable Auditing: Investigating the Impact of Publicly Naming Biased Performance Results of Commercial AI Products', in *Proceedings of AIES '19*, pp. 429-435.
**DOI.** 10.1145/3306618.3314244.
**Annotation.** Shows public disclosure can drive measurable accuracy improvements in commercial vision systems. Argues FORMA's honest-limitation stance is a recognised accountability mechanism.

### wilson2019predictiveinequity
**Harvard.** Wilson, B., Hoffman, J. and Morgenstern, J. (2019) 'Predictive Inequity in Object Detection', *arXiv preprint* arXiv:1902.11097.
**Annotation.** SOTA pedestrian detectors had lower accuracy on Fitzpatrick IV-VI skin tones. Direct precedent for suspecting MediaPipe BlazePose may exhibit similar skin-tone disparities.

### gustafson2023facet
**Harvard.** Gustafson, L. *et al.* (2023) 'FACET: Fairness in Computer Vision Evaluation Benchmark', in *IEEE/CVF ICCV 2023*, pp. 20370-20382.
**Annotation.** 32 k-image benchmark with perceived-skin-tone and hair-type annotations. The benchmark a future fairness audit of FORMA's pose stack could evaluate against.

---

## 10.2  ML accountability + documentation

### mitchell2019modelcards
**Harvard.** Mitchell, M. *et al.* (2019) 'Model Cards for Model Reporting', in *Proceedings of FAT* '19*, pp. 220-229.
**DOI.** 10.1145/3287560.3287596.
**Annotation.** Proposes short standardised documents accompanying ML models. Cite as the documentation standard FORMA should adopt for each detector.

### gebru2021datasheets
**Harvard.** Gebru, T. *et al.* (2021) 'Datasheets for Datasets', *Communications of the ACM*, 64(12), pp. 86-92.
**DOI.** 10.1145/3458723.
**Annotation.** Every ML dataset should ship with a structured datasheet (motivation, collection, composition, recommended uses, ethics). Cite as the standard FORMA's Kaggle/YouTube training set falls short of, and as a recommended future-work item.

---

## 10.3  Medical-device regulation (SaMD boundary)

### imdrf2013samdkeydefs
**Harvard.** International Medical Device Regulators Forum (2013) *Software as a Medical Device (SaMD): Key Definitions*. IMDRF/SaMD WG/N10FINAL:2013.
**Annotation.** Defines SaMD. Cite to establish where FORMA sits: a fitness app that coaches form is **general wellness, not SaMD**.
**Suggested quote.** "Software intended to be used for one or more medical purposes that perform these purposes without being part of a hardware medical device."

### fda2022mma
**Harvard.** U.S. Food and Drug Administration (2022) *Policy for Device Software Functions and Mobile Medical Applications: Guidance for Industry and FDA Staff*. Silver Spring, MD: FDA.
**Annotation.** FDA's risk-based enforcement policy. Justifies FORMA being outside FDA enforcement priority while still shipping a medical disclaimer.

### fda2019precert
**Harvard.** U.S. Food and Drug Administration (2019) *Developing a Software Precertification Program: A Working Model, v1.0*. Silver Spring, MD: FDA.
**Annotation.** FDA's attempt to regulate digital health by certifying the developer. Context for why hobby/student SaMD projects cannot yet benefit from streamlined regulatory pathways.

---

## 10.4  GDPR, ICO biometrics, privacy

### eu2016gdpr
**Harvard.** European Parliament and Council of the European Union (2016) *Regulation (EU) 2016/679 (General Data Protection Regulation)*. *Official Journal of the European Union*, L 119, pp. 1-88.
**Annotation.** Primary legal instrument. Cite Article 5 (data-minimisation) and Article 9 (biometric data as special category) to justify FORMA's design: skeleton landmarks streamed in-memory, not raw video stored server-side.
**Suggested quote.** "personal data shall be ... adequate, relevant and limited to what is necessary in relation to the purposes for which they are processed ('data minimisation')." — Art. 5(1)(c).

### voigt2017gdprpracticalguide
**Harvard.** Voigt, P. and von dem Bussche, A. (2017) *The EU General Data Protection Regulation (GDPR): A Practical Guide*. Cham: Springer.
**DOI.** 10.1007/978-3-319-57959-7.
**Annotation.** Practitioner commentary on GDPR implementation. Interpretive authority on data-minimisation and DPIA requirements affecting FORMA's webcam pipeline.

### ico2024biometric
**Harvard.** Information Commissioner's Office (2024) *Biometric Data Guidance: Biometric Recognition*. Wilmslow: ICO, 5 March 2024.
**Annotation.** UK regulator's guidance on processing biometric data under UK GDPR. UK-jurisdiction authority (relevant for Greenwich FYP) that informs whether MediaPipe landmarks count as biometric data when used for fitness vs recognition.

### gates2011biometricfuture
**Harvard.** Gates, K.A. (2011) *Our Biometric Future: Facial Recognition Technology and the Culture of Surveillance*. New York: NYU Press.
**Annotation.** Critical-studies account of how biometric surveillance becomes normalised through "smart" consumer tech. Frames why FORMA should refuse to store raw webcam frames even when technically easy.

---

## 10.5  Professional ethics + SE standards

### acm2018ethics
**Harvard.** Association for Computing Machinery (2018) *ACM Code of Ethics and Professional Conduct*. ACM. Available at: https://www.acm.org/code-of-ethics.
**Annotation.** Professional-ethics baseline. Cite principles 1.2 (avoid harm), 1.6 (respect privacy), 2.5 (give comprehensive evaluations) for FORMA's design choices.

### ieee2019ead
**Harvard.** IEEE Global Initiative on Ethics of Autonomous and Intelligent Systems (2019) *Ethically Aligned Design: A Vision for Prioritizing Human Well-being with Autonomous and Intelligent Systems*. 1st edn. Piscataway, NJ: IEEE.
**Annotation.** IEEE framework for A/IS ethics. Principles of human well-being, transparency, accountability — FORMA's choice to disclose model F1 scores rather than hide them.

### isoiec25010_2011
**Harvard.** International Organization for Standardization (2011) *ISO/IEC 25010:2011 Systems and Software Engineering — SQuaRE — System and Software Quality Models*. Geneva: ISO.
**Annotation.** Quality model (Functional Suitability, Performance Efficiency, Usability, Reliability, Security, Maintainability, Portability). Structures FORMA's SE chapter — which characteristics were tested, which deferred.

### isoiecieee12207_2017
**Harvard.** ISO/IEC/IEEE (2017) *ISO/IEC/IEEE 12207:2017 Systems and Software Engineering — Software Life Cycle Processes*. Geneva: ISO.
**Annotation.** Lifecycle process standard. Maps FORMA's agile-ish milestone structure (data → model → UI → user test) to named processes (implementation, verification, validation).

---

## 10.6  Bioethics + consent

### beauchamp2019biomedicalethics
**Harvard.** Beauchamp, T.L. and Childress, J.F. (2019) *Principles of Biomedical Ethics*. 8th edn. New York: Oxford University Press.
**Annotation.** Foundational four-principles framework (autonomy, non-maleficence, beneficence, justice). For articulating informed-consent norms and the bounded-beneficence claim (FORMA cannot replace a physiotherapist).

### nuffield2015biologicaldata
**Harvard.** Nuffield Council on Bioethics (2015) *The collection, linking and use of data in biomedical research and health care: ethical issues*. London: Nuffield Council on Bioethics.
**Annotation.** UK-authoritative bioethics report. "Reasonable expectations" framework around consent and governance of biological/health data.

---

## 10.7  Health-app context + critical commentary

### torous2018mentalhealthapps
**Harvard.** Torous, J., Nicholas, J., Larsen, M.E. and Firth, J. (2018) 'Clinical review of user engagement with mental health smartphone apps: evidence, theory and improvements', *Evidence-Based Mental Health*, 21(3), pp. 116-119.
**Annotation.** Documents that consumer-health apps often do not "respect privacy, are not seen as trustworthy and are unhelpful in emergencies." Motivates FORMA's medical-disclaimer and trust-building design.

### lupton2014digitalhealth
**Harvard.** Lupton, D. (2014) 'Health promotion in the digital era: a critical commentary', *Health Promotion International*, 30(1), pp. 174-183.
**DOI.** 10.1093/heapro/dau091.
**Annotation.** Critical commentary that digital-health tools over-emphasise individual responsibility. Counterweight when framing FORMA's contribution — algorithmic coaching is not a substitute for professional guidance or structural access.

### orben2019adolescentwellbeing
**Harvard.** Orben, A. and Przybylski, A.K. (2019) 'The association between adolescent well-being and digital technology use', *Nature Human Behaviour*, 3(2), pp. 173-182.
**DOI.** 10.1038/s41562-018-0506-1.
**Annotation.** Specification-curve analysis (n=355 358): digital-tech well-being effect is negative but tiny (<0.4% variance). Useful as a counter-hyping reference when discussing AI/wellbeing generally.

---

## 10.8  Edge computing + on-device vs cloud

### shi2016edgecomputing
**Harvard.** Shi, W. *et al.* (2016) 'Edge Computing: Vision and Challenges', *IEEE Internet of Things Journal*, 3(5), pp. 637-646.
**DOI.** 10.1109/JIOT.2016.2579198.
**Annotation.** Seminal edge-computing paper — processing near source for latency, bandwidth, AND privacy. Justifies FORMA's architecture: MediaPipe inference in-browser, only 33 landmarks traverse WebSocket.

### mach2017mecsurvey
**Harvard.** Mach, P. and Becvar, Z. (2017) 'Mobile Edge Computing: A Survey on Architecture and Computation Offloading', *IEEE Communications Surveys & Tutorials*, 19(3), pp. 1628-1656.
**DOI.** 10.1109/COMST.2017.2682318.
**Annotation.** Systematic survey of on-device vs edge vs cloud offloading decisions. Explains why FORMA runs pose estimation client-side despite OpenAI chat going over the network.

---

**Gaps in this section** — Abbas 2017 MEC survey and Froomkin & Arencibia 2020 Algorithmic Accountability Act could not be verified. Sarkar 2020 and Kyrollos 2021 pose-fairness papers not located; FACET 2023 (above) is the closest canonical pose-fairness benchmark.
