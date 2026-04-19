# Topic 1 — Pose Estimation

Sources for FORMA's choice of MediaPipe BlazePose, the wider pose-estimation
landscape (OpenPose, HRNet, AlphaPose), foundational architectures,
standard datasets, and evaluation metrics.

---

## 1.1  BlazePose / MediaPipe (FORMA's backend)

### bazarevsky2020blazepose
**Harvard.** Bazarevsky, V., Grishchenko, I., Raveendran, K., Zhu, T., Zhang, F. and Grundmann, M. (2020) 'BlazePose: On-device Real-time Body Pose Tracking', *arXiv preprint* arXiv:2006.10204 (CVPR Workshop on Computer Vision for Augmented and Virtual Reality).
**Annotation.** The canonical citation for FORMA's pose backend. Two-stage detector-tracker pipeline producing 33 body keypoints (COCO-17 extended with hand/foot/face points), running over 30 FPS on a Pixel 2.
**Suggested quote.** "produces 33 body keypoints for a single person and runs at over 30 frames per second on a Pixel 2 phone."

### bazarevsky2019blazeface
**Harvard.** Bazarevsky, V., Kartynnik, Y., Vakunov, A., Raveendran, K. and Grundmann, M. (2019) 'BlazeFace: Sub-millisecond Neural Face Detection on Mobile GPUs', *arXiv preprint* arXiv:1907.05047.
**Annotation.** BlazeFace is the detector stage reused by BlazePose's two-stage pipeline (detector finds ROI → tracker produces landmarks). Cite when explaining MediaPipe Pose's first stage.

### sandler2018mobilenetv2 — Mobile CNN backbone
**Harvard.** Sandler, M., Howard, A., Zhu, M., Zhmoginov, A. and Chen, L.-C. (2018) 'MobileNetV2: Inverted Residuals and Linear Bottlenecks', in *IEEE CVPR*, pp. 4510-4520.
**arXiv.** 1801.04381.
**Annotation.** The mobile CNN backbone family (inverted residuals, depthwise separable convolutions) that enables BlazePose-Lite to run in real time on CPUs. Cite when explaining why MediaPipe Pose is light enough for consumer laptops.

---

## 1.2  Alternatives (justifying MediaPipe over OpenPose / HRNet / AlphaPose)

### cao2019openpose — OpenPose
**Harvard.** Cao, Z., Hidalgo, G., Simon, T., Wei, S.-E. and Sheikh, Y. (2021) 'OpenPose: Realtime Multi-Person 2D Pose Estimation Using Part Affinity Fields', *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 43(1), pp. 172-186.
**DOI.** 10.1109/TPAMI.2019.2929257. **arXiv.** 1812.08008.
**Annotation.** The primary alternative to MediaPipe. Bottom-up approach using Part Affinity Fields for multi-person pose. Cite in FORMA's "why MediaPipe over OpenPose" justification: OpenPose excels at multi-person but is heavier (GPU-bound), whereas FORMA needs single-user real-time on consumer hardware.
**Suggested quote.** "the first open-source realtime system for multi-person 2D pose detection, including body, foot, hand, and facial keypoints."

### sun2019hrnet — HRNet
**Harvard.** Sun, K., Xiao, B., Liu, D. and Wang, J. (2019) 'Deep High-Resolution Representation Learning for Human Pose Estimation', in *IEEE/CVF CVPR*, pp. 5693-5703.
**arXiv.** 1902.09212.
**Annotation.** HRNet maintains high-resolution representations throughout the network rather than recovering from low-res features. State-of-the-art accuracy on COCO/MPII in 2019. Cite as a high-accuracy alternative traded away for FORMA's real-time CPU constraint.

### fang2017rmpe — RMPE / AlphaPose
**Harvard.** Fang, H.-S., Xie, S., Tai, Y.-W. and Lu, C. (2017) 'RMPE: Regional Multi-person Pose Estimation', in *IEEE ICCV*, pp. 2334-2343.
**arXiv.** 1612.00137.
**Annotation.** Top-down framework, basis of the AlphaPose system. Handles inaccurate bounding boxes via a Symmetric Spatial Transformer Network. Cite in the related-work comparison of top-down (AlphaPose/HRNet) vs bottom-up (OpenPose) vs single-person-tracker (BlazePose) paradigms.

### papandreou2017accurate — Google's PoseNet/MoveNet lineage
**Harvard.** Papandreou, G., Zhu, T., Kanazawa, N., Toshev, A., Tompson, J., Bregler, C. and Murphy, K. (2017) 'Towards Accurate Multi-Person Pose Estimation in the Wild', in *IEEE CVPR*, pp. 4903-4911.
**Annotation.** Google's top-down multi-person pose paper; precursor to PoseNet / MoveNet. Cite when comparing Google's earlier PoseNet / MoveNet lineage with BlazePose.

---

## 1.3  Foundational pose architectures

### toshev2014deeppose
**Harvard.** Toshev, A. and Szegedy, C. (2014) 'DeepPose: Human Pose Estimation via Deep Neural Networks', in *IEEE CVPR*, pp. 1653-1660.
**arXiv.** 1312.4659.
**Annotation.** First work applying deep neural networks to human pose estimation, formulating it as a DNN-based regression problem with cascaded AlexNet regressors. Seminal citation to open FORMA's pose-estimation literature review.

### newell2016hourglass
**Harvard.** Newell, A., Yang, K. and Deng, J. (2016) 'Stacked Hourglass Networks for Human Pose Estimation', in *ECCV*, pp. 483-499.
**arXiv.** 1603.06937.
**Annotation.** Introduces the stacked hourglass architecture (symmetric pool-then-upsample blocks with intermediate supervision). Established heatmap regression as the dominant paradigm and directly influenced BlazePose's hybrid heatmap-plus-regression design.

### wei2016cpm — Convolutional Pose Machines
**Harvard.** Wei, S.-E., Ramakrishna, V., Kanade, T. and Sheikh, Y. (2016) 'Convolutional Pose Machines', in *IEEE CVPR*, pp. 4724-4732.
**arXiv.** 1602.00134.
**Annotation.** Sequential convolutional architecture that refines belief maps stage-by-stage, implicitly modelling long-range part dependencies. Basis for OpenPose.

---

## 1.4  Benchmarks + evaluation metrics (MPJPE, PCK, OKS)

### ionescu2014human36m — MPJPE benchmark
**Harvard.** Ionescu, C., Papava, D., Olaru, V. and Sminchisescu, C. (2014) 'Human3.6M: Large Scale Datasets and Predictive Methods for 3D Human Sensing in Natural Environments', *IEEE TPAMI*, 36(7), pp. 1325-1339.
**DOI.** 10.1109/TPAMI.2013.248.
**Annotation.** The standard benchmark for monocular 3D human pose: 3.6 M 3D poses, 11 subjects, 4 viewpoints. Cite when discussing MPJPE evaluation and when justifying FORMA's use of MediaPipe's 3D world-landmark (hip-centered) output.

### lin2014coco — COCO 17-keypoint topology + OKS
**Harvard.** Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P. and Zitnick, C.L. (2014) 'Microsoft COCO: Common Objects in Context', in *ECCV*, pp. 740-755.
**arXiv.** 1405.0312.
**Annotation.** COCO provides the 17-keypoint topology that BlazePose's 33-keypoint schema extends, and the OKS metric used to evaluate pose accuracy. Cite when describing keypoint conventions and discussing why BlazePose adds foot/hand points for fitness applications.

### andriluka2014mpii
**Harvard.** Andriluka, M., Pishchulin, L., Gehler, P. and Schiele, B. (2014) '2D Human Pose Estimation: New Benchmark and State of the Art Analysis', in *IEEE CVPR*, pp. 3686-3693.
**Annotation.** The MPII Human Pose benchmark: 25 K images, 40 K people, 410 human activities. Dominant 2D pose benchmark for PCKh evaluation. Relevant because it spans everyday physical activities including fitness.

### yang2013articulated — PCK metric
**Harvard.** Yang, Y. and Ramanan, D. (2013) 'Articulated Human Detection with Flexible Mixtures of Parts', *IEEE TPAMI*, 35(12), pp. 2878-2890.
**Annotation.** Introduced the Percentage of Correctly Localized Keypoints (PCK) evaluation metric that is now standard for 2D pose. Cite when defining PCK in FORMA's evaluation chapter.

### ronchi2017benchmarking
**Harvard.** Ronchi, M.R. and Perona, P. (2017) 'Benchmarking and Error Diagnosis in Multi-Instance Pose Estimation', in *IEEE ICCV*, pp. 369-378.
**arXiv.** 1707.05388.
**Annotation.** Defines and characterises localisation, scoring, and background errors in pose estimation, plus in-depth discussion of OKS. Cite in the evaluation-metrics subsection alongside MPJPE and PCK.
