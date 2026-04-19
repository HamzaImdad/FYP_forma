# Topic 3 — Classifier Architectures

Sources for the ML architectures used and compared in FORMA: Random Forest /
SVM / Logistic Regression baselines, LSTM / BiLSTM, CNN-LSTM, attention
mechanisms, focal loss, SWA, mixup, and skeleton-based action recognition.

---

## 3.1  Classical baselines

### breiman2001random — Random Forests
**Harvard.** Breiman, L. (2001) 'Random Forests', *Machine Learning*, 45(1), pp. 5-32.
**Venue.** *Machine Learning* (Springer). **DOI.** 10.1023/A:1010933404324.
**Annotation.** Seminal Random Forest paper. FORMA uses RF as one of three classical baselines (alongside SVM and LR) against which the CNN-BiLSTM is compared per-exercise. Bagging with random feature subsets gives a robust non-linear baseline for pose-feature classification.
**Suggested quote.** "Random forests are a combination of tree predictors such that each tree depends on the values of a random vector sampled independently and with the same distribution for all trees in the forest."

### cortes1995support — Support-Vector Networks
**Harvard.** Cortes, C. and Vapnik, V. (1995) 'Support-Vector Networks', *Machine Learning*, 20(3), pp. 273-297.
**Venue.** *Machine Learning* (Springer). **DOI.** 10.1007/BF00994018.
**Annotation.** Original SVM paper. Used in FORMA as a kernel-based baseline classifier for pose-feature vectors, particularly where the 99-dim landmark space may exhibit non-linear class boundaries.
**Suggested quote.** "The support-vector network implements the following idea: it maps the input vectors into some high dimensional feature space Z through some non-linear mapping chosen a priori."

### burges1998tutorial — SVM tutorial
**Harvard.** Burges, C.J.C. (1998) 'A Tutorial on Support Vector Machines for Pattern Recognition', *Data Mining and Knowledge Discovery*, 2(2), pp. 121-167.
**Venue.** *Data Mining and Knowledge Discovery*. **DOI.** 10.1023/A:1009715923555.
**Annotation.** Canonical tutorial reference for SVM theory. Cited when justifying kernel choice and C-parameter tuning for FORMA's SVM baseline.

---

## 3.2  Sequence models: LSTM, BiLSTM, CNN-LSTM

### hochreiter1997lstm — Long Short-Term Memory
**Harvard.** Hochreiter, S. and Schmidhuber, J. (1997) 'Long Short-Term Memory', *Neural Computation*, 9(8), pp. 1735-1780.
**Venue.** *Neural Computation* (MIT Press). **DOI.** 10.1162/neco.1997.9.8.1735.
**Annotation.** Original LSTM paper introducing gated memory cells that solve the vanishing gradient problem in RNNs. Forms the recurrent backbone of FORMA's CNN-BiLSTM sequence model for 30-frame pose windows.
**Suggested quote.** "Long Short-Term Memory (LSTM) can learn to bridge minimal time lags in excess of 1000 discrete time steps by enforcing constant error flow through constant error carrousels within special units."

### schuster1997bidirectional — BiLSTM
**Harvard.** Schuster, M. and Paliwal, K.K. (1997) 'Bidirectional Recurrent Neural Networks', *IEEE Transactions on Signal Processing*, 45(11), pp. 2673-2681.
**Venue.** IEEE TSP. **DOI.** 10.1109/78.650093.
**Annotation.** Introduces bidirectional RNNs that process sequences forward and backward simultaneously. Directly justifies FORMA's BiLSTM layer, which uses past and future pose context within a window to classify the centre frame's form.
**Suggested quote.** "Bidirectional recurrent neural networks (BRNN) can be trained using all available input information in the past and future of a specific time frame."

### karim2017lstm — LSTM-FCN
**Harvard.** Karim, F., Majumdar, S., Darabi, H. and Chen, S. (2018) 'LSTM Fully Convolutional Networks for Time Series Classification', *IEEE Access*, 6, pp. 1662-1669.
**Venue.** IEEE Access. **DOI.** 10.1109/ACCESS.2017.2779939. **arXiv.** 1709.05206.
**Annotation.** LSTM-FCN combines Conv1D feature extractors with an LSTM branch for univariate time-series classification — the direct architectural ancestor of FORMA's Conv1D front-end + BiLSTM + attention stack.
**Suggested quote.** "We propose the augmentation of fully convolutional networks with long short term memory recurrent neural network (LSTM RNN) sub-modules for time series classification."

### wang2015imaging — Imaging time-series for CNNs
**Harvard.** Wang, Z. and Oates, T. (2015) 'Imaging Time-Series to Improve Classification and Imputation', in *Proceedings of the 24th International Joint Conference on Artificial Intelligence (IJCAI)*, pp. 3939-3945.
**Venue.** IJCAI 2015. **arXiv.** 1506.00327.
**Annotation.** Foundational reference for CNN-based time-series classification. Supports FORMA's design choice of Conv1D layers over raw landmark sequences to extract local temporal patterns before recurrent processing.

---

## 3.3  Attention

### bahdanau2015neural — Additive attention
**Harvard.** Bahdanau, D., Cho, K. and Bengio, Y. (2015) 'Neural Machine Translation by Jointly Learning to Align and Translate', in *Proceedings of the 3rd International Conference on Learning Representations (ICLR)*.
**Venue.** ICLR 2015. **arXiv.** 1409.0473.
**Annotation.** Introduces additive attention, the first neural attention mechanism. FORMA's BiLSTM uses a Bahdanau-style attention pool over time steps to emphasise informative frames within the 30-frame window (e.g. the bottom of a squat).
**Suggested quote.** "We propose to extend this by allowing a model to automatically (soft-)search for parts of a source sentence that are relevant to predicting a target word."

### luong2015effective — Multiplicative attention
**Harvard.** Luong, M.-T., Pham, H. and Manning, C.D. (2015) 'Effective Approaches to Attention-based Neural Machine Translation', in *Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pp. 1412-1421.
**Venue.** EMNLP 2015. **DOI.** 10.18653/v1/D15-1166. **arXiv.** 1508.04025.
**Annotation.** Proposes multiplicative (dot-product) attention variants simpler than Bahdanau's additive form. Cited alongside Bahdanau 2015 to justify the scoring function used in FORMA's sequence model.

### vaswani2017attention — Transformer / self-attention
**Harvard.** Vaswani, A. *et al.* (2017) 'Attention Is All You Need', in *Advances in Neural Information Processing Systems (NeurIPS)* 30, pp. 5998-6008.
**Venue.** NeurIPS 2017. **arXiv.** 1706.03762.
**Annotation.** The Transformer paper — scaled dot-product and multi-head self-attention. Cited as the theoretical basis for FORMA's attention pooling and as motivation for why attention augments (rather than replaces) the BiLSTM in our hybrid design.

---

## 3.4  Skeleton-based action recognition

### yan2018stgcn — ST-GCN
**Harvard.** Yan, S., Xiong, Y. and Lin, D. (2018) 'Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition', in *Proceedings of the AAAI Conference on Artificial Intelligence*, 32(1), pp. 7444-7452.
**Venue.** AAAI 2018. **DOI.** 10.1609/aaai.v32i1.12328. **arXiv.** 1801.07455.
**Annotation.** ST-GCN is the canonical skeleton-based action recognition model. Cited as the state-of-the-art baseline for pose-sequence classification, and to justify FORMA's simpler BiLSTM choice (lower latency, easier to train on per-exercise datasets).

### duan2022poseconv3d — PoseConv3D
**Harvard.** Duan, H., Zhao, Y., Chen, K., Lin, D. and Dai, B. (2022) 'Revisiting Skeleton-based Action Recognition', in *Proceedings of the IEEE/CVF CVPR*, pp. 2969-2978.
**Venue.** CVPR 2022. **DOI.** 10.1109/CVPR52688.2022.00298. **arXiv.** 2104.13586.
**Annotation.** PoseConv3D casts skeleton sequences as 3D volumes and applies 3D CNNs, outperforming ST-GCN. Cited to show FORMA is aware of current SOTA and to justify the lighter BiLSTM choice for real-time inference on commodity webcams.

---

## 3.5  Losses + regularisation tied to architecture

### lin2017focal — Focal Loss
**Harvard.** Lin, T.-Y., Goyal, P., Girshick, R., He, K. and Dollár, P. (2017) 'Focal Loss for Dense Object Detection', in *Proceedings of the IEEE ICCV*, pp. 2980-2988.
**Venue.** ICCV 2017. **DOI.** 10.1109/ICCV.2017.324. **arXiv.** 1708.02002.
**Annotation.** Introduces Focal Loss `FL(pt) = -α(1-pt)^γ log(pt)`, used in FORMA's CNN-BiLSTM training to handle the correct/incorrect imbalance. Down-weights easy examples and focuses training on hard negatives.
**Suggested quote.** "Focal Loss reshapes the cross-entropy loss such that it down-weights the loss assigned to well-classified examples."

### izmailov2018swa — Stochastic Weight Averaging
**Harvard.** Izmailov, P., Podoprikhin, D., Garipov, T., Vetrov, D. and Wilson, A.G. (2018) 'Averaging Weights Leads to Wider Optima and Better Generalization', in *Proceedings of UAI 2018*, pp. 876-885.
**Venue.** UAI 2018. **arXiv.** 1803.05407.
**Annotation.** SWA averages weights collected along the SGD trajectory, finding wider and flatter minima. FORMA's training pipeline invokes SWA in the final epochs of cosine annealing.

### zhang2018mixup — mixup
**Harvard.** Zhang, H., Cissé, M., Dauphin, Y.N. and Lopez-Paz, D. (2018) 'mixup: Beyond Empirical Risk Minimization', in *Proceedings of ICLR 2018*.
**Venue.** ICLR 2018. **arXiv.** 1710.09412.
**Annotation.** Introduces mixup: convex combinations of input pairs and their labels. FORMA applies mixup to pose-landmark sequences and one-hot labels (with a Beta(α, α) mixing coefficient) as a regulariser against memorisation.

### szegedy2016rethinking — Label smoothing (Inception-v3)
**Harvard.** Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J. and Wojna, Z. (2016) 'Rethinking the Inception Architecture for Computer Vision', in *Proceedings of IEEE CVPR*, pp. 2818-2826.
**Venue.** CVPR 2016. **DOI.** 10.1109/CVPR.2016.308. **arXiv.** 1512.00567.
**Annotation.** Inception-v3 paper introducing label smoothing regularisation (LSR). FORMA applies label smoothing ε = 0.1 to the binary correct/incorrect targets to discourage overconfident predictions.

### muller2019when — When does label smoothing help?
**Harvard.** Müller, R., Kornblith, S. and Hinton, G.E. (2019) 'When Does Label Smoothing Help?', in *NeurIPS* 32, pp. 4694-4703.
**Venue.** NeurIPS 2019. **arXiv.** 1906.02629.
**Annotation.** Empirical analysis of when label smoothing helps and when it hurts (notably for knowledge distillation). Cited to motivate FORMA's ablation comparing BCE with and without label smoothing.

---

## 3.6  Interpretability vs opaque models (why FSMs beat networks for user-facing feedback)

### rudin2019stop — Interpretable models for high-stakes decisions
**Harvard.** Rudin, C. (2019) 'Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead', *Nature Machine Intelligence*, 1(5), pp. 206-215.
**Venue.** *Nature Machine Intelligence*. **DOI.** 10.1038/s42256-019-0048-x. **arXiv.** 1811.10154.
**Annotation.** Argues interpretable models should be preferred over post-hoc explanations in high-stakes settings. Cited to justify FORMA's primary use of rule-based angle-threshold detectors for user-facing feedback, with the CNN-BiLSTM serving as a secondary classifier.
**Suggested quote.** "Explainable ML methods provide explanations that are not faithful to what the original model computes ... rather than trying to create models that are inherently interpretable."

### doshi2017towards — Towards a rigorous science of interpretable ML
**Harvard.** Doshi-Velez, F. and Kim, B. (2017) 'Towards a Rigorous Science of Interpretable Machine Learning', *arXiv preprint* arXiv:1702.08608.
**Annotation.** Position paper formalising interpretability research. Cited alongside Rudin 2019 to motivate FORMA's per-joint feedback (colour-coded landmarks, one-error-at-a-time) rather than opaque class probabilities.

---

## 3.7  Action quality assessment (continuous form scoring)

### parmar2017learning — Learning to Score Olympic Events
**Harvard.** Parmar, P. and Morris, B.T. (2017) 'Learning to Score Olympic Events', in *Proceedings of IEEE CVPRW*, pp. 76-84.
**Venue.** CVPRW 2017. **DOI.** 10.1109/CVPRW.2017.16. **arXiv.** 1611.05125.
**Annotation.** One of the first CNN-based action-quality-assessment (AQA) works. Directly relevant to FORMA as a precedent for continuous form scoring (0-100) rather than binary classification.

### parmar2019what — Multitask AQA
**Harvard.** Parmar, P. and Morris, B.T. (2019) 'What and How Well You Performed? A Multitask Learning Approach to Action Quality Assessment', in *Proceedings of IEEE/CVF CVPR*, pp. 304-313.
**Venue.** CVPR 2019. **DOI.** 10.1109/CVPR.2019.00039. **arXiv.** 1904.04346.
**Annotation.** Multitask AQA combining action recognition, commentary, and score regression. Cited to position FORMA's per-rep form scoring within the AQA literature and to discuss the tradeoff between binary classification and regression targets.

---

**Gaps in this section** — Logistic regression has no single canonical citation for pose classification; fall back to Hosmer, Lemeshow & Sturdivant (2013) *Applied Logistic Regression* if a general citation is needed.
