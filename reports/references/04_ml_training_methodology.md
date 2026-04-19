# Topic 4 — ML Training Methodology

Sources for FORMA's training pipeline: video-level splits, class-imbalance
handling, pose-landmark augmentation, cosine LR with warm restarts, linear
warmup, early stopping on val F1, PR/threshold analysis, mixed precision,
gradient clipping, optimisers, and HAR-style sliding-window classification.

---

## 4.1  Data leakage + video-level splits

### kaufman2012leakage — Leakage in data mining
**Harvard.** Kaufman, S., Rosset, S., Perlich, C. and Stitelman, O. (2012) 'Leakage in Data Mining: Formulation, Detection, and Avoidance', *ACM Transactions on Knowledge Discovery from Data*, 6(4), Article 15.
**Venue.** ACM TKDD. **DOI.** 10.1145/2382577.2382579.
**Annotation.** Defines data leakage and its taxonomy. FORMA cites this to justify video-level train/val/test splits: naïve frame-level shuffling would leak within-video appearance cues and inflate F1.
**Suggested quote.** "Leakage in data mining is essentially the introduction of information about the target of a data mining problem that should not be legitimately available to mine from."

### kapoor2023leakage — Reproducibility crisis
**Harvard.** Kapoor, S. and Narayanan, A. (2023) 'Leakage and the reproducibility crisis in machine-learning-based science', *Patterns*, 4(9), Article 100804.
**Venue.** *Patterns* (Cell Press). **DOI.** 10.1016/j.patter.2023.100804. **arXiv.** 2207.07048.
**Annotation.** Surveys 294 papers across 17 fields where leakage caused over-optimistic results. Strong contemporary citation for FORMA's split-by-video methodology and the accompanying "what could have gone wrong" evaluation section.

---

## 4.2  Class imbalance

### he2009learning — Learning from Imbalanced Data
**Harvard.** He, H. and Garcia, E.A. (2009) 'Learning from Imbalanced Data', *IEEE Transactions on Knowledge and Data Engineering*, 21(9), pp. 1263-1284.
**Venue.** IEEE TKDE. **DOI.** 10.1109/TKDE.2008.239.
**Annotation.** Canonical survey of class-imbalance techniques. Used in FORMA to frame the correct-vs-incorrect imbalance problem and motivate the use of SMOTE-style oversampling plus Focal Loss.

### chawla2002smote — SMOTE
**Harvard.** Chawla, N.V., Bowyer, K.W., Hall, L.O. and Kegelmeyer, W.P. (2002) 'SMOTE: Synthetic Minority Over-sampling Technique', *Journal of Artificial Intelligence Research*, 16, pp. 321-357.
**Venue.** JAIR. **DOI.** 10.1613/jair.953.
**Annotation.** SMOTE is the most-cited oversampling technique. FORMA's `balance_data.py` oversamples the minority "incorrect" class by synthesising interpolated pose-feature vectors in this spirit.

### kubat1997addressing — One-sided undersampling
**Harvard.** Kubat, M. and Matwin, S. (1997) 'Addressing the Curse of Imbalanced Training Sets: One-Sided Selection', in *Proceedings of ICML 1997*, pp. 179-186.
**Venue.** ICML 1997.
**Annotation.** Foundational one-sided-undersampling paper. Cited for FORMA's majority-class undersampling step in `balance_data.py`.

---

## 4.3  Pose-landmark augmentation

### rao2021augmented — AS-CAL augmented skeleton learning
**Harvard.** Rao, H., Xu, S., Hu, X., Cheng, J. and Hu, B. (2021) 'Augmented Skeleton Based Contrastive Action Learning with Momentum LSTM for Unsupervised Action Recognition', *Information Sciences*, 569, pp. 90-109.
**Venue.** *Information Sciences* (Elsevier). **DOI.** 10.1016/j.ins.2021.04.023. **arXiv.** 2008.00188.
**Annotation.** Proposes seven skeleton-specific augmentations (rotation, shear, Gaussian noise, Gaussian blur, channel masking, spatial flip, temporal crop). Direct source for FORMA's pose-landmark augmentation set (noise, scaling, joint dropout, translation, L-R mirror, time warp).

### gong2021poseaug — PoseAug
**Harvard.** Gong, K., Zhang, J. and Feng, J. (2021) 'PoseAug: A Differentiable Pose Augmentation Framework for 3D Human Pose Estimation', in *Proceedings of IEEE/CVF CVPR*, pp. 8575-8584.
**Venue.** CVPR 2021. **DOI.** 10.1109/CVPR46437.2021.00847. **arXiv.** 2105.02465.
**Annotation.** PoseAug adjusts posture, body size, viewpoint and position via differentiable operations. Cited to situate FORMA's simpler hand-crafted pose augmentations within the learned-augmentation literature.

---

## 4.4  Learning-rate schedules + warmup

### loshchilov2017sgdr — Cosine annealing with warm restarts
**Harvard.** Loshchilov, I. and Hutter, F. (2017) 'SGDR: Stochastic Gradient Descent with Warm Restarts', in *Proceedings of ICLR 2017*.
**Venue.** ICLR 2017. **arXiv.** 1608.03983.
**Annotation.** Proposes cosine annealing with warm restarts. FORMA's training uses cosine LR with restart periods `(T_0=50, T_mult=2)` to escape saddles and to ensemble via SWA.

### goyal2017accurate — Linear LR warmup
**Harvard.** Goyal, P. *et al.* (2017) 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour', *arXiv preprint* arXiv:1706.02677.
**Annotation.** Popularised linear LR warmup. FORMA uses warmup for the first 5 epochs of training to stabilise the Conv1D front-end before cosine annealing kicks in.

### he2019bag — Bag of Tricks
**Harvard.** He, T., Zhang, Z., Zhang, H., Zhang, Z., Xie, J. and Li, M. (2019) 'Bag of Tricks for Image Classification with Convolutional Neural Networks', in *Proceedings of IEEE/CVF CVPR*, pp. 558-567.
**Venue.** CVPR 2019. **DOI.** 10.1109/CVPR.2019.00065. **arXiv.** 1812.01187.
**Annotation.** Empirical study of training tricks (LR warmup, label smoothing, zero-gamma BN, mixup) whose combination yields substantial accuracy gains. Cited as template for FORMA's training-recipe ablation.

---

## 4.5  Early stopping + evaluation metrics

### prechelt2012early — Early Stopping, But When?
**Harvard.** Prechelt, L. (2012) 'Early Stopping – But When?', in Montavon, G., Orr, G.B. and Müller, K.-R. (eds.) *Neural Networks: Tricks of the Trade*. 2nd edn. LNCS 7700. Berlin: Springer, pp. 53-67.
**DOI.** 10.1007/978-3-642-35289-8_5.
**Annotation.** Classic reference for early-stopping heuristics (GL, PQ, UP). FORMA stops on val-F1 plateau (patience=50 epochs) following this paper.

### powers2011evaluation — F1/P/R definitions
**Harvard.** Powers, D.M.W. (2011) 'Evaluation: From precision, recall and F-measure to ROC, informedness, markedness and correlation', *Journal of Machine Learning Technologies*, 2(1), pp. 37-63.
**arXiv.** 2010.16061.
**Annotation.** Canonical reference for precision/recall/F1 definitions and their limitations. FORMA cites this to justify reporting F1 as the primary metric given the imbalance between correct/incorrect classes.

### saito2015precision — PR curves > ROC under imbalance
**Harvard.** Saito, T. and Rehmsmeier, M. (2015) 'The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets', *PLoS ONE*, 10(3), e0118432.
**DOI.** 10.1371/journal.pone.0118432.
**Annotation.** Shows PR curves are more informative than ROC under class imbalance. FORMA reports PR curves per exercise (alongside F1) given the minority of incorrect-form samples in some exercises.

### lipton2014optimal — Optimal F1 thresholding
**Harvard.** Lipton, Z.C., Elkan, C. and Naryanaswamy, B. (2014) 'Optimal Thresholding of Classifiers to Maximize F1 Measure', in *ECML PKDD 2014*. LNCS 8725. Berlin: Springer, pp. 225-239.
**DOI.** 10.1007/978-3-662-44851-9_15. **arXiv.** 1402.1892.
**Annotation.** Formal justification for F1-maximising threshold search on the validation set. FORMA's per-exercise decision thresholds (e.g. squat 0.15, tricep_dip 0.50) are selected by sweeping the val PR curve per this paper.

---

## 4.6  Cross-exercise transfer learning

### carreira2017i3d — I3D + Kinetics
**Harvard.** Carreira, J. and Zisserman, A. (2017) 'Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset', in *Proceedings of IEEE CVPR*, pp. 4724-4733.
**Venue.** CVPR 2017. **DOI.** 10.1109/CVPR.2017.502. **arXiv.** 1705.07750.
**Annotation.** I3D and Kinetics established large-scale pretrain-then-finetune as the dominant action-recognition paradigm. Cited to motivate FORMA's exploration of cross-exercise pretraining for low-data exercises (pullup, plank).

---

## 4.7  Optimisation internals

### micikevicius2018mixed — Mixed precision
**Harvard.** Micikevicius, P. *et al.* (2018) 'Mixed Precision Training', in *Proceedings of ICLR 2018*.
**arXiv.** 1710.03740.
**Annotation.** Introduces FP16/FP32 mixed precision with loss scaling. FORMA uses `torch.cuda.amp` autocast + GradScaler on the RTX 4070 to roughly halve training time without loss of F1.

### pascanu2013difficulty — Gradient clipping
**Harvard.** Pascanu, R., Mikolov, T. and Bengio, Y. (2013) 'On the difficulty of training recurrent neural networks', in *Proceedings of ICML 2013*, pp. 1310-1318.
**arXiv.** 1211.5063.
**Annotation.** Formal analysis of exploding/vanishing gradients in RNNs and the introduction of gradient norm clipping. FORMA clips gradients at ||g||=1.0 during CNN-BiLSTM training per this paper.

### kingma2015adam — Adam optimiser
**Harvard.** Kingma, D.P. and Ba, J. (2015) 'Adam: A Method for Stochastic Optimization', in *Proceedings of ICLR 2015*.
**arXiv.** 1412.6980.
**Annotation.** Adam combining momentum and RMSProp. Used throughout FORMA training (`lr = 1e-3`, `betas = (0.9, 0.999)`).

### loshchilov2019decoupled — AdamW
**Harvard.** Loshchilov, I. and Hutter, F. (2019) 'Decoupled Weight Decay Regularization', in *Proceedings of ICLR 2019*.
**arXiv.** 1711.05101.
**Annotation.** AdamW: decouples weight decay from the gradient update. FORMA uses AdamW (`weight_decay = 1e-4`) when regularisation matters more than raw convergence.

### srivastava2014dropout — Dropout
**Harvard.** Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I. and Salakhutdinov, R. (2014) 'Dropout: A Simple Way to Prevent Neural Networks from Overfitting', *Journal of Machine Learning Research*, 15, pp. 1929-1958.
**Annotation.** Original dropout paper. FORMA applies dropout p=0.3 between BiLSTM and the classifier head to regularise on small per-exercise datasets.

### ioffe2015batch — Batch Normalization
**Harvard.** Ioffe, S. and Szegedy, C. (2015) 'Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift', in *Proceedings of ICML 2015*, pp. 448-456.
**arXiv.** 1502.03167.
**Annotation.** Introduces BatchNorm. FORMA applies BatchNorm1D after the Conv1D front-end blocks to accelerate convergence on pose-feature distributions that vary between exercises.

---

## 4.8  HAR / sliding-window classification

### bulling2014tutorial — HAR tutorial
**Harvard.** Bulling, A., Blanke, U. and Schiele, B. (2014) 'A tutorial on human activity recognition using body-worn inertial sensors', *ACM Computing Surveys*, 46(3), Article 33.
**Venue.** ACM CSUR. **DOI.** 10.1145/2499621.
**Annotation.** Canonical HAR tutorial defining the Activity Recognition Chain (segmentation, feature extraction, classification). Cited to justify FORMA's sliding-window approach with velocity/acceleration features in temporal 330-dim mode.

### wang2019deeplearning — Deep-learning HAR survey
**Harvard.** Wang, J., Chen, Y., Hao, S., Peng, X. and Hu, L. (2019) 'Deep learning for sensor-based activity recognition: A survey', *Pattern Recognition Letters*, 119, pp. 3-11.
**DOI.** 10.1016/j.patrec.2018.02.010. **arXiv.** 1707.03502.
**Annotation.** Survey of deep learning for HAR covering CNN, RNN, hybrid, and autoencoder architectures. Cited to situate FORMA's CNN-BiLSTM within the broader HAR deep-learning taxonomy.

### hammerla2016deep — BiLSTM vs CNN benchmarks
**Harvard.** Hammerla, N.Y., Halloran, S. and Plötz, T. (2016) 'Deep, Convolutional, and Recurrent Models for Human Activity Recognition using Wearables', in *Proceedings of IJCAI 2016*, pp. 1533-1540.
**arXiv.** 1604.08880.
**Annotation.** Benchmarks CNN, LSTM, and bidirectional LSTM on HAR datasets. Their finding that BiLSTM wins for short-duration activities directly supports FORMA's BiLSTM choice for sub-second movements (e.g. rep bottom).
**Suggested quote.** "Bi-directional LSTMs perform best in tasks where the data naturally has short-duration, atomic activities."

### ordonez2016deep — DeepConvLSTM
**Harvard.** Ordóñez, F.J. and Roggen, D. (2016) 'Deep Convolutional and LSTM Recurrent Neural Networks for Multimodal Wearable Activity Recognition', *Sensors*, 16(1), Article 115.
**DOI.** 10.3390/s16010115.
**Annotation.** DeepConvLSTM — the direct architectural template for FORMA's CNN-BiLSTM. Applies Conv1D feature extractors to raw sensor windows followed by LSTM layers for temporal fusion.
