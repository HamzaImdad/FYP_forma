# Topic 5 — Real-Time System Design

Sources for FORMA's transport choice (Flask + Socket.IO / WebSocket), the
client-side MediaPipe pipeline (WASM + WebGL + TF.js), response-time
thresholds, video codec choices, and consumer-hardware performance bounds.

---

## 5.1  Transport: WebSocket vs WebRTC vs HTTP polling

### fette2011websocket
**Harvard.** Fette, I. and Melnikov, A. (2011) *The WebSocket Protocol*. RFC 6455. IETF. Available at: https://www.rfc-editor.org/rfc/rfc6455.
**Annotation.** Defines the WebSocket protocol — full-duplex, low-overhead alternative to HTTP polling. Canonical standards citation used to justify FORMA's choice of Socket.IO (a WebSocket-based transport) over long-polling for 10-30 Hz pose data.
**Suggested quote.** "provide a mechanism for browser-based applications that need two-way communication with servers that does not rely on opening multiple HTTP connections."

### alvestrand2021rfc8825
**Harvard.** Alvestrand, H.T. (2021) *Overview: Real-Time Protocols for Browser-Based Applications*. RFC 8825. IETF. Available at: https://www.rfc-editor.org/rfc/rfc8825.
**Annotation.** Normative overview of the WebRTC protocol suite (SRTP, ICE, DTLS, SCTP). Establishes the reference architecture against which FORMA's chosen Flask + Socket.IO transport is compared and justified as simpler and sufficient for unidirectional landmark streaming.

### jansen2018webrtc
**Harvard.** Jansen, B. *et al.* (2018) 'Performance evaluation of WebRTC-based video conferencing', *ACM SIGMETRICS PER*, 45(3), pp. 56-68.
**DOI.** 10.1145/3199524.3199534.
**Annotation.** Empirical latency/packet-loss study of WebRTC under varying RTT. Used to defend the FORMA decision to run MediaPipe client-side (no media upstream over WebRTC) and send only landmark tensors over WebSocket.

### pimentel2012websocket
**Harvard.** Pimentel, V. and Nickerson, B.G. (2012) 'Communicating and Displaying Real-Time Data with WebSocket', *IEEE Internet Computing*, 16(4), pp. 45-53.
**DOI.** 10.1109/MIC.2012.64.
**Annotation.** Empirical comparison of WebSocket vs HTTP polling for a live sensor feed, measuring one-way transmission latency. Cited to support FORMA's choice of WebSocket (Socket.IO) over polling for the pose-landmark stream.

### grigorik2013hpbn
**Harvard.** Grigorik, I. (2013) *High Performance Browser Networking*. Sebastopol, CA: O'Reilly Media.
**Annotation.** Engineering reference covering TCP, TLS, HTTP/2, SSE, WebSocket and WebRTC latency budgets. Used to contextualise FORMA's sub-100 ms round-trip target on LAN/Wi-Fi.

---

## 5.2  Browser-side ML inference (MediaPipe + WASM + WebGL)

### haas2017webassembly
**Harvard.** Haas, A. *et al.* (2017) 'Bringing the web up to speed with WebAssembly', in *Proceedings of PLDI 2017*. New York: ACM, pp. 185-200.
**DOI.** 10.1145/3062341.3062363.
**Annotation.** Foundational design paper for WebAssembly — the near-native compilation target that MediaPipe Tasks use to run BlazePose inference inside the browser.
**Suggested quote.** "WebAssembly is the first truly portable native target for the web ... safe, fast, portable low-level code on the web."

### smilkov2019tensorflowjs
**Harvard.** Smilkov, D. *et al.* (2019) 'TensorFlow.js: Machine learning for the web and beyond', in *MLSys 2019*, pp. 309-321.
**Annotation.** Architecture paper for TensorFlow.js, including WebGL and WebAssembly back-ends that MediaPipe Pose Landmarker Web build upon. Frames FORMA's browser-based ML inference against an established peer-reviewed benchmark.

### lugaresi2019mediapipe
**Harvard.** Lugaresi, C. *et al.* (2019) 'MediaPipe: A Framework for Building Perception Pipelines', *arXiv preprint* arXiv:1906.08172.
**Annotation.** Describes the cross-platform dataflow graph underlying MediaPipe Tasks (Python, Android, iOS, Web). Source for FORMA's pose pipeline, including calculator abstraction and GPU/WebGL scheduling.

### ignatov2019aibenchmark
**Harvard.** Ignatov, A. *et al.* (2019) 'AI Benchmark: All about deep learning on smartphones in 2019', in *IEEE/CVF ICCVW*, pp. 3617-3635.
**DOI.** 10.1109/ICCVW.2019.00447.
**Annotation.** Large-scale measurement study of NN inference latency on 10 000+ devices and 50+ SoCs. Establishes a realistic ceiling of 15-25 FPS on mid-range phones for FORMA.

---

## 5.3  Video codec for session recording

### wiegand2003h264
**Harvard.** Wiegand, T., Sullivan, G.J., Bjøntegaard, G. and Luthra, A. (2003) 'Overview of the H.264/AVC Video Coding Standard', *IEEE TCSVT*, 13(7), pp. 560-576.
**DOI.** 10.1109/TCSVT.2003.815165.
**Annotation.** Authoritative ITU/ISO overview of H.264/AVC. Referenced to defend trade-offs between WebM/VP8-9 and H.264 in FORMA's 480p 10 fps recording stream.

---

## 5.4  Response-time limits + perceptual fusion

### miller1968response
**Harvard.** Miller, R.B. (1968) 'Response time in man-computer conversational transactions', in *AFIPS '68*. New York: ACM, pp. 267-277.
**DOI.** 10.1145/1476589.1476628.
**Annotation.** First systematic taxonomy of response-time limits. Established the <2 s "instantaneous response" and <0.1 s "continuous-feedback" bands that FORMA cites as its latency budget.
**Suggested quote.** "Responses to 'light pen' hits and other continuous manipulations should be immediately visible — of the order of 0.1 second."

### card1983psychology
**Harvard.** Card, S.K., Moran, T.P. and Newell, A. (1983) *The Psychology of Human-Computer Interaction*. Hillsdale, NJ: Lawrence Erlbaum Associates.
**Annotation.** Formalises the Model Human Processor's perceptual (~100 ms), cognitive (~70 ms) and motor (~70 ms) cycle times. Justifies FORMA's ≤100 ms pose-to-feedback target as the threshold for perceptual fusion.
**Suggested quote.** "Perceptual processor cycle time τ_p ≈ 100 ms (50~200 ms)."

### nielsen1993usability — 0.1 s / 1 s / 10 s rules
**Harvard.** Nielsen, J. (1993) *Usability Engineering*. San Francisco: Morgan Kaufmann.
**Annotation.** Canonical three response-time limits — the single most-cited source for "real-time feel" in HCI. Directly underpins the claim that per-frame feedback must land under 100 ms.

### fitts1954information
**Harvard.** Fitts, P.M. (1954) 'The information capacity of the human motor system in controlling the amplitude of movement', *Journal of Experimental Psychology*, 47(6), pp. 381-391.
**DOI.** 10.1037/h0055392.
**Annotation.** Foundational information-theoretic bound on human motor control (~5 bits/s). Defends FORMA's design decision to emit at most one correction per second; more than this exceeds the user's motor-update bandwidth.

### seow2005information
**Harvard.** Seow, S.C. (2005) 'Information theoretic models of HCI: A comparison of the Hick-Hyman Law and Fitts' Law', *Human-Computer Interaction*, 20(3), pp. 315-352.
**DOI.** 10.1207/s15327051hci2003_3.
**Annotation.** Synthesis of two information-theoretic bounds on HCI performance. Supports FORMA's feedback-rate argument — Hick-Hyman predicts reaction-time growth with the number of simultaneous cues.

---

**Gaps in this section** — No peer-reviewed paper compared Flask + Socket.IO vs WebRTC vs gRPC specifically for fitness/pose use. Assembled from RFC 6455, RFC 8825, Jansen 2018, Pimentel & Nickerson 2012. WebGL-Sørensen 2020 and Castle 2022 could not be verified and were dropped.
