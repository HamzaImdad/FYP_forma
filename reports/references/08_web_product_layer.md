# Topic 8 — Web Product Layer

Sources for FORMA's auth, LLM function calling + RAG, streaming, rate
limiting, gamification + SDT, adaptive plans, and prompt-injection defence.

---

## 8.1  Auth + cookies

### jones2015jwt
**Harvard.** Jones, M., Bradley, J. and Sakimura, N. (2015) *JSON Web Token (JWT)*. RFC 7519. Internet Engineering Task Force. Available at: https://www.rfc-editor.org/rfc/rfc7519 (Accessed 19 April 2026).
**Annotation.** Canonical JWT standard — defines claim set and compact serialisation. FORMA encodes `user_id` in a signed JWT carried as an HttpOnly cookie; iss/sub/exp semantics follow this RFC.
**Suggested quote.** "JSON Web Token (JWT) is a compact, URL-safe means of representing claims to be transferred between two parties."

### jones2015jws
**Harvard.** Jones, M., Bradley, J. and Sakimura, N. (2015) *JSON Web Signature (JWS)*. RFC 7515. IETF. Available at: https://www.rfc-editor.org/rfc/rfc7515.
**Annotation.** Specifies the integrity-protected signed representation underpinning JWT. FORMA signs JWTs with HS256 as registered in JWS.

### jones2015jwe
**Harvard.** Jones, M. and Hildebrand, J. (2015) *JSON Web Encryption (JWE)*. RFC 7516. IETF. Available at: https://www.rfc-editor.org/rfc/rfc7516.
**Annotation.** Companion to JWS covering encrypted payloads. Cited to justify FORMA's design choice of signed-but-not-encrypted tokens (payload is non-sensitive `user_id`).

### barth2011cookies
**Harvard.** Barth, A. (2011) *HTTP State Management Mechanism*. RFC 6265. IETF. Available at: https://www.rfc-editor.org/rfc/rfc6265.
**Annotation.** The cookie standard. Defines HttpOnly, Secure, Path/Domain scoping used by FORMA's auth cookie.
**Suggested quote.** "The HttpOnly attribute limits the scope of the cookie to HTTP requests."

### west2020samesite
**Harvard.** West, M. and Wilander, J. (2020) *Cookies: HTTP State Management Mechanism (6265bis)*. IETF Internet-Draft `draft-ietf-httpbis-rfc6265bis`.
**Annotation.** Successor draft to RFC 6265 normalising `SameSite=Lax/Strict/None` and "Lax-by-default". FORMA sets `SameSite=Lax` to mitigate CSRF on state-changing endpoints.

### owasp2024cheatsheet — Practitioner reference
**Harvard.** OWASP Foundation (2024) *Session Management Cheat Sheet*. OWASP Cheat Sheet Series. Available at: https://cheatsheetseries.owasp.org/cheatsheets/Session_Management_Cheat_Sheet.html.
**Annotation.** Practitioner reference recommending HttpOnly + Secure + SameSite cookie configuration. Used to justify FORMA's cookie flags in the security section. Cite as grey literature alongside the RFCs above.

---

## 8.2  LLM tool use / function calling

### schick2023toolformer
**Harvard.** Schick, T. *et al.* (2023) 'Toolformer: Language Models Can Teach Themselves to Use Tools', in *NeurIPS* 36.
**arXiv.** 2302.04761.
**Annotation.** Introduces self-supervised tool-use training. Academic grounding for FORMA's GPT-4o function-calling layer (9 user-scoped tools in `chat_tools.py`).

### yao2023react
**Harvard.** Yao, S. *et al.* (2023) 'ReAct: Synergizing Reasoning and Acting in Language Models', in *ICLR 2023*.
**arXiv.** 2210.03629.
**Annotation.** Interleaved reasoning-and-action prompting — foundation for chain-of-thought tool loops. FORMA's plan-creator chatbot follows a ReAct-style thought → call → observe loop.

---

## 8.3  Retrieval-Augmented Generation (RAG)

### lewis2020rag
**Harvard.** Lewis, P. *et al.* (2020) 'Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks', in *NeurIPS* 33, pp. 9459-9474.
**arXiv.** 2005.11401.
**Annotation.** Seminal RAG paper. FORMA's public guide chatbot retrieves top-K chunks from a 64-chunk exercise KB and conditions GPT-4o on them.
**Suggested quote.** "We combine pre-trained parametric and non-parametric memory for language generation."

### borgeaud2022retro
**Harvard.** Borgeaud, S. *et al.* (2022) 'Improving Language Models by Retrieving from Trillions of Tokens', in *ICML 2022*, PMLR 162, pp. 2206-2240.
**arXiv.** 2112.04426.
**Annotation.** DeepMind's RETRO — retrieval at trillion-token scale. Scalability evidence for the RAG pattern.

### guu2020realm
**Harvard.** Guu, K. *et al.* (2020) 'REALM: Retrieval-Augmented Language Model Pre-Training', in *ICML 2020*, PMLR 119, pp. 3929-3938.
**arXiv.** 2002.08909.
**Annotation.** Contemporary alternative retrieval-augmented architecture (trained end-to-end). Shows the RAG design space FORMA considered before adopting decoupled retrieval.

### izacard2023atlas
**Harvard.** Izacard, G. *et al.* (2023) 'Atlas: Few-shot Learning with Retrieval Augmented Language Models', *JMLR*, 24(251), pp. 1-43.
**Annotation.** Strong RAG performance with only a handful of training examples. Supports FORMA's zero-shot deployment of the guide chatbot against a small curated KB.

---

## 8.4  Streaming + rate limiting

### hickson2015sse — Server-Sent Events
**Harvard.** Hickson, I. (ed.) (2015) *Server-Sent Events*. W3C Recommendation. W3C. Available at: https://www.w3.org/TR/eventsource/.
**Annotation.** Canonical spec for `text/event-stream` + `EventSource` API used by FORMA's `/api/chat/*` endpoints.
**Suggested quote.** "This specification defines an API for opening an HTTP connection for receiving push notifications from a server."

### turner1986leakybucket
**Harvard.** Turner, J.S. (1986) 'New directions in communications (or which way to the information age?)', *IEEE Communications Magazine*, 24(10), pp. 8-15.
**DOI.** 10.1109/MCOM.1986.1092946.
**Annotation.** Origin paper for the leaky-bucket traffic-shaping metaphor. Theoretical basis for FORMA's per-user rate limiting (30/hour on `/api/chat/plan`).

### shenker1995fundamental — Token-bucket policing
**Harvard.** Shenker, S. (1995) 'Fundamental design issues for the future Internet', *IEEE J. Sel. Areas Commun.*, 13(7), pp. 1176-1188.
**DOI.** 10.1109/49.414637.
**Annotation.** Foundational paper covering token-bucket policing. Supports FORMA's burst-tolerant rate limiting.

---

## 8.5  Gamification + SDT

### deterding2011gamification
**Harvard.** Deterding, S., Dixon, D., Khaled, R. and Nacke, L. (2011) 'From game design elements to gamefulness: defining "gamification"', in *Proceedings of MindTrek 2011*. New York: ACM, pp. 9-15.
**DOI.** 10.1145/2181037.2181040.
**Annotation.** Definitive academic definition of gamification. FORMA's 13 badges, streaks, and milestone system are direct instances of "game design elements in non-game contexts".
**Suggested quote.** "Gamification is the use of game design elements in non-game contexts."

### hamari2014does
**Harvard.** Hamari, J., Koivisto, J. and Sarsa, H. (2014) 'Does Gamification Work? — A Literature Review of Empirical Studies on Gamification', in *HICSS 2014*. IEEE, pp. 3025-3034.
**DOI.** 10.1109/HICSS.2014.377.
**Annotation.** Meta-review: gamification has positive-but-qualified effects, context-dependent. Justifies FORMA's conservative gamification layer (badges + streaks, no competitive leaderboard).

### ryan2000sdt
**Harvard.** Ryan, R.M. and Deci, E.L. (2000) 'Self-determination theory and the facilitation of intrinsic motivation, social development, and well-being', *American Psychologist*, 55(1), pp. 68-78.
**DOI.** 10.1037/0003-066X.55.1.68.
**Annotation.** Foundational SDT paper (autonomy, competence, relatedness). Theoretical anchor for why FORMA shows progress milestones and choice-preserving plan editing rather than prescriptive programmes.

### deci1985intrinsic
**Harvard.** Deci, E.L. and Ryan, R.M. (1985) *Intrinsic Motivation and Self-Determination in Human Behavior*. New York: Plenum.
**DOI.** 10.1007/978-1-4899-2271-7.
**Annotation.** Book-length foundation for SDT. Used for the motivational framing of FORMA's streak and goal system.

### cugelman2013gamification
**Harvard.** Cugelman, B. (2013) 'Gamification: What It Is and Why It Matters to Digital Health Behavior Change Developers', *JMIR Serious Games*, 1(1), e3.
**DOI.** 10.2196/games.3139.
**Annotation.** Peer-reviewed review linking gamification mechanics to behaviour-change outcomes in health apps.

### johnson2016gamification
**Harvard.** Johnson, D. *et al.* (2016) 'Gamification for health and wellbeing: A systematic review of the literature', *Internet Interventions*, 6, pp. 89-106.
**DOI.** 10.1016/j.invent.2016.10.002.
**Annotation.** Systematic review supporting positive efficacy of gamification for health behaviour. Cited as evidence base for FORMA's badge/streak strategy.

---

## 8.6  Adaptive plans + RPE

### kraemer2004progressive
**Harvard.** Kraemer, W.J. and Ratamess, N.A. (2004) 'Fundamentals of resistance training: progression and exercise prescription', *Med. Sci. Sports Exerc.*, 36(4), pp. 674-688.
**DOI.** 10.1249/01.MSS.0000121945.36635.61.
**Annotation.** Progressive-overload reference; FORMA's plan engine increases load/volume across weeks.
**Note.** Shared key with `07_exercise_science.md` → use the same BibTeX entry (`kraemer2004fundamentals`).

### helms2016rpe
**Harvard.** Helms, E.R., Cronin, J., Storey, A. and Zourdos, M.C. (2016) 'Application of the Repetitions in Reserve-Based Rating of Perceived Exertion Scale for Resistance Training', *Strength and Conditioning Journal*, 38(4), pp. 42-49.
**DOI.** 10.1519/SSC.0000000000000218.
**Annotation.** RIR-RPE methodology for auto-regulated training. Relevant to FORMA's adaptive plan adjustment (sets adjusted based on prior-session form scores as a proxy for readiness).

### zourdos2016rpe
**Harvard.** Zourdos, M.C. *et al.* (2016) 'Novel Resistance Training-Specific Rating of Perceived Exertion Scale Measuring Repetitions in Reserve', *J. Strength Cond. Res.*, 30(1), pp. 267-275.
**DOI.** 10.1519/JSC.0000000000001049.
**Annotation.** Validation study for the RIR-RPE scale. Supports FORMA's use of proxy-readiness signals when auto-regulating plan difficulty.

---

## 8.7  Prompt injection / LLM security

### perez2022ignore
**Harvard.** Perez, F. and Ribeiro, I. (2022) 'Ignore Previous Prompt: Attack Techniques For Language Models', in *NeurIPS 2022 ML Safety Workshop*.
**arXiv.** 2211.09527.
**Annotation.** First systematic treatment of prompt-injection attacks. FORMA's chatbots enforce role-locked system prompts, a sanitiser, and tool-call allow-lists.
**Suggested quote.** "We find that it is possible to hijack large language models via carefully crafted adversarial prompts."

### greshake2023indirect
**Harvard.** Greshake, K. *et al.* (2023) 'Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection', in *Proceedings of AISec 2023*. New York: ACM, pp. 79-90.
**DOI.** 10.1145/3605764.3623985.
**Annotation.** Covers indirect prompt injection via retrieved documents — directly relevant to FORMA's RAG guide chatbot.

---

**Gaps in this section** — Token-budgeting LLM serving: no verified peer-reviewed citation; suggested substitute is **Chen et al. 2023 "FrugalGPT"** (arXiv:2305.05176). LLM streaming-UX: no peer-reviewed source located; fall back to Nielsen Norman Group perceived-performance articles cited in topic 09.
