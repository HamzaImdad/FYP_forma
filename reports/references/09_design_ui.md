# Topic 9 — Design / UI

Sources for FORMA's UI decisions: WCAG, dark-mode UX, typography,
microinteractions, scrollytelling, `prefers-reduced-motion`, progressive
disclosure, response-time thresholds, emotional design.

---

## 9.1  Accessibility + dark mode

### wcag21
**Harvard.** W3C (2018) *Web Content Accessibility Guidelines (WCAG) 2.1*. W3C Recommendation, 5 June 2018. Edited by A. Kirkpatrick, J. O'Connor, A. Campbell and M. Cooper. Available at: https://www.w3.org/TR/WCAG21/ (Accessed 19 April 2026).
**Annotation.** The normative accessibility standard. FORMA's dark-mode palette (#0A0A0A background, #AEE710 accent) is audited against WCAG 1.4.3 (contrast minimum 4.5:1 body, 3:1 large).
**Suggested quote.** "The visual presentation of text ... has a contrast ratio of at least 4.5:1."

### pedersen2020darkmode
**Harvard.** Pedersen, L.A. *et al.* (2020) 'User Interfaces in Dark Mode During Daytime — Improved Productivity or Just Cool-Looking?', in *UAHCI 2020*. LNCS 12188. Cham: Springer, pp. 178-187.
**DOI.** 10.1007/978-3-030-49282-3_13.
**Annotation.** Empirical study comparing dark/light mode productivity. Supports nuanced claim that dark mode is a legitimate accessibility/aesthetic option but not universally better for readability.

### erickson2020dark
**Harvard.** Erickson, A., Kim, K., Bruder, G. and Welch, G.F. (2020) 'Effects of Dark Mode Graphics on Visual Acuity and Fatigue with Virtual Reality Head-Mounted Displays', in *IEEE VR 2020*. IEEE, pp. 434-442.
**DOI.** 10.1109/VR46266.2020.00064.
**Annotation.** Peer-reviewed evidence on dark-mode visual comfort. Useful for FORMA's accessibility discussion.

### mediaqueries5 — prefers-reduced-motion / prefers-color-scheme
**Harvard.** Jackson, D., Mogilevsky, D. and Rivoal, F. (eds.) (2021) *Media Queries Level 5*. W3C Working Draft. W3C. Available at: https://www.w3.org/TR/mediaqueries-5/.
**Annotation.** Defines `prefers-reduced-motion` and `prefers-color-scheme`. FORMA respects `prefers-reduced-motion` to disable GSAP/Framer Motion entrance animations.
**Suggested quote.** "The prefers-reduced-motion media feature is used to detect if the user has requested the system minimize the amount of non-essential motion it uses."

### ahmadian2023vestibular
**Harvard.** Ahmadian, L. *et al.* (2023) 'Web accessibility for users with vestibular disorders: A scoping review', *Universal Access in the Information Society*, 22, pp. 1-15.
**DOI.** 10.1007/s10209-022-00875-x.
**Annotation.** Supports FORMA's `prefers-reduced-motion` handling for vestibular-sensitive users. If the DOI does not resolve in your library, fall back to WCAG 2.3.3 Animation from Interactions.

---

## 9.2  Typography + grid

### bringhurst2005
**Harvard.** Bringhurst, R. (2005) *The Elements of Typographic Style*. 3rd edn. Point Roberts, WA: Hartley & Marks.
**ISBN.** 9780881792065.
**Annotation.** Canonical typographic reference. Justifies FORMA's scale ratio (major third 1.250), body line-height, and use of Bebas Neue condensed display type.
**Suggested quote.** "Typography exists to honor content."

### muller1981grid
**Harvard.** Müller-Brockmann, J. (1981) *Grid Systems in Graphic Design*. Teufen: Arthur Niggli.
**ISBN.** 9783721201451.
**Annotation.** Swiss-design grid-systems classic. Cited for FORMA's 12-column responsive grid and editorial layout decisions.

### lupton2010thinking
**Harvard.** Lupton, E. (2010) *Thinking with Type: A Critical Guide for Designers, Writers, Editors, & Students*. 2nd edn. New York: Princeton Architectural Press.
**ISBN.** 9781568989693.
**Annotation.** Approachable typographic reference. Supports FORMA's editorial approach: Bebas Neue display, Outfit body, Cormorant Garamond italic accents.

---

## 9.3  Motion + microinteractions

### saffer2013microinteractions
**Harvard.** Saffer, D. (2013) *Microinteractions: Designing with Details*. Sebastopol, CA: O'Reilly Media.
**ISBN.** 9781491945926.
**Annotation.** Foundational reference for microinteractions design. Maps directly to FORMA's button-press feedback, toast notifications, and loading states.
**Suggested quote.** "Microinteractions are the small moments that can either elevate a product from mediocre to memorable, or reduce it to frustrating."

### seyser2018scrollytelling
**Harvard.** Seyser, D. and Zeiller, M. (2018) 'Scrollytelling — An Analysis of Visual Storytelling in Online Journalism', in *IEEE Information Visualisation (IV)*. IEEE, pp. 401-406.
**DOI.** 10.1109/iV.2018.00075.
**Annotation.** Peer-reviewed analysis of scroll-driven storytelling. Supports FORMA's GSAP ScrollTrigger-based narrative landing page.

### harley2014animation — NNG practitioner reference
**Harvard.** Harley, A. (2014) *Animation for Attention and Comprehension*. Nielsen Norman Group. Available at: https://www.nngroup.com/articles/animation-attention/.
**Annotation.** NNG's practitioner reference on purposeful animation. Cite as the authority on "motion that communicates, not decorates".

---

## 9.4  Interaction + information architecture

### frost2016atomic
**Harvard.** Frost, B. (2016) *Atomic Design*. Pittsburgh, PA: Brad Frost. Available at: https://atomicdesign.bradfrost.com/.
**Annotation.** Atoms → molecules → organisms → templates → pages. FORMA's React component hierarchy mirrors this.

### nielsen2006progressive
**Harvard.** Nielsen, J. (2006) *Progressive Disclosure*. Nielsen Norman Group. Available at: https://www.nngroup.com/articles/progressive-disclosure/.
**Annotation.** Defines progressive disclosure. Matches FORMA's dashboard drill-down (chip → deep-dive → session → rep) and the collapsible EditPlanDayModal.

### nielsen1993usability — Response-time rules
**Harvard.** Nielsen, J. (1993) *Usability Engineering*. San Francisco, CA: Morgan Kaufmann.
**DOI.** 10.1016/C2009-0-21512-1.
**Annotation.** Authoritative source for the 0.1 s / 1.0 s / 10 s response-time rules used to justify FORMA's skeleton loaders and streaming chat.
**Suggested quote.** "0.1 second is about the limit for having the user feel that the system is reacting instantaneously."

### card1991info
**Harvard.** Card, S.K., Robertson, G.G. and Mackinlay, J.D. (1991) 'The Information Visualizer, an information workspace', in *Proceedings of ACM CHI*. New York: ACM, pp. 181-186.
**DOI.** 10.1145/108844.108874.
**Annotation.** Classic source for the 0.1/1/10-second perceptual thresholds underpinning perceived-performance design.

### shneiderman1996eyes — Information-seeking mantra
**Harvard.** Shneiderman, B. (1996) 'The eyes have it: a task by data type taxonomy for information visualizations', in *IEEE Symposium on Visual Languages*. IEEE, pp. 336-343.
**DOI.** 10.1109/VL.1996.545307.
**Annotation.** "Overview first, zoom and filter, details on demand." Matches FORMA's dashboard drill-down exactly.

---

## 9.5  Emotional + hedonic design

### hassenzahl2006hedonic
**Harvard.** Hassenzahl, M. (2006) 'Hedonic, emotional, and experiential perspectives on product quality', in Ghaoui, C. (ed.) *Encyclopedia of Human Computer Interaction*. Hershey, PA: Idea Group, pp. 266-272.
**DOI.** 10.4018/978-1-59140-562-7.ch042.
**Annotation.** Foundational HCI source on hedonic (beauty, identity, stimulation) vs pragmatic product qualities. Justifies why FORMA invests in editorial/cinematic aesthetics beyond bare utility.

### norman2004emotional
**Harvard.** Norman, D.A. (2004) *Emotional Design: Why We Love (or Hate) Everyday Things*. New York: Basic Books.
**ISBN.** 9780465051366.
**Annotation.** Norman's visceral/behavioural/reflective model. Frames FORMA's cinematic dark aesthetic as a visceral-level investment that increases tolerance of minor usability imperfections.
**Suggested quote.** "Attractive things work better."

### brooke1996sus — SUS usability scale
**Harvard.** Brooke, J. (1996) 'SUS: A quick and dirty usability scale', in Jordan, P.W. *et al.* (eds.) *Usability Evaluation in Industry*. London: Taylor & Francis, pp. 189-194.
**Annotation.** Standard 10-item usability instrument. Recommended for FORMA's UI-evaluation chapter.

---

**Gaps in this section** — No peer-reviewed source verified for "hero video in product marketing" or "industrial/editorial aesthetic in fitness products"; the typography + emotional-design sources above support the broader argument. Butt 2018 / Li 2018 (perceived page-load performance) were unverified and replaced with Card 1991 + Nielsen 1993.
