# ExerVision — Planned Features

Scope: features discussed for the next iteration of ExerVision. Built on the existing MediaPipe + dedicated detector foundation. Target deployment: Railway free tier, so every feature must respect memory/CPU limits and avoid constant LLM calls.

---

## 1. Real-Time Voice Coaching (no LLM in the hot path)

**Why not an LLM here:** Round-tripping joint angles to a cloud LLM takes 500 ms – 2 s. The rep is over before the app finishes speaking. A real trainer corrects you in ~100 ms. Latency kills the feature.

**Approach:** Hard-coded coaching commands, one JSON file per exercise, multiple phrasing variants per error so it never sounds robotic.

**Data shape:**
```json
{
  "push_up": {
    "hip_sag": [
      "Keep your hips up",
      "Tighten your core, hips level",
      "Don't let those hips drop",
      "Core tight, body straight"
    ],
    "head_down": [
      "Head straight, eyes down",
      "Keep your neck neutral",
      "Don't crane your head back"
    ]
  }
}
```

**Rules:**
- 3–5 variants per error per exercise (rotate so it feels natural).
- Fire only on *transition* (error just became true) — not every frame.
- Debounce: same error cannot re-trigger for N seconds.
- Text-to-speech via Web Speech API (free, client-side, no backend cost).
- Works offline.

**Coverage target:** all 10 exercises, all form errors currently flagged by the dedicated detectors in `src/classification/*_detector.py`.

---

## 2. AI Chatbot — User History & Analytics

**Role:** This is where the "AI" in the project actually lives. Not real-time coaching — a conversational layer that knows the user.

**What it knows:**
- Every session the user has recorded (exercise, rep count, form scores, timestamps, errors flagged).
- Aggregates: reps per day/week/month, average form score per exercise, trend direction.
- Active plans/milestones and current progress.

**What it can answer:**
- "How many push-ups have I done in the last 30 days?"
- "When did my squat form peak?"
- "What exercise do I consistently fail on?"
- "Am I improving week over week?"

**Architecture:**
- SQLite (already in repo as `exervision.db`) stores session/rep/score history.
- When user asks a question, backend queries DB for the *relevant slice* only (e.g. last 30 days of push-ups).
- Slice + question sent to LLM. Never send the entire history.
- LLM returns natural-language answer.

**Cost control — prompt caching:**
- Use LLM prompt caching: system prompt + user profile summary gets cached, subsequent queries within the 5-minute window cost ~10% of uncached tokens.
- Maintain a pre-computed `user_summary` row in DB (updated after each session) so the chatbot doesn't have to re-aggregate on every query.
- Hard ceiling: free-tier users get N chatbot messages per day.

---

## 3. Goal-Driven Workout Plans

**User flow:** "I want to do 1,000 push-ups and 200 squats in the next 10 days."

**What the chatbot does:**
1. Reads user's current fitness level from DB (avg reps per session, last 30 days).
2. Sends goal + fitness context to LLM.
3. LLM returns a day-by-day distribution with rest days.
4. Plan is saved to DB as a set of `milestones` (exercise, target reps, due date, status).

**Example output:**
```
Day 1: 150 push-ups, 30 squats
Day 2: 120 push-ups, 25 squats
Day 3: Rest
Day 4: 180 push-ups, 35 squats
...
```

**User can:**
- Accept the plan → milestones are committed.
- Ask the chatbot to adjust ("make it easier", "skip weekends").
- Cancel the plan at any time.

---

## 4. Automatic Milestone Tracking

**How it works:**
- Every finished session updates the DB.
- A background job (or end-of-session hook) compares the day's completed reps vs. the active milestone's target.
- Milestone status: `complete` / `partial` / `missed`.

**Notifications — tone matters:**
- ✅ Auto-mark completed milestones. Silent success or small celebration.
- ✅ Soft nudge if behind: *"You've got 30 push-ups left for today — want to finish strong?"*
- ✅ One nudge per day max.
- ❌ No guilt-trip notifications ("You failed today's target"). Demotivates users and they uninstall.
- ✅ Visual progress bar on the dashboard always visible.

**Fallback:** if a user misses several days, the chatbot offers to rebalance the plan instead of marking it as failed.

---

## 5. Dashboard / Progress UI

**Pages needed:**
- **Home** — today's milestone, streak, quick-start workout.
- **History** — session list with form scores, filterable by exercise.
- **Analytics** — charts (reps over time, form score trend, best/worst exercises).
- **Chatbot** — conversational interface to the AI layer.
- **Plans & Milestones** — current plan progress, past plans.

**Design direction:** industrial luxury fitness (dark charcoal #0D0D0D, gold #D4A574, burnt orange #E8572A) — already established in the project spec. Bebas Neue + Outfit + Cormorant Garamond. Every page must be visually rich, not sterile.

---

## 6. Admin / Debug Panel (testing infrastructure)

**Purpose:** every feature above depends on the detector being accurate. If MediaPipe mis-measures a deadlift back angle, the coaching is wrong, the form score is wrong, the chatbot analytics are wrong, everything downstream is poisoned. So we need a way to *see* what the detector saw.

**Developer Mode toggle in the existing app:**
- Skeleton + angle numbers overlaid on the live video.
- Per-frame log: timestamps, all joint angles, MediaPipe confidence per landmark, error flags raised by the detector.
- Session recording saved to `data/sessions/` (already partially implemented).

**Admin playback view:**
- Replay any saved session with skeleton + angles synced to the video.
- Timeline chart of angles vs. error flags.
- Annotate frames where the detector disagreed with reality: *"Frame 47: detector said hip angle 155°, actually ~165°"*, *"Frames 20–30: spine midpoint lost, angles garbage"*.

**Iteration loop:**
1. Record a user doing an exercise.
2. Replay in admin panel.
3. Spot detection failures.
4. Adjust thresholds in `src/classification/<exercise>_detector.py` or smoothing params.
5. Re-record, compare.

**Per-exercise reliability report:** after testing, document for each of the 10 exercises which angles MediaPipe can measure reliably and which are unreliable. Deadlift spine is the known hard case.

**Graceful degradation:** if MediaPipe can't see a key joint (visibility below threshold), the app must *not* guess. It should say *"I can't see your form clearly — adjust the camera angle"* instead of scoring incorrectly. Silence is better than wrong feedback.

---

## 7. Railway Free-Tier Constraints

Every decision above was made with the free tier in mind:

| Constraint | Mitigation |
|---|---|
| Limited RAM | Detectors stay lightweight; no on-device LLM. |
| Limited CPU | Real-time inference runs in the browser via MediaPipe JS, not on the server. |
| Egress / compute cost | LLM calls only when user explicitly asks the chatbot or creates a plan. No LLM in real-time loop. |
| Token budget | Server-side prompt caching + pre-computed user summary + query-only-relevant-slices. |
| Storage | SQLite file, rotate old session recordings, store keyframes not full video. |

---

## Build Order (priority, not timeline)

1. **Detector reliability pass** — admin/debug panel + per-exercise testing. Nothing else matters if form detection is wrong.
2. **Real-time voice coaching** — JSON command library + TTS hookup.
3. **Session logging to SQLite** — ensure every session writes reps, scores, errors, timestamps.
4. **Chatbot MVP** — read-only Q&A over user history, with prompt caching.
5. **Plans & milestones** — LLM plan generation + auto-tracking + soft nudges.
6. **Dashboard polish** — analytics, charts, industrial-luxury UI pass.

---

## Non-Goals (explicitly out of scope)

- On-device LLMs (battery + model size + still laggy).
- Sending video frames or images to an LLM (expensive, slow, not needed).
- Real-time LLM coaching during reps (latency kills it).
- Harsh / punitive notifications.
- Feature flags / backwards-compatibility shims for features not yet built.
