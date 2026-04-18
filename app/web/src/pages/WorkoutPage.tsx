import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useParams, Navigate, useSearchParams } from "react-router-dom";
import { io, Socket } from "socket.io-client";
import { FlipHorizontal, Square, X } from "lucide-react";
import type { PoseLandmarker } from "@mediapipe/tasks-vision";
import { exerciseBySlug, isExerciseSlug } from "../types/exercise";
import type { PhoneOrientation } from "../types/exercise";
import { drawSkeleton, flattenLandmarks, getPoseLandmarker } from "../lib/pose";
import { useOrientation } from "../hooks/useOrientation";
import { usePhone } from "../hooks/usePhone";
import { OrientationGate } from "../components/session/OrientationGate";

type Phase =
  | "idle"
  | "requesting-camera"
  | "loading-pose"
  | "camera-setup"
  | "connecting"
  | "running"
  | "error";

type ResultPayload = {
  rep_count: number;
  form_score: number;
  is_active: boolean;
  is_correct: boolean | null;
  fps: number;
  details: string[];
  joint_feedback: Record<string, string>;
  session_state?: "active" | "resting" | "setup" | "idle";
  set_count?: number;
  reps_in_set?: number;
  last_set_reps?: number;
  // Plank only
  hold_duration?: number;
  current_set_hold?: number;
  total_hold_duration?: number;
  last_set_hold?: number;
};

type OverlayState =
  | "hidden"
  | "position"
  | "resting"
  | "out_of_frame"
  | "form_locked";

const FORM_LOCKED_DURATION_MS = 2500;

const TARGET_SEND_FPS = 15;
const MIN_SEND_INTERVAL_MS = 1000 / TARGET_SEND_FPS;
// Only fire client-side "rest" after prolonged stillness.
// A breather between reps (~2–4s) should NOT count as rest.
const CLIENT_REST_MS = 7000;

// Body-in-frame check: at least one side (shoulder + hip + knee) must be
// visible. Debounced over MIN_MISSING_FRAMES so a single blip doesn't
// flicker the overlay.
const VISIBILITY_FLOOR = 0.4;
const MIN_MISSING_FRAMES = 12;
const LEFT_CHAIN = [11, 23, 25];  // shoulder, hip, knee
const RIGHT_CHAIN = [12, 24, 26];

const RING_RADIUS = 48;
const RING_CIRC = 2 * Math.PI * RING_RADIUS;

function formatElapsed(ms: number): string {
  const total = Math.max(0, Math.floor(ms / 1000));
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
}

function formatDuration(seconds: number): string {
  const total = Math.max(0, Math.floor(seconds));
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${s.toString().padStart(2, "0")}`;
}

function scoreColor(score: number): string {
  if (score >= 75) return "#AEE710";
  if (score >= 50) return "#FBBF24";
  return "#F87171";
}

export function WorkoutPage() {
  const { exercise: exerciseParam } = useParams<{ exercise: string }>();
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();

  if (!isExerciseSlug(exerciseParam)) {
    return <Navigate to="/" replace />;
  }

  const exercise = exerciseBySlug(exerciseParam);
  const isPlank = exercise.slug === "plank";

  // Weight for weight-based exercises (deadlift etc.) — read from ?weight=.
  // Only used for display + start_session payload; the detector never sees it.
  const weightKg = useMemo(() => {
    if (!exercise.isWeighted) return null;
    const raw = searchParams.get("weight");
    if (!raw) return null;
    const n = parseFloat(raw);
    return Number.isFinite(n) && n > 0 ? n : null;
  }, [exercise.isWeighted, searchParams]);

  const [phase, setPhase] = useState<Phase>("idle");
  const [error, setError] = useState<string | null>(null);
  const [mirrored, setMirrored] = useState<boolean>(exercise.slug !== "pushup");
  const [saving, setSaving] = useState(false);

  // ── Phone orientation (Part B) ────────────────────────────────────────
  // Current phone orientation tracked via matchMedia.
  const orientation = useOrientation();
  const isPhone = usePhone();
  // User's chosen target orientation — starts at the exercise default,
  // flips when they pick the other option in the camera-setup toggle.
  const [desiredOrientation, setDesiredOrientation] = useState<PhoneOrientation>(
    exercise.preferredOrientation,
  );
  // If the user decides to proceed despite a mismatch, we stop showing
  // the rotate-phone gate for this session.
  const [orientationOverridden, setOrientationOverridden] = useState(false);
  const orientationMismatch =
    isPhone && orientation !== desiredOrientation && !orientationOverridden;
  const landscapePhone = isPhone && orientation === "landscape";

  // DOM refs
  const videoRef = useRef<HTMLVideoElement>(null);
  const skeletonCanvasRef = useRef<HTMLCanvasElement>(null);
  const timerRef = useRef<HTMLSpanElement>(null);
  const scoreValueRef = useRef<HTMLDivElement>(null);
  const scoreRingFillRef = useRef<SVGCircleElement>(null);
  const scoreValueRefMobile = useRef<HTMLDivElement>(null);
  const scoreRingFillRefMobile = useRef<SVGCircleElement>(null);
  const setsDoneRef = useRef<HTMLDivElement>(null);
  const currentSetRef = useRef<HTMLDivElement>(null);
  const currentSetLabelRef = useRef<HTMLDivElement>(null);
  const totalRepsRef = useRef<HTMLDivElement>(null);
  const feedbackRef = useRef<HTMLDivElement>(null);
  const fpsRef = useRef<HTMLDivElement>(null);
  const waitingOverlayRef = useRef<HTMLDivElement>(null);
  const overlayLabelRef = useRef<HTMLDivElement>(null);
  const overlaySubtextRef = useRef<HTMLDivElement>(null);

  // State refs
  const socketRef = useRef<Socket | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const poseRef = useRef<PoseLandmarker | null>(null);
  const rafRef = useRef<number | null>(null);
  const lastPoseTimestampRef = useRef<number>(-1);
  const lastSentRef = useRef(0);
  const endedRef = useRef(false);
  const sessionStartRef = useRef<number>(0);
  const lastActiveRef = useRef<number>(0);
  const wasActiveEverRef = useRef<boolean>(false);
  const jointFeedbackRef = useRef<Record<string, string>>({});
  const lastOverlayStateRef = useRef<OverlayState>("position");
  const prevSessionStateRef = useRef<string | undefined>(undefined);
  const formLockedUntilRef = useRef<number>(0);
  // Out-of-frame tracking: count consecutive frames where neither side's
  // shoulder+hip+knee chain has acceptable visibility.
  const missingFramesRef = useRef<number>(0);
  const outOfFrameRef = useRef<boolean>(false);
  // Camera-setup gate: the async begin() flow suspends on this until the
  // user clicks "I'm Ready". See handleReady() below.
  const userReadyRef = useRef<(() => void) | null>(null);

  // Session timer — rAF-driven, writes to DOM directly.
  useEffect(() => {
    if (phase !== "running") return;
    sessionStartRef.current = Date.now();
    let id = 0;
    let lastText = "";
    const tick = () => {
      const el = timerRef.current;
      if (el) {
        const text = formatElapsed(Date.now() - sessionStartRef.current);
        if (text !== lastText) {
          el.textContent = text;
          lastText = text;
        }
      }
      id = window.setTimeout(tick, 250);
    };
    tick();
    return () => window.clearTimeout(id);
  }, [phase]);

  const cleanup = useCallback(() => {
    if (rafRef.current != null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    const socket = socketRef.current;
    if (socket) {
      if (!endedRef.current) {
        try {
          socket.emit("end_session");
        } catch {
          // ignore
        }
        endedRef.current = true;
      }
      socket.disconnect();
      socketRef.current = null;
    }
    const stream = streamRef.current;
    if (stream) {
      stream.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
  }, []);

  useEffect(() => {
    let cancelled = false;

    async function begin() {
      // 1. Request camera
      setPhase("requesting-camera");
      let stream: MediaStream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: "user",
            width: { ideal: 1280 },
            height: { ideal: 720 },
            frameRate: { ideal: 30, max: 30 },
          },
          audio: false,
        });
      } catch {
        if (cancelled) return;
        setError("Camera access denied. Allow camera permissions and reload.");
        setPhase("error");
        return;
      }
      if (cancelled) {
        stream.getTracks().forEach((t) => t.stop());
        return;
      }

      streamRef.current = stream;
      const video = videoRef.current;
      if (!video) return;
      video.srcObject = stream;
      try {
        await video.play();
      } catch {
        // muted + playsInline should cover it
      }

      // Size skeleton canvas to match video natural dimensions
      const vw = video.videoWidth || 1280;
      const vh = video.videoHeight || 720;
      const skel = skeletonCanvasRef.current;
      if (skel) {
        skel.width = vw;
        skel.height = vh;
      }

      // 2. Load PoseLandmarker
      setPhase("loading-pose");
      try {
        poseRef.current = await getPoseLandmarker();
      } catch (e) {
        if (cancelled) return;
        setError("Failed to load pose model. Check your connection and reload.");
        setPhase("error");
        return;
      }
      if (cancelled) return;

      // 3. Camera-setup gate — the user confirms their framing before we
      // even connect to the server. Webcam is already streaming to the
      // hidden video element; the overlay shows the guidance text and an
      // "I'm Ready" button which calls userReadyRef.current() to release.
      setPhase("camera-setup");
      await new Promise<void>((resolve) => {
        userReadyRef.current = () => {
          userReadyRef.current = null;
          resolve();
        };
      });
      if (cancelled) return;

      // 4. Connect socket
      setPhase("connecting");
      const socket = io({
        transports: ["websocket", "polling"],
        reconnection: true,
        reconnectionAttempts: 5,
        timeout: 20000,
      });
      socketRef.current = socket;

      socket.on("connect", () => {
        socket.emit("start_session", {
          exercise: exercise.slug,
          classifier: "rule_based",
          weight_kg: weightKg,
        });
      });

      socket.on("session_started", () => {
        setPhase("running");
        rafRef.current = requestAnimationFrame(poseLoop);
      });

      socket.on("result", (payload: ResultPayload) => {
        const score = Math.round((payload.form_score ?? 0) * 100);
        const totalReps = payload.rep_count ?? 0;
        const active = !!payload.is_active;
        const sessionState = payload.session_state;
        const firstDetail =
          Array.isArray(payload.details) && payload.details.length > 0
            ? payload.details[0]
            : "";

        if (active) wasActiveEverRef.current = true;

        // Score ring — update both desktop + mobile rings (only one is
        // visible at a time via CSS, but both are in the DOM).
        const pct = Math.max(0, Math.min(100, score)) / 100;
        const dashOffset = String(RING_CIRC * (1 - pct));
        const stroke = scoreColor(score);
        const scoreText = active ? String(score) : "--";
        for (const ring of [scoreRingFillRef.current, scoreRingFillRefMobile.current]) {
          if (ring) {
            ring.style.strokeDashoffset = dashOffset;
            ring.style.stroke = stroke;
          }
        }
        for (const el of [scoreValueRef.current, scoreValueRefMobile.current]) {
          if (el) el.textContent = scoreText;
        }

        // Counters — prefer server-provided set fields, fall back to total reps.
        const serverSets = payload.set_count;
        const serverRepsInSet = payload.reps_in_set;
        const lastSetReps = payload.last_set_reps ?? 0;

        const displaySets = serverSets ?? 0;

        if (setsDoneRef.current)
          setsDoneRef.current.textContent = String(displaySets);
        if (currentSetLabelRef.current)
          currentSetLabelRef.current.textContent = `Set ${displaySets + 1} · ${
            isPlank ? "Hold" : "Reps"
          }`;

        if (isPlank) {
          // Plank: display time held, not reps.
          const currentHold = payload.current_set_hold ?? 0;
          const totalHold = payload.total_hold_duration ?? 0;
          if (currentSetRef.current)
            currentSetRef.current.textContent = formatDuration(currentHold);
          if (totalRepsRef.current)
            totalRepsRef.current.textContent = formatDuration(totalHold);
        } else {
          const displayCurrentSet =
            serverRepsInSet != null ? serverRepsInSet : totalReps;
          if (currentSetRef.current)
            currentSetRef.current.textContent = String(displayCurrentSet);
          if (totalRepsRef.current)
            totalRepsRef.current.textContent = String(totalReps);
        }

        const now = Date.now();
        if (active) {
          lastActiveRef.current = now;
        }

        // Detect session_state → active transition (start of every set / first lock).
        // Fires FORM LOCKED briefly so the user knows they can start repping.
        if (
          sessionState === "active" &&
          prevSessionStateRef.current !== "active"
        ) {
          formLockedUntilRef.current = now + FORM_LOCKED_DURATION_MS;
        }
        if (sessionState !== undefined) {
          prevSessionStateRef.current = sessionState;
        }

        const outOfFrameHint =
          !active &&
          /full body|Ensure|Step into|visible/i.test(firstDetail);

        let nextState: OverlayState;
        let label = "";
        let subtext = "";

        // STEP INTO FRAME — highest priority. If the client-side pose loop
        // can't see the user's body for MIN_MISSING_FRAMES consecutive
        // frames, surface this before anything else. Reps can't count
        // when joints aren't visible.
        if (outOfFrameRef.current) {
          nextState = "out_of_frame";
          label = "STEP INTO FRAME";
          subtext = "Move back so your full body is visible.";
        }
        // FORM LOCKED flash — next priority when armed.
        else if (now < formLockedUntilRef.current) {
          nextState = "form_locked";
          label = "FORM LOCKED";
          subtext = isPlank ? "Hold the position." : "Start your reps.";
        }
        // Push-up (and any exercise the server emits session_state for):
        // trust the server exclusively, ignore transient is_active flutter.
        else if (sessionState !== undefined) {
          if (sessionState === "resting") {
            nextState = "resting";
            label = "RESTING";
            if (isPlank) {
              const lastHold = payload.last_set_hold ?? 0;
              subtext =
                lastHold > 0
                  ? `${formatDuration(lastHold)} held in attempt ${Math.max(
                      displaySets,
                      1,
                    )}. Resume when ready.`
                  : "Resume when ready.";
            } else {
              subtext =
                lastSetReps > 0
                  ? `${lastSetReps} reps in set ${Math.max(displaySets, 1)}. Resume when ready.`
                  : "Resume when ready.";
            }
          } else if (sessionState === "setup" || sessionState === "idle") {
            nextState = "position";
            label = "GET INTO POSITION";
            subtext = firstDetail || "Line up in front of the camera.";
          } else {
            // active
            nextState = "hidden";
          }
        }
        // Exercises without server session_state — client heuristic.
        // Critical: brief pauses between reps must NOT trigger RESTING.
        else if (outOfFrameHint) {
          nextState = "out_of_frame";
          label = "STEP INTO FRAME";
          subtext = "Make sure your full body is visible.";
        } else if (
          !active &&
          wasActiveEverRef.current &&
          totalReps > 0 &&
          lastActiveRef.current > 0 &&
          now - lastActiveRef.current > CLIENT_REST_MS
        ) {
          nextState = "resting";
          label = "RESTING";
          subtext = `${totalReps} total reps. Resume when ready.`;
        } else if (!active && !wasActiveEverRef.current) {
          nextState = "position";
          label = "GET INTO POSITION";
          subtext = firstDetail || "Line up in front of the camera.";
        } else {
          // Active, or brief pause mid-set — keep overlay hidden.
          nextState = "hidden";
        }

        if (waitingOverlayRef.current) {
          waitingOverlayRef.current.style.opacity =
            nextState === "hidden" ? "0" : "1";
        }
        if (nextState !== lastOverlayStateRef.current) {
          lastOverlayStateRef.current = nextState;
          if (overlayLabelRef.current) {
            overlayLabelRef.current.textContent = label;
            overlayLabelRef.current.style.color =
              nextState === "form_locked"
                ? "#7ed957"
                : "rgba(255,255,255,0.9)";
          }
        } else if (overlayLabelRef.current && nextState !== "hidden") {
          if (overlayLabelRef.current.textContent !== label)
            overlayLabelRef.current.textContent = label;
        }
        if (overlaySubtextRef.current && nextState !== "hidden") {
          if (overlaySubtextRef.current.textContent !== subtext)
            overlaySubtextRef.current.textContent = subtext;
        }

        // Feedback line — hide when overlay is visible (avoid double text).
        if (feedbackRef.current) {
          const line = nextState === "hidden" ? firstDetail : "";
          if (feedbackRef.current.textContent !== line) {
            feedbackRef.current.textContent = line;
          }
        }

        if (fpsRef.current) {
          fpsRef.current.textContent = `${Math.round(payload.fps ?? 0)} fps`;
        }

        // Stash joint feedback for skeleton coloring
        jointFeedbackRef.current = payload.joint_feedback ?? {};
      });
    }

    // Per-frame pose loop — runs at display rate, emits landmarks throttled.
    function poseLoop() {
      rafRef.current = requestAnimationFrame(poseLoop);

      const video = videoRef.current;
      const canvas = skeletonCanvasRef.current;
      const pose = poseRef.current;
      const socket = socketRef.current;
      if (!video || !canvas || !pose || !socket) return;
      if (video.readyState < 2) return;

      const ts = performance.now();
      // MediaPipe VIDEO mode requires strictly increasing timestamps
      if (ts <= lastPoseTimestampRef.current) return;
      lastPoseTimestampRef.current = ts;

      let result;
      try {
        result = pose.detectForVideo(video, ts);
      } catch {
        return;
      }
      if (!result) return;

      // Draw skeleton overlay
      const ctx = canvas.getContext("2d");
      if (ctx) {
        drawSkeleton(
          ctx,
          result,
          canvas.width,
          canvas.height,
          jointFeedbackRef.current,
        );
      }

      // Body-in-frame check: require at least one side chain
      // (shoulder+hip+knee) visible at >= VISIBILITY_FLOOR. Debounced over
      // MIN_MISSING_FRAMES consecutive bad frames to avoid flicker.
      const lms = result.landmarks?.[0];
      if (lms) {
        const leftOk = LEFT_CHAIN.every(
          (i) => (lms[i]?.visibility ?? 0) >= VISIBILITY_FLOOR,
        );
        const rightOk = RIGHT_CHAIN.every(
          (i) => (lms[i]?.visibility ?? 0) >= VISIBILITY_FLOOR,
        );
        if (leftOk || rightOk) {
          missingFramesRef.current = 0;
          outOfFrameRef.current = false;
        } else {
          missingFramesRef.current += 1;
          if (missingFramesRef.current >= MIN_MISSING_FRAMES) {
            outOfFrameRef.current = true;
          }
        }
      } else {
        missingFramesRef.current += 1;
        if (missingFramesRef.current >= MIN_MISSING_FRAMES) {
          outOfFrameRef.current = true;
        }
      }

      // Throttled emit
      if (ts - lastSentRef.current >= MIN_SEND_INTERVAL_MS && socket.connected) {
        const flat = flattenLandmarks(result);
        if (flat) {
          lastSentRef.current = ts;
          socket.emit("landmarks", {
            landmarks: flat.landmarks,
            world_landmarks: flat.worldLandmarks,
            timestamp: Math.round(ts),
          });
        }
      }
    }

    begin();

    return () => {
      cancelled = true;
      cleanup();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [exercise.slug]);

  // ── Re-acquire camera on phone orientation change ────────────────────
  // The browser orients the video stream to match device orientation at
  // the moment getUserMedia() resolves — if the user rotates mid-session,
  // the stream stays in its original orientation and the preview looks
  // sideways. Re-requesting with matching width/height hints gets a fresh,
  // correctly-oriented stream (same as a native camera app).
  const lastAcquiredOrientationRef = useRef<PhoneOrientation | null>(null);
  useEffect(() => {
    if (!isPhone) return;
    // Don't re-acquire until the first camera grab has happened and the
    // user has actually changed orientation.
    if (streamRef.current == null) {
      lastAcquiredOrientationRef.current = orientation;
      return;
    }
    if (lastAcquiredOrientationRef.current === orientation) return;
    // Skip if we're not in a phase where the video is being shown.
    if (phase !== "running" && phase !== "camera-setup") return;

    let cancelled = false;
    const reacquire = async () => {
      const landscape = orientation === "landscape";
      try {
        const fresh = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: "user",
            width: { ideal: landscape ? 1280 : 720 },
            height: { ideal: landscape ? 720 : 1280 },
            frameRate: { ideal: 30, max: 30 },
          },
          audio: false,
        });
        if (cancelled) {
          fresh.getTracks().forEach((t) => t.stop());
          return;
        }
        const previous = streamRef.current;
        streamRef.current = fresh;
        const video = videoRef.current;
        if (video) {
          video.srcObject = fresh;
          try {
            await video.play();
          } catch {
            /* muted + playsInline */
          }
          // Wait a frame for videoWidth/Height to update, then resize canvas.
          requestAnimationFrame(() => {
            const vw = video.videoWidth || (landscape ? 1280 : 720);
            const vh = video.videoHeight || (landscape ? 720 : 1280);
            const skel = skeletonCanvasRef.current;
            if (skel) {
              skel.width = vw;
              skel.height = vh;
            }
          });
        }
        if (previous) previous.getTracks().forEach((t) => t.stop());
        lastAcquiredOrientationRef.current = orientation;
      } catch {
        /* keep the existing stream if the re-acquire fails */
      }
    };
    reacquire();
    return () => {
      cancelled = true;
    };
  }, [orientation, isPhone, phase]);

  const handleExit = useCallback(() => {
    const socket = socketRef.current;

    // If there's nothing live to flush, just clean up and go.
    if (!socket || !socket.connected || endedRef.current) {
      cleanup();
      navigate("/");
      return;
    }

    // Stop streaming landmarks immediately so the server's last known state
    // is what it scores against. We still need the socket alive to receive
    // the session_completed ack.
    if (rafRef.current != null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }

    setSaving(true);

    let done = false;
    const finish = () => {
      if (done) return;
      done = true;
      cleanup();
      navigate("/");
    };

    socket.once("session_completed", finish);
    // session_report fires just before session_completed and also indicates
    // the session was persisted — either one is a valid "we're saved" signal.
    socket.once("session_report", finish);

    try {
      socket.emit("end_session");
    } catch {
      // If emit throws, fall through to the timeout.
    }
    endedRef.current = true;

    // Hard ceiling — never leave the user staring at a spinner.
    window.setTimeout(finish, 2500);
  }, [cleanup, navigate]);

  // Cancel flow — explicitly DISCARDS the session. Sends discard_session
  // so the server tears down in-memory state and purges the on-disk capture
  // without saving anything to the sessions table. Used for the X button.
  const handleDiscard = useCallback(() => {
    const socket = socketRef.current;

    if (!socket || !socket.connected || endedRef.current) {
      cleanup();
      navigate("/");
      return;
    }

    if (rafRef.current != null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }

    let done = false;
    const finish = () => {
      if (done) return;
      done = true;
      cleanup();
      navigate("/");
    };
    socket.once("session_discarded", finish);
    try {
      socket.emit("discard_session");
    } catch {
      /* fall through to timeout */
    }
    endedRef.current = true;
    window.setTimeout(finish, 1200);
  }, [cleanup, navigate]);

  const handleReady = useCallback(() => {
    userReadyRef.current?.();
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") handleExit();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [handleExit]);

  const running = phase === "running";

  return (
    <div className="fixed inset-0 z-[400] bg-black text-white overflow-hidden font-[family-name:var(--font-body)]">
      {/* Live video + skeleton overlay (native 30fps).
          Phone landscape uses object-cover for a true full-bleed fit;
          portrait / desktop letterbox to preserve aspect.
          Orientation handling is the browser's job at getUserMedia time
          plus our re-acquire effect on rotation change — no CSS rotation
          transforms (they were making the body flip upside down on some
          device orientations). */}
      <video
        ref={videoRef}
        muted
        playsInline
        className={`absolute inset-0 h-full w-full bg-black ${
          landscapePhone ? "object-cover" : "object-contain"
        }`}
        style={{ transform: mirrored ? "scaleX(-1)" : "none" }}
      />
      <canvas
        ref={skeletonCanvasRef}
        className={`absolute inset-0 h-full w-full pointer-events-none ${
          landscapePhone ? "object-cover" : "object-contain"
        }`}
        style={{ transform: mirrored ? "scaleX(-1)" : "none" }}
      />

      {/* TOP-LEFT — exercise + timer */}
      <div className="absolute top-3 left-3 md:top-8 md:left-10 landscape:top-8 landscape:left-10 max-w-[58%] md:max-w-none landscape:max-w-none">
        <div className="text-[9px] md:text-[10px] uppercase tracking-[0.24em] text-[color:var(--color-gold-soft)]">
          Live Session
        </div>
        <div className="mt-1 font-[family-name:var(--font-display)] text-2xl md:text-5xl landscape:text-4xl tracking-[0.04em] leading-none">
          {exercise.name}
          {weightKg != null && (
            <span className="text-[color:var(--color-gold-soft)]">
              {" · "}
              {weightKg} KG
            </span>
          )}
        </div>
        <div className="mt-2 md:mt-3 inline-flex items-center gap-2 px-2.5 py-1 md:px-3 md:py-1.5 rounded-sm bg-black/70 border border-white/10">
          <span className="h-1.5 w-1.5 rounded-full bg-[color:var(--color-gold-soft)] animate-pulse" />
          <span
            ref={timerRef}
            className="font-mono text-xs md:text-sm tabular-nums tracking-[0.1em] text-[color:var(--color-gold-soft)]"
          >
            00:00
          </span>
        </div>
      </div>

      {/* TOP-RIGHT — score ring (desktop only) + X cancel + Mirror + Stop */}
      <div className="absolute top-3 right-3 md:top-8 md:right-10 landscape:top-8 landscape:right-10 flex items-start gap-3 md:gap-5">
        {/* Score ring — hidden on mobile portrait, visible on desktop + landscape */}
        <div className="relative h-[110px] w-[110px] rounded-full bg-black/70 border border-white/10 p-2 hidden md:block landscape:block">
          <svg viewBox="0 0 120 120" className="h-full w-full -rotate-90">
            <circle
              cx="60"
              cy="60"
              r={RING_RADIUS}
              fill="none"
              stroke="rgba(255,255,255,0.12)"
              strokeWidth="5"
            />
            <circle
              ref={scoreRingFillRef}
              cx="60"
              cy="60"
              r={RING_RADIUS}
              fill="none"
              stroke="#AEE710"
              strokeWidth="5"
              strokeLinecap="round"
              strokeDasharray={RING_CIRC}
              strokeDashoffset={RING_CIRC}
              style={{
                transition:
                  "stroke-dashoffset 220ms ease-out, stroke 300ms ease-out",
              }}
            />
          </svg>
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <div
              ref={scoreValueRef}
              className="font-[family-name:var(--font-display)] text-3xl leading-none tabular-nums"
            >
              --
            </div>
            <div className="mt-0.5 text-[9px] uppercase tracking-[0.2em] text-white/60">
              Form
            </div>
          </div>
        </div>

        <div className="flex flex-col items-end gap-1.5 md:gap-2">
          {/* Cancel (X) — discards without saving. Circular icon button,
              sits in the top-right corner of the button stack so it reads
              as the escape hatch. */}
          <button
            type="button"
            onClick={handleDiscard}
            aria-label="Cancel session without saving"
            title="Cancel (don't save)"
            className="h-10 w-10 flex items-center justify-center rounded-full bg-black/70 border border-white/20 text-white/75 hover:text-white hover:bg-black/90 hover:border-white/45 transition-colors"
          >
            <X size={18} />
          </button>
          <button
            type="button"
            onClick={() => setMirrored((m) => !m)}
            className="inline-flex items-center justify-center gap-1.5 md:gap-2 px-2.5 py-2 md:px-4 md:py-2.5 min-h-[44px] min-w-[44px] bg-black/70 border border-white/20 text-[10px] uppercase tracking-[0.18em] hover:bg-white/10 hover:border-white/40 transition-colors"
            aria-label={mirrored ? "Disable mirror" : "Enable mirror"}
            title={mirrored ? "Mirror: on" : "Mirror: off"}
          >
            <FlipHorizontal size={14} />
            <span className="hidden sm:inline">
              {mirrored ? "Mirror" : "Flip"}
            </span>
          </button>
          {/* Stop & Save — prominent gold-filled button. Primary action. */}
          <button
            type="button"
            onClick={handleExit}
            className="inline-flex items-center justify-center gap-2 px-4 py-3 md:px-5 md:py-3.5 min-h-[48px] bg-[color:var(--color-gold-soft)] text-[#0A0A0A] text-xs md:text-sm uppercase tracking-[0.16em] font-medium shadow-lg shadow-[color:var(--color-gold-soft)]/20 hover:bg-[color:var(--color-gold)] transition-colors"
            aria-label="Stop and save session"
            title="Stop and save"
          >
            <Square size={16} fill="currentColor" />
            <span>Stop &amp; Save</span>
          </button>
        </div>
      </div>

      {/* COUNTERS — horizontal strip on mobile portrait, vertical on desktop/landscape */}
      <div className="absolute left-3 right-3 top-[88px] grid grid-cols-3 gap-2 md:left-10 md:right-auto md:top-1/2 md:-translate-y-1/2 md:w-[170px] md:block md:space-y-3 landscape:left-10 landscape:right-auto landscape:top-1/2 landscape:-translate-y-1/2 landscape:w-[170px] landscape:block landscape:space-y-3">
        <div className="px-2.5 py-2 md:px-4 md:py-3 rounded-sm bg-black/70 border border-white/10">
          <div className="text-[8px] md:text-[9px] uppercase tracking-[0.22em] text-white/55">
            Sets Done
          </div>
          <div
            ref={setsDoneRef}
            className="mt-0.5 md:mt-1 font-[family-name:var(--font-display)] leading-none tabular-nums text-[1.5rem] md:text-[2rem] text-white/90"
          >
            0
          </div>
        </div>
        <div className="px-2.5 py-2 md:px-4 md:py-3 rounded-sm bg-black/70 border border-[color:var(--color-gold-soft)]/40">
          <div
            ref={currentSetLabelRef}
            className="text-[8px] md:text-[9px] uppercase tracking-[0.22em] text-white/55 truncate"
          >
            Set 1 · {isPlank ? "Hold" : "Reps"}
          </div>
          <div
            ref={currentSetRef}
            className="mt-0.5 md:mt-1 font-[family-name:var(--font-display)] leading-none tabular-nums text-[2.25rem] md:text-[3.25rem] text-[color:var(--color-gold-soft)]"
          >
            0
          </div>
        </div>
        <div className="px-2.5 py-2 md:px-4 md:py-3 rounded-sm bg-black/70 border border-white/10">
          <div className="text-[8px] md:text-[9px] uppercase tracking-[0.22em] text-white/55 truncate">
            {isPlank ? "Total Time" : "Total Reps"}
          </div>
          <div
            ref={totalRepsRef}
            className="mt-0.5 md:mt-1 font-[family-name:var(--font-display)] leading-none tabular-nums text-[1.5rem] md:text-[2rem] text-white/90"
          >
            0
          </div>
        </div>
      </div>

      {/* MOBILE SCORE RING — bottom-left, only shown on mobile portrait */}
      <div className="absolute bottom-20 left-3 h-[84px] w-[84px] rounded-full bg-black/70 border border-white/10 p-1.5 md:hidden landscape:hidden">
        <svg viewBox="0 0 120 120" className="h-full w-full -rotate-90">
          <circle
            cx="60"
            cy="60"
            r={RING_RADIUS}
            fill="none"
            stroke="rgba(255,255,255,0.12)"
            strokeWidth="6"
          />
          <circle
            ref={scoreRingFillRefMobile}
            cx="60"
            cy="60"
            r={RING_RADIUS}
            fill="none"
            stroke="#AEE710"
            strokeWidth="6"
            strokeLinecap="round"
            strokeDasharray={RING_CIRC}
            strokeDashoffset={RING_CIRC}
            style={{
              transition:
                "stroke-dashoffset 220ms ease-out, stroke 300ms ease-out",
            }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <div
            ref={scoreValueRefMobile}
            className="font-[family-name:var(--font-display)] text-2xl leading-none tabular-nums"
          >
            --
          </div>
          <div className="text-[8px] uppercase tracking-[0.2em] text-white/60">
            Form
          </div>
        </div>
      </div>

      {/* State overlay (get into position / resting / out of frame) */}
      {running && (
        <div
          ref={waitingOverlayRef}
          className="pointer-events-none absolute inset-0 flex items-center justify-center transition-opacity duration-300"
          style={{ opacity: 1 }}
        >
          <div className="text-center px-4 md:px-8 max-w-2xl">
            <div
              ref={overlayLabelRef}
              className="font-[family-name:var(--font-display)] tracking-[0.06em] text-white/90 leading-tight"
              style={{ fontSize: "clamp(1.75rem, 7vw, 3.75rem)" }}
            >
              GET INTO POSITION
            </div>
            <div
              ref={overlaySubtextRef}
              className="mt-3 md:mt-4 font-[family-name:var(--font-serif)] italic text-[color:var(--color-gold-soft)]/80"
              style={{ fontSize: "clamp(0.95rem, 2.4vw, 1.25rem)" }}
            />
          </div>
        </div>
      )}

      {/* Bottom feedback — on mobile it sits to the right of the score ring
          (ring is bottom-20 left-3), so indent the feedback past it */}
      <div className="absolute bottom-8 right-3 left-[108px] md:inset-x-0 md:px-10 md:left-0">
        <div
          ref={feedbackRef}
          className="min-h-[2.25rem] text-left md:text-center font-[family-name:var(--font-serif)] italic text-base md:text-2xl text-[color:var(--color-gold-soft)]"
        />
      </div>

      <div
        ref={fpsRef}
        className="absolute bottom-1.5 right-3 md:bottom-4 md:right-6 text-[9px] md:text-[10px] uppercase tracking-[0.18em] text-white/40 tabular-nums"
      >
        0 fps
      </div>

      {/* Rotate-phone gate — shown on phones when the current orientation
          doesn't match the user's chosen target. Sits on top of the
          camera-setup overlay (higher z-index) so it blocks the "I'm Ready"
          button until the user either rotates or opts to override. */}
      {phase === "camera-setup" && orientationMismatch && (
        <OrientationGate
          exercise={{ ...exercise, preferredOrientation: desiredOrientation }}
          current={orientation}
          onOverride={() => setOrientationOverridden(true)}
        />
      )}

      {/* Camera-setup overlay — video is visible underneath so the user
          can verify their framing while reading the guidance. */}
      {phase === "camera-setup" && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/55 backdrop-blur-[2px] p-6">
          <div className="w-full max-w-xl bg-black/80 border border-[color:var(--color-gold-soft)]/40 p-8 md:p-10">
            <div className="text-[10px] uppercase tracking-[0.24em] text-[color:var(--color-gold-soft)] mb-3">
              Camera Setup · {exercise.cameraGuidance.view.replace("_", " ")} view
            </div>
            <h2 className="font-[family-name:var(--font-display)] text-3xl md:text-4xl text-white tracking-[0.04em] leading-tight">
              {exercise.cameraGuidance.headline}
            </h2>
            <p className="mt-4 font-[family-name:var(--font-serif)] italic text-lg text-white/80 leading-relaxed">
              {exercise.cameraGuidance.detail}
            </p>

            {/* Phone-only orientation toggle. Recommended pre-selected;
                flipping it shows an amber note. Desktop hides this row. */}
            {isPhone && (
              <div className="mt-6">
                <div className="text-[10px] uppercase tracking-[0.18em] text-white/55 mb-2">
                  Phone orientation
                </div>
                <div
                  role="radiogroup"
                  aria-label="Phone orientation"
                  className="grid grid-cols-2 gap-0 border border-white/15 overflow-hidden rounded-sm"
                >
                  {(["portrait", "landscape"] as PhoneOrientation[]).map((opt) => {
                    const selected = desiredOrientation === opt;
                    const recommended = exercise.preferredOrientation === opt;
                    return (
                      <button
                        key={opt}
                        type="button"
                        role="radio"
                        aria-checked={selected}
                        onClick={() => {
                          setDesiredOrientation(opt);
                          setOrientationOverridden(false);
                        }}
                        className={`py-3 text-[11px] uppercase tracking-[0.18em] transition-colors ${
                          selected
                            ? "bg-[color:var(--color-gold-soft)] text-[#0A0A0A]"
                            : "bg-transparent text-white/65 hover:bg-white/5"
                        }`}
                      >
                        {opt}
                        {recommended && (
                          <span className="ml-1.5 text-[9px] opacity-70">(rec.)</span>
                        )}
                      </button>
                    );
                  })}
                </div>
                {desiredOrientation !== exercise.preferredOrientation && (
                  <div className="mt-3 border-l-2 border-[color:var(--color-warn)] bg-[color:var(--color-warn)]/10 px-3 py-2 text-[11px] text-white/80">
                    {exercise.preferredOrientation === "landscape"
                      ? "Landscape is recommended — in portrait your body may not fit the frame."
                      : "Portrait is recommended — landscape may cut off your standing height."}
                  </div>
                )}
              </div>
            )}

            {/* Redesign Phase 4 — weighted exercises want a working
                weight recorded so strength goals and weighted plan-day
                completion can attribute the session. Non-blocking
                warning; user can still proceed freestyle. */}
            {exercise.isWeighted && weightKg == null && (
              <div className="mt-6 border-l-2 border-[color:var(--color-warn)] bg-[color:var(--color-warn)]/10 px-4 py-3 text-[12px] text-white/85">
                <div className="uppercase tracking-[0.18em] text-[10px] text-[color:var(--color-warn)] mb-1">
                  No weight recorded
                </div>
                Enter your working weight on the /exercises card before
                starting so this session can count toward a weighted plan
                day or strength goal. You can still continue freestyle.
              </div>
            )}
            <div className="mt-8 flex items-center justify-between gap-4">
              <button
                type="button"
                onClick={handleExit}
                className="inline-flex items-center gap-2 px-5 py-2.5 border border-white/30 text-[10px] uppercase tracking-[0.18em] text-white/70 hover:text-white hover:border-white/60 transition-colors"
              >
                ← Cancel
              </button>
              <button
                type="button"
                onClick={handleReady}
                className="inline-flex items-center gap-2 px-8 py-3 bg-[color:var(--color-gold-soft)] text-[#0A0A0A] text-xs uppercase tracking-[0.14em] font-medium hover:bg-[color:var(--color-gold)] transition-colors"
              >
                I'm Ready →
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Saving overlay — shown after handleExit while we wait for the
          session_completed ack so the user knows the session is being
          persisted and doesn't bounce to `/` too early. */}
      {saving && (
        <div className="absolute inset-0 z-[500] flex items-center justify-center bg-black/85 backdrop-blur-sm">
          <div className="text-center max-w-md px-6">
            <div className="text-xs uppercase tracking-[0.24em] text-[color:var(--color-gold-soft)] mb-4">
              Saving session
            </div>
            <p className="font-[family-name:var(--font-serif)] italic text-xl text-white/80">
              Writing reps, sets and form scores…
            </p>
          </div>
        </div>
      )}

      {/* Loading / error / connecting overlay (full block) */}
      {!running && phase !== "camera-setup" && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/75">
          <div className="text-center max-w-md px-6">
            {phase === "error" ? (
              <>
                <div className="text-xs uppercase tracking-[0.24em] text-[color:var(--color-orange)] mb-4">
                  Error
                </div>
                <p className="font-[family-name:var(--font-serif)] italic text-xl text-white/80">
                  {error}
                </p>
                <button
                  type="button"
                  onClick={handleExit}
                  className="mt-8 inline-flex items-center gap-2 px-5 py-2.5 border border-white/30 text-xs uppercase tracking-[0.14em] hover:bg-white/10 transition-colors"
                >
                  ← Back
                </button>
              </>
            ) : (
              <>
                <div className="text-xs uppercase tracking-[0.24em] text-[color:var(--color-gold-soft)] mb-4">
                  {phase === "requesting-camera"
                    ? "Camera"
                    : phase === "loading-pose"
                      ? "Loading Pose Model"
                      : "Connecting"}
                </div>
                <p className="font-[family-name:var(--font-serif)] italic text-xl text-white/70">
                  {phase === "requesting-camera"
                    ? "Allow camera access to begin"
                    : phase === "loading-pose"
                      ? "Downloading MediaPipe (first time only)…"
                      : "Opening session…"}
                </p>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
