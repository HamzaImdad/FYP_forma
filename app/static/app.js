/**
 * ExerVision SPA client — Industrial Luxury v5.
 *
 * Pages: home, exercises, guide, session, report, about.
 * Communicates with Flask server via Socket.IO for real-time frame processing.
 *
 * Visual effects: AOS scroll reveals, subtle Vanta.js NET on hero.
 * No Vanilla-Tilt, no Zdog, no custom cursor.
 */

// ── Exercise Images Map ────────────────────────────────────────────────

const EXERCISE_IMAGES = {
    squat: '/static/images/ex_squat.jpg',
    lunge: '/static/images/ex_lunge.jpg',
    deadlift: '/static/images/ex_deadlift.jpg',
    bench_press: '/static/images/ex_bench_press.jpg',
    overhead_press: '/static/images/ex_overhead_press.jpg',
    pullup: '/static/images/ex_pullup.jpg',
    pushup: '/static/images/ex_pushup.jpg',
    plank: '/static/images/ex_plank.jpg',
    bicep_curl: '/static/images/ex_bicep_curl.jpg',
    tricep_dip: '/static/images/ex_tricep_dip.jpg',
};

// ── Page Navigation ─────────────────────────────────────────────────────

const pages = {
    home: document.getElementById("page-home"),
    exercises: document.getElementById("page-exercises"),
    guide: document.getElementById("page-guide"),
    session: document.getElementById("page-session"),
    report: document.getElementById("page-report"),
    about: document.getElementById("page-about"),
};

let currentPage = "home";

// Track Vanta effect for cleanup
let vantaEffect = null;

function showPage(name) {
    // Don't leave active session via nav
    if (currentPage === "session" && name !== "report") return;

    const leavingPage = currentPage;

    Object.values(pages).forEach((p) => p.classList.remove("active"));
    pages[name].classList.add("active");
    currentPage = name;
    window.scrollTo(0, 0);

    // Update nav active state
    document.querySelectorAll(".nav-link").forEach((link) => {
        link.classList.remove("active");
        if (link.dataset.nav === name) link.classList.add("active");
    });

    // Hide nav during session for full-screen video
    const nav = document.getElementById("main-nav");
    const footer = document.querySelector("footer");
    if (name === "session") {
        nav.style.display = "none";
        footer.style.display = "none";
    } else {
        nav.style.display = "";
        footer.style.display = "";
    }

    // Close mobile menu
    document.getElementById("nav-links").classList.remove("open");

    // --- Effect lifecycle management ---
    // Destroy Vanta when entering session (performance critical)
    if (name === "session") {
        destroyVanta();
    }

    // Vanta: only active on home page
    if (name === "home") {
        requestAnimationFrame(() => {
            initVanta();
        });
    } else {
        destroyVanta();
    }

    // Refresh AOS after page transition
    requestAnimationFrame(() => {
        if (typeof AOS !== "undefined") {
            AOS.refresh();
        }
    });
}

// Navigation click handlers (all [data-nav] elements)
document.addEventListener("click", (e) => {
    const navEl = e.target.closest("[data-nav]");
    if (!navEl) return;
    e.preventDefault();
    const target = navEl.dataset.nav;
    if (pages[target]) showPage(target);
});

// Mobile hamburger toggle
document.getElementById("nav-toggle").addEventListener("click", () => {
    document.getElementById("nav-links").classList.toggle("open");
});

// ── State ───────────────────────────────────────────────────────────────

let selectedExercise = "squat";
let selectedClassifier = "rule_based";
let exerciseList = [];

let socket = null;
let stream = null;
let sending = false;
let animFrameId = null;
let sessionTimerInterval = null;
let sessionStartTime = null;
let isPaused = false;
let pausedElapsed = 0;
let isMirrored = true;
let sessionEnding = false;

// Session recording
let mediaRecorder = null;
let recordedChunks = [];
let recordingBlob = null;
let recordingCanvas = null;
let recordingCtx = null;

const SEND_INTERVAL_MS = 83; // ~12 fps max
const SEND_TIMEOUT_MS = 3000;
let lastSendTime = 0;
let sendStartTime = 0;

const frameImg = new Image();

// Form score smoothing (running average over last 5 readings)
const SCORE_BUFFER_SIZE = 5;
let scoreBuffer = [];
let isUserActive = false;

// Score history for real-time mini chart
const MAX_SCORE_HISTORY = 60;
let scoreHistory = [];

// Track active frames for session stats
let activeFrameCount = 0;
let totalFrameCount = 0;

// Latency instrumentation
const latencyHistory = [];
const LATENCY_WINDOW = 100; // rolling window for p50/p95

// Browser-side MediaPipe (hybrid mode)
let poseLandmarker = null;
let hybridMode = false;
let lastJointFeedback = {};

// Push-up session HUD state (updated from server responses)
let hudState = {
    repCount: 0,
    setCount: 0,
    repsInSet: 0,
    formScore: 0,
    phase: "",
    progress: "",
    isActive: false,
    lastActiveTime: 0,
    wasActive: false,        // was active last frame
    setJustCompleted: false,  // show set-complete overlay
    completedSetReps: 0,      // reps in the set that just finished
    completedSetNum: 0,       // which set just finished
};
let skeletonCanvas = null;
let skeletonCtx = null;

// Skeleton connections matching src/utils/constants.py
const SKELETON_CONNECTIONS = [
    [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
    [11, 23], [12, 24], [23, 24],
    [23, 25], [25, 27], [24, 26], [26, 28],
];

// Joint index to name mapping for feedback coloring
const JOINT_INDEX_NAMES = {
    11: "left_shoulder", 12: "right_shoulder",
    13: "left_elbow", 14: "right_elbow",
    15: "left_wrist", 16: "right_wrist",
    23: "left_hip", 24: "right_hip",
    25: "left_knee", 26: "right_knee",
    27: "left_ankle", 28: "right_ankle",
};

// Joint angle definitions: [point_a, vertex, point_b] — angle measured at vertex
// Mirrors src/utils/constants.py JOINT_ANGLES
const JOINT_ANGLES = {
    left_knee:      [23, 25, 27],
    right_knee:     [24, 26, 28],
    left_hip:       [11, 23, 25],
    right_hip:      [12, 24, 26],
    left_elbow:     [11, 13, 15],
    right_elbow:    [12, 14, 16],
    left_shoulder:  [23, 11, 13],
    right_shoulder: [24, 12, 14],
    left_ankle:     [25, 27, 29],
    right_ankle:    [26, 28, 30],
};

// Which angles to show per exercise (primary + secondary)
// Mirrors src/feature_extraction/exercise_features.py
const EXERCISE_ANGLES = {
    squat:          ["left_knee","right_knee","left_hip","right_hip","left_ankle","right_ankle"],
    lunge:          ["left_knee","right_knee","left_hip","right_hip","left_ankle","right_ankle"],
    deadlift:       ["left_hip","right_hip","left_knee","right_knee","left_shoulder","right_shoulder"],
    bench_press:    ["left_elbow","right_elbow","left_shoulder","right_shoulder"],
    overhead_press: ["left_elbow","right_elbow","left_shoulder","right_shoulder","left_hip","right_hip"],
    pullup:         ["left_elbow","right_elbow","left_shoulder","right_shoulder"],
    pushup:         ["left_elbow","right_elbow","left_shoulder","right_shoulder","left_hip","right_hip"],
    plank:          ["left_hip","right_hip","left_shoulder","right_shoulder","left_knee","right_knee"],
    bicep_curl:     ["left_elbow","right_elbow","left_shoulder","right_shoulder"],
    tricep_dip:     ["left_elbow","right_elbow","left_shoulder","right_shoulder"],
};

// Last world landmarks for angle computation
let lastWorldLandmarks = null;

// ── MediaPipe Hybrid Mode Functions ─────────────────────────────────────

async function initMediaPipe() {
    try {
        if (typeof FilesetResolver === "undefined" || typeof PoseLandmarker === "undefined") {
            console.warn("MediaPipe Tasks Vision SDK not loaded — using legacy server mode");
            return false;
        }
        const vision = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
        );
        poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: "/static/models/pose_landmarker_full.task",
                delegate: "GPU",
            },
            runningMode: "VIDEO",
            numPoses: 1,
            minPoseDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5,
        });
        console.log("MediaPipe PoseLandmarker initialised (browser-side, GPU)");
        return true;
    } catch (err) {
        console.warn("MediaPipe init failed, falling back to server mode:", err);
        poseLandmarker = null;
        return false;
    }
}

function getJointColor(idx, jointFeedback) {
    const name = JOINT_INDEX_NAMES[idx];
    if (!name || !jointFeedback) return "#9CA3AF";
    const status = jointFeedback[name];
    if (status === "incorrect") return "#EF4444";
    if (status === "warning") return "#F59E0B";
    if (status === "correct") return "#34D399";
    return "#9CA3AF";
}

function getConnectionColor(a, b, jointFeedback) {
    const ca = getJointColor(a, jointFeedback);
    const cb = getJointColor(b, jointFeedback);
    if (ca === "#EF4444" || cb === "#EF4444") return "#EF4444";
    if (ca === "#F59E0B" || cb === "#F59E0B") return "#F59E0B";
    if (ca === "#34D399" && cb === "#34D399") return "#34D399";
    return "#9CA3AF";
}

function computeAngle3D(a, b, c) {
    // Angle at vertex b formed by points a-b-c, using world coordinates
    const bax = a.x - b.x, bay = a.y - b.y, baz = a.z - b.z;
    const bcx = c.x - b.x, bcy = c.y - b.y, bcz = c.z - b.z;
    const baNorm = Math.sqrt(bax * bax + bay * bay + baz * baz);
    const bcNorm = Math.sqrt(bcx * bcx + bcy * bcy + bcz * bcz);
    if (baNorm < 1e-6 || bcNorm < 1e-6) return null;
    const dot = bax * bcx + bay * bcy + baz * bcz;
    const cosine = Math.max(-1, Math.min(1, dot / (baNorm * bcNorm)));
    return Math.round(Math.acos(cosine) * (180 / Math.PI));
}

function getAngleColor(angleName, jointFeedback) {
    if (!jointFeedback || !jointFeedback[angleName]) return "#FFFFFF";
    const status = jointFeedback[angleName];
    if (status === "incorrect") return "#EF4444";
    if (status === "warning") return "#F59E0B";
    if (status === "correct") return "#34D399";
    return "#FFFFFF";
}

function drawAngleLabels(landmarks, worldLandmarks, jointFeedback, exercise) {
    if (!skeletonCtx || !skeletonCanvas || !worldLandmarks) return;
    const w = skeletonCanvas.width;
    const h = skeletonCanvas.height;

    const anglesToShow = EXERCISE_ANGLES[exercise] || Object.keys(JOINT_ANGLES);

    for (const angleName of anglesToShow) {
        const def = JOINT_ANGLES[angleName];
        if (!def) continue;
        const [idxA, idxB, idxC] = def;

        // Check visibility
        if (landmarks[idxA].visibility < 0.5 ||
            landmarks[idxB].visibility < 0.5 ||
            landmarks[idxC].visibility < 0.5) continue;

        // Compute angle using world landmarks (accurate 3D)
        const angle = computeAngle3D(worldLandmarks[idxA], worldLandmarks[idxB], worldLandmarks[idxC]);
        if (angle === null) continue;

        // Vertex position in image space
        const vx = landmarks[idxB].x * w;
        const vy = landmarks[idxB].y * h;

        // Push label outward from bone midpoint
        const ax = landmarks[idxA].x * w, ay = landmarks[idxA].y * h;
        const cx = landmarks[idxC].x * w, cy = landmarks[idxC].y * h;
        const midX = (ax + cx) / 2, midY = (ay + cy) / 2;
        let outX = vx - midX, outY = vy - midY;
        const outLen = Math.sqrt(outX * outX + outY * outY);
        if (outLen > 1) {
            outX = outX / outLen * 40;
            outY = outY / outLen * 28;
        } else {
            outX = angleName.startsWith("left_") ? -40 : 40;
            outY = 0;
        }

        const labelX = vx + outX;
        const labelY = vy + outY;

        const color = getAngleColor(angleName, jointFeedback);
        const label = angle + "\u00B0";

        // Measure text
        skeletonCtx.font = "bold 14px 'Outfit', sans-serif";
        const metrics = skeletonCtx.measureText(label);
        const textW = metrics.width;
        const textH = 14;
        const padX = 5, padY = 3;

        // Background pill
        const bgX = labelX - padX;
        const bgY = labelY - textH - padY;
        const bgW = textW + padX * 2;
        const bgH = textH + padY * 2;
        const radius = 4;

        skeletonCtx.fillStyle = "rgba(0, 0, 0, 0.75)";
        skeletonCtx.beginPath();
        if (skeletonCtx.roundRect) {
            skeletonCtx.roundRect(bgX, bgY, bgW, bgH, radius);
        } else {
            skeletonCtx.rect(bgX, bgY, bgW, bgH);
        }
        skeletonCtx.fill();
        skeletonCtx.strokeStyle = color;
        skeletonCtx.lineWidth = 1.5;
        skeletonCtx.beginPath();
        if (skeletonCtx.roundRect) {
            skeletonCtx.roundRect(bgX, bgY, bgW, bgH, radius);
        } else {
            skeletonCtx.rect(bgX, bgY, bgW, bgH);
        }
        skeletonCtx.stroke();

        // Angle text
        skeletonCtx.fillStyle = color;
        skeletonCtx.fillText(label, labelX, labelY);
    }
}

function drawSkeletonOverlay(landmarks, jointFeedback) {
    if (!skeletonCtx || !skeletonCanvas) return;
    const w = skeletonCanvas.width;
    const h = skeletonCanvas.height;
    skeletonCtx.clearRect(0, 0, w, h);

    // Draw connections with glow
    for (const [a, b] of SKELETON_CONNECTIONS) {
        if (landmarks[a].visibility < 0.5 || landmarks[b].visibility < 0.5) continue;
        const color = getConnectionColor(a, b, jointFeedback);
        skeletonCtx.strokeStyle = color;
        skeletonCtx.lineWidth = 4;
        skeletonCtx.shadowColor = color;
        skeletonCtx.shadowBlur = 10;
        skeletonCtx.beginPath();
        skeletonCtx.moveTo(landmarks[a].x * w, landmarks[a].y * h);
        skeletonCtx.lineTo(landmarks[b].x * w, landmarks[b].y * h);
        skeletonCtx.stroke();
    }
    skeletonCtx.shadowBlur = 0;

    // Draw landmarks with halo (skip face: indices 0-10)
    for (let i = 11; i < 33; i++) {
        if (landmarks[i].visibility < 0.5) continue;
        const color = getJointColor(i, jointFeedback);
        const x = landmarks[i].x * w;
        const y = landmarks[i].y * h;

        // Outer halo glow
        skeletonCtx.beginPath();
        skeletonCtx.arc(x, y, 10, 0, 2 * Math.PI);
        skeletonCtx.fillStyle = color.replace(')', ', 0.2)').replace('rgb', 'rgba');
        skeletonCtx.fill();

        // Inner dot
        skeletonCtx.fillStyle = color;
        skeletonCtx.beginPath();
        skeletonCtx.arc(x, y, 5, 0, 2 * Math.PI);
        skeletonCtx.fill();
        skeletonCtx.strokeStyle = "rgba(0,0,0,0.5)";
        skeletonCtx.lineWidth = 1;
        skeletonCtx.stroke();
    }

    // Draw angle labels on top of skeleton
    drawAngleLabels(landmarks, lastWorldLandmarks, jointFeedback, selectedExercise);

    // Draw session HUD on top of everything (push-ups only)
    if (selectedExercise === "pushup") {
        drawSessionHUD();
    }
}

function updateJointHealth(jointFeedback, exercise) {
    if (!jointHealthList) return;
    const anglesToShow = EXERCISE_ANGLES[exercise] || [];
    let html = '';
    for (const joint of anglesToShow) {
        const status = jointFeedback[joint] || 'neutral';
        const dotClass = status === 'correct' ? 'good' : status === 'warning' ? 'warn' : status === 'incorrect' ? 'bad' : 'neutral';
        const displayName = joint.replace(/_/g, ' ');
        html += `<div class="joint-health-item"><span class="joint-health-dot ${dotClass}"></span>${displayName}</div>`;
    }
    jointHealthList.innerHTML = html;
}

function drawSessionHUD() {
    if (!skeletonCtx || !skeletonCanvas) return;
    const w = skeletonCanvas.width;
    const h = skeletonCanvas.height;
    const s = hudState;

    // ── Rep counter (top-left) ──
    skeletonCtx.font = "bold 42px 'Bebas Neue', 'Outfit', sans-serif";
    skeletonCtx.fillStyle = "rgba(0,0,0,0.6)";
    skeletonCtx.fillRect(8, 8, 120, s.setCount > 0 ? 80 : 52);
    skeletonCtx.fillStyle = "#FFFFFF";
    skeletonCtx.fillText(`REP ${s.repsInSet || s.repCount}`, 16, 44);
    if (s.setCount > 0) {
        skeletonCtx.font = "bold 22px 'Outfit', sans-serif";
        skeletonCtx.fillStyle = "#D4A574";
        skeletonCtx.fillText(`SET ${s.setCount + 1}`, 16, 72);
    }

    // ── Form score (top-right) ──
    if (s.isActive && s.formScore > 0) {
        const scorePct = Math.round(s.formScore * 100);
        const scoreColor = scorePct >= 70 ? "#34D399" : scorePct >= 40 ? "#F59E0B" : "#EF4444";
        const scoreX = w - 70;
        // Background circle
        skeletonCtx.beginPath();
        skeletonCtx.arc(scoreX, 40, 30, 0, Math.PI * 2);
        skeletonCtx.fillStyle = "rgba(0,0,0,0.6)";
        skeletonCtx.fill();
        // Score arc (partial ring showing score)
        skeletonCtx.beginPath();
        skeletonCtx.arc(scoreX, 40, 30, -Math.PI / 2, -Math.PI / 2 + (Math.PI * 2 * s.formScore));
        skeletonCtx.strokeStyle = scoreColor;
        skeletonCtx.lineWidth = 4;
        skeletonCtx.stroke();
        // Score number
        skeletonCtx.font = "bold 22px 'Outfit', sans-serif";
        skeletonCtx.fillStyle = scoreColor;
        skeletonCtx.textAlign = "center";
        skeletonCtx.fillText(scorePct, scoreX, 47);
        skeletonCtx.textAlign = "start";
    }

    // ── Phase indicator (bottom-center) ──
    if (s.isActive && s.phase) {
        const phaseText = {
            "Top position": "GO DOWN",
            "Lowering": "LOWERING...",
            "Bottom position": "GOOD DEPTH! PUSH UP!",
            "Pushing up": "PUSH!",
        }[s.phase] || s.phase;
        skeletonCtx.font = "bold 20px 'Outfit', sans-serif";
        const tm = skeletonCtx.measureText(phaseText);
        const px = (w - tm.width) / 2;
        const py = h - 30;
        skeletonCtx.fillStyle = "rgba(0,0,0,0.6)";
        skeletonCtx.fillRect(px - 12, py - 20, tm.width + 24, 30);
        skeletonCtx.fillStyle = s.phase === "Bottom position" ? "#34D399" : "#FFFFFF";
        skeletonCtx.fillText(phaseText, px, py);
    }

    // ── Progress bar (bottom edge) ──
    if (s.isActive && s.progress) {
        const pct = parseInt(s.progress) || 0;
        if (pct > 0) {
            const barW = (w - 20) * (pct / 100);
            skeletonCtx.fillStyle = "rgba(0,0,0,0.4)";
            skeletonCtx.fillRect(10, h - 6, w - 20, 4);
            const barColor = pct >= 80 ? "#34D399" : pct >= 40 ? "#F59E0B" : "#9CA3AF";
            skeletonCtx.fillStyle = barColor;
            skeletonCtx.fillRect(10, h - 6, barW, 4);
        }
    }
}

function updateHudState(data) {
    const now = Date.now();
    const s = hudState;
    const wasActive = s.isActive;
    const prevSetCount = s.setCount;

    s.repCount = data.rep_count || 0;
    s.setCount = data.set_count || 0;
    s.repsInSet = data.reps_in_set || 0;
    s.formScore = data.form_score || 0;
    s.phase = data.phase || "";
    s.progress = data.progress || "";
    s.isActive = data.is_active === true;

    if (s.isActive) {
        s.lastActiveTime = now;
        s.setJustCompleted = false;
    }

    // Detect set completion (set count increased)
    if (s.setCount > prevSetCount) {
        s.setJustCompleted = true;
        s.completedSetNum = s.setCount;
        s.completedSetReps = s.repCount - s.repsInSet;
    }

    // Clear set-complete overlay after user starts moving again
    if (s.setJustCompleted && s.isActive && s.repsInSet > 0) {
        s.setJustCompleted = false;
    }

    s.wasActive = wasActive;
}

function updateSessionOverlay(data) {
    const overlay = document.getElementById("session-state-overlay");
    const overlayText = document.getElementById("session-state-text");
    const overlaySubtext = document.getElementById("session-state-subtext");
    if (!overlay || !overlayText) return;

    const s = hudState;

    // Determine which state to show
    if (s.setJustCompleted) {
        // Set complete — resting
        overlayText.textContent = `SET ${s.completedSetNum} COMPLETE`;
        overlaySubtext.textContent = `${s.completedSetReps} reps done. Rest. Next set starts when you begin.`;
        overlay.className = "session-state-overlay set-complete";
        overlay.classList.remove("hidden");
    } else if (!s.isActive && data.details && data.details.length > 0 && (data.details[0].includes("full body") || data.details[0].includes("Ensure") || data.details[0].includes("Step into"))) {
        // Out of frame
        overlayText.textContent = "STEP INTO FRAME";
        overlaySubtext.textContent = "Ensure your full body is visible — shoulders, elbows, hips and ankles.";
        overlay.className = "session-state-overlay out-of-frame";
        overlay.classList.remove("hidden");
    } else if (!s.isActive && data.details && data.details.length > 0 && (data.details[0].includes("position") || data.details[0].includes("Extend") || data.details[0].includes("Get into") || data.details[0].includes("Straighten"))) {
        // Not in push-up position yet
        overlayText.textContent = "GET INTO POSITION";
        overlaySubtext.textContent = data.details[0];
        overlay.className = "session-state-overlay setup";
        overlay.classList.remove("hidden");
    } else if (!s.isActive && s.wasActive && s.repCount > 0) {
        // Was exercising, now stopped (resting or repositioning)
        overlayText.textContent = "RESTING";
        overlaySubtext.textContent = `${s.repsInSet || s.repCount} reps so far. Resume when ready.`;
        overlay.className = "session-state-overlay resting";
        overlay.classList.remove("hidden");
    } else if (s.isActive) {
        // Exercising — hide overlay
        overlay.classList.add("hidden");
    }
}

function handleLandmarkResult(data) {
    // Latency tracking
    if (data.client_timestamp && data.timing) {
        const roundTripMs = performance.now() - data.client_timestamp;
        const entry = { roundTrip: Math.round(roundTripMs), server: data.timing.total_ms };
        latencyHistory.push(entry);
        if (latencyHistory.length > LATENCY_WINDOW) latencyHistory.shift();
        if (latencyHistory.length > 0 && latencyHistory.length % 30 === 0) {
            const rts = latencyHistory.map(e => e.roundTrip).sort((a, b) => a - b);
            const p50 = rts[Math.floor(rts.length * 0.5)];
            const p95 = rts[Math.floor(rts.length * 0.95)];
            console.log(`[Hybrid Latency] p50=${p50}ms p95=${p95}ms server=${Math.round(latencyHistory.reduce((s,e)=>s+e.server,0)/latencyHistory.length)}ms`);
        }
    }

    // Update joint feedback for next skeleton draw
    lastJointFeedback = data.joint_feedback || {};

    // Update push-up HUD state and overlays
    if (selectedExercise === "pushup") {
        updateHudState(data);
        updateSessionOverlay(data);
    }

    totalFrameCount++;

    repNumber.textContent = data.rep_count || 0;
    fpsDisplay.textContent = `FPS: ${data.fps || "--"}`;

    isUserActive = data.is_active === true;

    if (isUserActive) {
        activeFrameCount++;
        inactiveOverlay.classList.add("hidden");
    } else {
        inactiveOverlay.classList.remove("hidden");
    }

    // Form score smoothing
    const rawScore = data.form_score != null ? data.form_score : 0;
    if (isUserActive) {
        scoreBuffer.push(rawScore);
        if (scoreBuffer.length > SCORE_BUFFER_SIZE) scoreBuffer.shift();
    }
    const smoothedScore = scoreBuffer.length > 0
        ? scoreBuffer.reduce((a, b) => a + b, 0) / scoreBuffer.length : 0;
    const scorePctLive = Math.round(smoothedScore * 100);

    // Score ring update
    const circumference = 2 * Math.PI * 52; // 326.73
    if (isUserActive && scoreBuffer.length > 0) {
        scoreRingValue.textContent = scorePctLive;
        const offset = circumference - (circumference * scorePctLive / 100);
        scoreRingFill.style.strokeDashoffset = offset;

        if (scorePctLive >= 70) {
            scoreRingFill.style.stroke = 'var(--good)';
            scoreRingContainer.classList.add('good-form');
            scoreRingContainer.classList.remove('mid-form', 'bad-form');
        } else if (scorePctLive >= 40) {
            scoreRingFill.style.stroke = 'var(--warn)';
            scoreRingContainer.classList.add('mid-form');
            scoreRingContainer.classList.remove('good-form', 'bad-form');
        } else {
            scoreRingFill.style.stroke = 'var(--bad)';
            scoreRingContainer.classList.add('bad-form');
            scoreRingContainer.classList.remove('good-form', 'mid-form');
        }

        scoreHistory.push(smoothedScore);
        if (scoreHistory.length > MAX_SCORE_HISTORY) scoreHistory.shift();
        if (scoreHistoryCtx) drawScoreHistory();
    } else if (!isUserActive) {
        scoreRingValue.textContent = '--';
        scoreRingFill.style.strokeDashoffset = circumference;
        scoreRingContainer.classList.remove('good-form', 'mid-form', 'bad-form');
    }

    // Joint health panel
    updateJointHealth(data.joint_feedback || {}, selectedExercise);

    // Status badge
    if (!isUserActive) {
        statusBadge.textContent = "Not Active";
        statusBadge.className = "status-badge neutral";
    } else if (data.is_correct === true) {
        statusBadge.textContent = "Good Form";
        statusBadge.className = "status-badge correct";
    } else if (data.is_correct === false) {
        statusBadge.textContent = "Check Form";
        statusBadge.className = "status-badge incorrect";
    } else {
        statusBadge.textContent = "Detecting...";
        statusBadge.className = "status-badge neutral";
    }

    // Confidence gauge (mini bar)
    const conf = data.confidence != null ? data.confidence : 0;
    const pct = Math.round(conf * 100);
    gaugeBar.style.setProperty('--gauge-width', pct + '%');
    gaugeText.textContent = pct + "%";
    if (conf >= 0.7) gaugeBar.className = "confidence-mini-bar good";
    else if (conf >= 0.4) gaugeBar.className = "confidence-mini-bar mid";
    else gaugeBar.className = "confidence-mini-bar low";

    // Feedback text
    if (!isUserActive) {
        feedbackDetails.textContent = "Begin exercise movement to receive feedback.";
        feedbackDetails.className = "feedback-text";
    } else if (data.details && data.details.length > 0) {
        feedbackDetails.textContent = data.details.join(" | ");
        feedbackDetails.className = "feedback-text issue";
    } else if (data.is_correct) {
        feedbackDetails.textContent = "Your form looks good! Keep it up.";
        feedbackDetails.className = "feedback-text";
    } else {
        feedbackDetails.textContent = "Position yourself in frame to begin.";
        feedbackDetails.className = "feedback-text";
    }

    updateFloatingHud();
    sending = false;
}

// ── DOM References ──────────────────────────────────────────────────────

const exerciseGrid = document.getElementById("exercise-grid");
const homeExerciseTags = document.getElementById("home-exercise-tags");
const guideTitle = document.getElementById("guide-title");
const guideMuscles = document.getElementById("guide-muscles");
const guideInstructions = document.getElementById("guide-instructions");
const guideMistakes = document.getElementById("guide-mistakes");
const beginSessionBtn = document.getElementById("begin-session-btn");
const guideBackBtn = document.getElementById("guide-back-btn");

const video = document.getElementById("local-video");
const canvas = document.getElementById("output-canvas");
const captureCanvas = document.getElementById("capture-canvas");
const ctx = canvas.getContext("2d");
const captureCtx = captureCanvas.getContext("2d");
const noCameraOverlay = document.getElementById("no-camera");
const spinnerOverlay = document.getElementById("spinner-overlay");
const pauseOverlay = document.getElementById("pause-overlay");
const reconnectBanner = document.getElementById("reconnect-banner");

const sessionExerciseLabel = document.getElementById("session-exercise-label");
const sessionTimer = document.getElementById("session-timer");
const repNumber = document.getElementById("rep-number");
const gaugeBar = document.getElementById("gauge-bar");
const gaugeText = document.getElementById("gauge-text");
const statusBadge = document.getElementById("status-badge");
const repHistory = document.getElementById("rep-history");
const feedbackDetails = document.getElementById("feedback-details");
const fpsDisplay = document.getElementById("fps-display");
const classifierLabel = document.getElementById("classifier-label");
const endSessionBtn = document.getElementById("end-session-btn");
const pauseBtn = document.getElementById("pause-btn");
const mirrorBtn = document.getElementById("mirror-btn");
const fullscreenBtn = document.getElementById("fullscreen-btn");

const scoreRingFill = document.getElementById("score-ring-fill");
const scoreRingValue = document.getElementById("score-ring-value");
const scoreRingContainer = document.getElementById("score-ring-container");
const jointHealthList = document.getElementById("joint-health-list");
const inactiveOverlay = document.getElementById("inactive-overlay");
const scoreHistoryCanvas = document.getElementById("score-history-chart");
const scoreHistoryCtx = scoreHistoryCanvas ? scoreHistoryCanvas.getContext("2d") : null;
const repChartWrap = document.getElementById("rep-chart-wrap");

const reportScore = document.getElementById("report-score");
const reportReps = document.getElementById("report-reps");
const reportGood = document.getElementById("report-good");
const reportDuration = document.getElementById("report-duration");
const repTableBody = document.getElementById("rep-table-body");
const commonIssuesList = document.getElementById("common-issues-list");
const commonIssuesSection = document.getElementById("common-issues-section");
const tryAgainBtn = document.getElementById("try-again-btn");
const chooseExerciseBtn = document.getElementById("choose-exercise-btn");

// ── Exercise SVG Icons (used in guide page) ────────────────────────────

const EXERCISE_ICONS = {
    squat: `<svg viewBox="0 0 48 48" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="24" cy="8" r="4"/><path d="M18 20h12M24 16v8M18 24l-4 10M30 24l4 10M14 34l-2 8M34 34l2 8"/>
    </svg>`,
    lunge: `<svg viewBox="0 0 48 48" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="24" cy="7" r="4"/><path d="M24 11v10M20 21l-8 10-2 8M28 21l6 6v12"/>
    </svg>`,
    deadlift: `<svg viewBox="0 0 48 48" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="24" cy="7" r="4"/><path d="M24 11v12M20 23l-4 8v8M28 23l4 8v8M10 40h28"/><rect x="12" y="38" width="24" height="4" rx="2"/>
    </svg>`,
    bench_press: `<svg viewBox="0 0 48 48" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="20" cy="18" r="3"/><path d="M20 21v8M16 29l-2 10M24 29l2 10M8 14h32"/><rect x="6" y="12" width="4" height="4" rx="1"/><rect x="38" y="12" width="4" height="4" rx="1"/>
    </svg>`,
    overhead_press: `<svg viewBox="0 0 48 48" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="24" cy="14" r="4"/><path d="M24 18v12M20 30l-2 12M28 30l2 12M16 6h16"/><path d="M18 18l-2-12M30 18l2-12"/>
    </svg>`,
    pullup: `<svg viewBox="0 0 48 48" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="24" cy="12" r="4"/><path d="M10 4h28M24 16v14M20 30l-2 12M28 30l2 12"/><path d="M18 16l-8-12M30 16l8-12"/>
    </svg>`,
    pushup: `<svg viewBox="0 0 48 48" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="10" cy="22" r="3"/><path d="M13 24l14 2 12 4"/><path d="M18 26l2 10M32 28l4 8"/>
    </svg>`,
    plank: `<svg viewBox="0 0 48 48" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="10" cy="20" r="3"/><path d="M13 22h26"/><path d="M14 22l-2 12M38 22l2 12"/>
    </svg>`,
    bicep_curl: `<svg viewBox="0 0 48 48" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
        <path d="M20 8v14l-6 4"/><circle cx="14" cy="28" r="3"/><path d="M20 8c4-2 8 0 8 6"/><path d="M18 36v6M22 36v6M20 22v14"/>
    </svg>`,
    tricep_dip: `<svg viewBox="0 0 48 48" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="24" cy="10" r="4"/><path d="M24 14v10"/><path d="M18 14l-8 4M30 14l8 4"/><path d="M24 24l-4 8v10M24 24l4 8v10"/><path d="M8 18h4M36 18h4"/>
    </svg>`,
};

// ── Exercise Loading ────────────────────────────────────────────────────

async function loadExercises() {
    exerciseGrid.innerHTML = '<div class="grid-loading">Loading exercises...</div>';
    try {
        const res = await fetch("/api/exercises");
        const data = await res.json();
        exerciseList = data.exercises;
        renderExerciseGrid();
        renderHomeTags();
    } catch (err) {
        exerciseGrid.innerHTML = '<div class="grid-loading">Failed to load exercises. Please refresh.</div>';
        console.error("Failed to load exercises:", err);
    }
}

function renderExerciseGrid() {
    exerciseGrid.innerHTML = "";
    exerciseList.forEach((ex, index) => {
        const card = document.createElement("div");
        card.className = "exercise-card";
        card.setAttribute("data-aos", "fade-up");
        card.setAttribute("data-aos-delay", String(Math.min(index * 50, 400)));

        const musclePills = ex.muscles.map(m => `<span class="muscle-pill">${m}</span>`).join("");
        const imgSrc = EXERCISE_IMAGES[ex.id] || '';

        card.innerHTML = `
            <div class="card-image">
                <img src="${imgSrc}" alt="${ex.display_name}" loading="lazy">
                <div class="card-image-overlay"></div>
            </div>
            <div class="card-body">
                <div class="card-name">${ex.display_name}</div>
                <div class="card-muscles">${musclePills}</div>
            </div>
        `;
        card.addEventListener("click", () => openGuide(ex.id));
        exerciseGrid.appendChild(card);
    });

    requestAnimationFrame(() => {
        if (typeof AOS !== "undefined") AOS.refresh();
    });
}

function renderHomeTags() {
    homeExerciseTags.innerHTML = "";
    exerciseList.forEach((ex) => {
        const tag = document.createElement("span");
        tag.className = "exercise-tag";
        tag.textContent = ex.display_name;
        homeExerciseTags.appendChild(tag);
    });
}

// ── Exercise Guide ──────────────────────────────────────────────────────

async function openGuide(exerciseId) {
    selectedExercise = exerciseId;

    try {
        const res = await fetch(`/api/exercise_guide/${exerciseId}`);
        const data = await res.json();

        guideTitle.textContent = data.display_name;
        guideMuscles.innerHTML = data.muscles
            .map((m) => `<span class="muscle-tag">${m}</span>`)
            .join("");

        // Camera setup notice (if provided for this exercise)
        const cameraSetupEl = document.getElementById("guide-camera-setup");
        if (cameraSetupEl) {
            if (data.camera_setup) {
                cameraSetupEl.textContent = data.camera_setup;
                cameraSetupEl.style.display = "block";
            } else {
                cameraSetupEl.style.display = "none";
            }
        }

        guideInstructions.innerHTML = data.instructions
            .map((s) => `<li>${s}</li>`)
            .join("");
        guideMistakes.innerHTML = data.common_mistakes
            .map((s) => `<li>${s}</li>`)
            .join("");
    } catch {
        guideTitle.textContent = exerciseId.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
        guideMuscles.innerHTML = "";
        guideInstructions.innerHTML = "<li>No instructions available.</li>";
        guideMistakes.innerHTML = "";
    }

    showPage("guide");
}

guideBackBtn.addEventListener("click", () => showPage("exercises"));

// Classifier radio buttons
document.querySelectorAll('input[name="classifier"]').forEach((radio) => {
    radio.addEventListener("change", () => {
        selectedClassifier = radio.value;
        document.querySelectorAll(".radio-card").forEach((card) => card.classList.remove("selected"));
        radio.closest(".radio-card").classList.add("selected");
    });
});
document.querySelector('input[name="classifier"]:checked')?.closest(".radio-card")?.classList.add("selected");

// ── Active Session ──────────────────────────────────────────────────────

beginSessionBtn.addEventListener("click", startSession);
endSessionBtn.addEventListener("click", endSession);
pauseBtn.addEventListener("click", togglePause);
mirrorBtn.addEventListener("click", toggleMirror);
if (fullscreenBtn) fullscreenBtn.addEventListener("click", toggleFullscreen);

// Fullscreen floating controls
const fsExitBtn = document.getElementById("fs-exit-btn");
const fsPauseBtn = document.getElementById("fs-pause-btn");
const fsEndBtn = document.getElementById("fs-end-btn");
if (fsExitBtn) fsExitBtn.addEventListener("click", toggleFullscreen);
if (fsPauseBtn) fsPauseBtn.addEventListener("click", togglePause);
if (fsEndBtn) fsEndBtn.addEventListener("click", endSession);

function toggleFullscreen() {
    const sessionLayout = document.querySelector(".session-layout");
    sessionLayout.classList.toggle("fullscreen-video");
    const isFS = sessionLayout.classList.contains("fullscreen-video");
    fullscreenBtn.innerHTML = isFS ? "&#x2716;" : "&#x26F6;";
    fullscreenBtn.title = isFS ? "Exit fullscreen" : "Toggle fullscreen";
}

// Sync floating HUD with main HUD values
function updateFloatingHud() {
    const fs = document.getElementById("floating-score");
    const fr = document.getElementById("floating-reps");
    const ff = document.getElementById("floating-fps");
    if (!fs) return;
    const score = scoreRingValue.textContent;
    const scoreColor = score !== "--" && parseInt(score) >= 70 ? "#34D399"
        : score !== "--" && parseInt(score) >= 40 ? "#F59E0B" : "#EF4444";
    fs.innerHTML = `<span style="color:${score === "--" ? "#999" : scoreColor}">${score}</span> <small>FORM</small>`;
    fr.innerHTML = `${repNumber.textContent} <small>REPS</small>`;
    ff.textContent = fpsDisplay.textContent;
}

async function startSession() {
    isPaused = false;
    pausedElapsed = 0;
    sessionEnding = false;
    pauseBtn.textContent = "Pause";
    pauseOverlay.classList.add("hidden");
    reconnectBanner.classList.add("hidden");

    repNumber.textContent = "0";
    repHistory.innerHTML = "";
    feedbackDetails.textContent = "Perform the exercise to receive feedback.";
    statusBadge.textContent = "Waiting";
    statusBadge.className = "status-badge neutral";
    gaugeBar.style.setProperty('--gauge-width', '0%');
    gaugeBar.className = 'confidence-mini-bar';
    gaugeText.textContent = "--";

    // Reset form score state
    scoreBuffer = [];
    scoreHistory = [];
    isUserActive = false;
    activeFrameCount = 0;
    totalFrameCount = 0;
    scoreRingValue.textContent = "--";
    scoreRingFill.style.strokeDashoffset = 2 * Math.PI * 52;
    scoreRingFill.style.stroke = 'var(--primary)';
    scoreRingContainer.classList.remove('good-form', 'mid-form', 'bad-form');
    inactiveOverlay.classList.add("hidden");
    if (jointHealthList) jointHealthList.innerHTML = '';
    if (scoreHistoryCtx) drawScoreHistory();

    const classifierNames = {
        rule_based: "Rule-Based",
        ml: "Random Forest",
        bilstm: "BiLSTM",
    };
    sessionExerciseLabel.textContent =
        selectedExercise.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
    classifierLabel.textContent = classifierNames[selectedClassifier] || "Rule-Based";

    // Push-ups: disable mirror by default (side view, and legacy mode text gets flipped)
    const effectiveMirror = selectedExercise === "pushup" ? false : isMirrored;
    if (effectiveMirror) {
        canvas.classList.add("mirrored");
        mirrorBtn.classList.add("active");
    } else {
        canvas.classList.remove("mirrored");
        mirrorBtn.classList.remove("active");
    }

    showPage("session");

    spinnerOverlay.classList.remove("hidden");
    noCameraOverlay.classList.add("hidden");

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

        video.srcObject = stream;
        await video.play();

        const vw = video.videoWidth || 1280;
        const vh = video.videoHeight || 720;
        // Legacy mode capture canvas — 720px max for server-side processing
        const maxDim = 720;
        const scale = Math.min(maxDim / Math.max(vw, vh), 1.0);
        captureCanvas.width = Math.round(vw * scale);
        captureCanvas.height = Math.round(vh * scale);
        canvas.width = vw;
        canvas.height = vh;
    } catch (err) {
        spinnerOverlay.classList.add("hidden");
        noCameraOverlay.classList.remove("hidden");
        noCameraOverlay.textContent = "";
        const p = document.createElement("p");
        p.textContent = "Camera access denied. Please allow camera permissions.";
        noCameraOverlay.appendChild(p);
        console.error("Camera error:", err);
        return;
    }

    // Initialise browser-side MediaPipe (hybrid mode)
    skeletonCanvas = document.getElementById("skeleton-canvas");
    if (skeletonCanvas) {
        skeletonCanvas.width = video.videoWidth || 640;
        skeletonCanvas.height = video.videoHeight || 480;
        skeletonCtx = skeletonCanvas.getContext("2d");
    }

    hybridMode = await initMediaPipe();
    if (hybridMode) {
        console.log("Hybrid mode enabled — browser-side MediaPipe + server classification");
        document.querySelector(".video-wrapper").classList.add("hybrid-mode");
        if (isMirrored) {
            video.classList.add("mirrored");
            if (skeletonCanvas) skeletonCanvas.classList.add("mirrored");
        }
    } else {
        console.log("Legacy mode — server-side MediaPipe");
        document.querySelector(".video-wrapper").classList.remove("hybrid-mode");
    }

    connectSocket();

    sessionStartTime = Date.now();
    sessionTimerInterval = setInterval(updateTimer, 1000);
    updateTimer();

    // Reset push-up HUD state
    hudState.repCount = 0;
    hudState.setCount = 0;
    hudState.repsInSet = 0;
    hudState.formScore = 0;
    hudState.phase = "";
    hudState.progress = "";
    hudState.isActive = false;
    hudState.setJustCompleted = false;

    // Hide session state overlay
    const stateOverlay = document.getElementById("session-state-overlay");
    if (stateOverlay) stateOverlay.classList.add("hidden");

    // Start recording (low-res for space efficiency)
    startRecording();
}

// ── Session Recording ──────────────────────────────────────────────────

function startRecording() {
    try {
        recordedChunks = [];
        recordingBlob = null;

        // Create a low-res composite canvas (360p for space)
        recordingCanvas = document.createElement("canvas");
        recordingCanvas.width = 480;
        recordingCanvas.height = 360;
        recordingCtx = recordingCanvas.getContext("2d");

        const recStream = recordingCanvas.captureStream(10); // 10 fps
        const mimeType = MediaRecorder.isTypeSupported("video/webm;codecs=vp9")
            ? "video/webm;codecs=vp9"
            : "video/webm";
        mediaRecorder = new MediaRecorder(recStream, {
            mimeType,
            videoBitsPerSecond: 500000, // 500kbps — low bitrate
        });
        mediaRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) recordedChunks.push(e.data);
        };
        mediaRecorder.start(1000); // chunk every 1s

        // Show REC indicator
        const recEl = document.getElementById("rec-indicator");
        if (recEl) recEl.classList.remove("hidden");

        console.log("Recording started (360p, 10fps, 500kbps)");
    } catch (err) {
        console.warn("Recording not supported:", err);
        mediaRecorder = null;
    }
}

function updateRecordingFrame() {
    if (!recordingCtx || !recordingCanvas || !mediaRecorder) return;
    const rw = recordingCanvas.width;
    const rh = recordingCanvas.height;

    // Draw video feed
    if (video.videoWidth) {
        recordingCtx.drawImage(video, 0, 0, rw, rh);
    }

    // Overlay skeleton canvas if in hybrid mode
    if (hybridMode && skeletonCanvas) {
        recordingCtx.drawImage(skeletonCanvas, 0, 0, rw, rh);
    } else if (!hybridMode && canvas.width > 0) {
        // Legacy mode — server-rendered frame is on output canvas
        recordingCtx.drawImage(canvas, 0, 0, rw, rh);
    }
}

function stopRecording() {
    const recEl = document.getElementById("rec-indicator");
    if (recEl) recEl.classList.add("hidden");

    if (!mediaRecorder || mediaRecorder.state === "inactive") return;

    return new Promise((resolve) => {
        mediaRecorder.onstop = () => {
            recordingBlob = new Blob(recordedChunks, { type: mediaRecorder.mimeType });
            console.log(`Recording saved: ${(recordingBlob.size / 1024 / 1024).toFixed(1)}MB`);
            recordedChunks = [];
            resolve();
        };
        mediaRecorder.stop();
    });
}

function connectSocket() {
    socket = io({
        reconnection: true,
        reconnectionAttempts: 5,
        reconnectionDelay: 1000,
    });

    socket.on("connect", () => {
        console.log("Connected to server");
        reconnectBanner.classList.add("hidden");
        if (!sessionEnding && currentPage === "session") {
            socket.emit("start_session", {
                exercise: selectedExercise,
                classifier: selectedClassifier,
            });
        }
    });

    socket.on("disconnect", () => {
        console.log("Disconnected from server");
        if (currentPage === "session") {
            reconnectBanner.textContent = "Connection lost \u2014 reconnecting...";
            reconnectBanner.className = "reconnect-banner warning";
        }
    });

    socket.io.on("reconnect", () => {
        console.log("Reconnected to server");
        reconnectBanner.classList.add("hidden");
    });

    socket.io.on("reconnect_failed", () => {
        reconnectBanner.textContent = "Connection lost. Please refresh the page.";
        reconnectBanner.className = "reconnect-banner error";
    });

    socket.on("session_started", () => {
        console.log("Session started on server");
        spinnerOverlay.classList.add("hidden");
        sendLoop();
    });

    socket.on("processed", handleProcessedFrame);
    socket.on("result", handleLandmarkResult);     // Hybrid mode response
    socket.on("rep_completed", handleRepCompleted);
    socket.on("session_report", handleSessionReport);
}

function handleProcessedFrame(data) {
    if (!data.image) { sending = false; return; }

    // Latency instrumentation
    if (data.client_timestamp && data.timing) {
        const roundTripMs = performance.now() - data.client_timestamp;
        const entry = {
            roundTrip: Math.round(roundTripMs),
            server: data.timing.total_ms,
            decode: data.timing.decode_ms,
            pipeline: data.timing.pipeline_ms,
            encode: data.timing.encode_ms,
        };
        latencyHistory.push(entry);
        if (latencyHistory.length > LATENCY_WINDOW) latencyHistory.shift();
        // Log p50/p95 every 30 frames
        if (latencyHistory.length > 0 && latencyHistory.length % 30 === 0) {
            const rts = latencyHistory.map(e => e.roundTrip).sort((a, b) => a - b);
            const p50 = rts[Math.floor(rts.length * 0.5)];
            const p95 = rts[Math.floor(rts.length * 0.95)];
            const avgServer = Math.round(latencyHistory.reduce((s, e) => s + e.server, 0) / latencyHistory.length);
            console.log(`[Latency] p50=${p50}ms p95=${p95}ms server_avg=${avgServer}ms (decode=${Math.round(latencyHistory.reduce((s,e)=>s+e.decode,0)/latencyHistory.length)}ms pipe=${Math.round(latencyHistory.reduce((s,e)=>s+e.pipeline,0)/latencyHistory.length)}ms enc=${Math.round(latencyHistory.reduce((s,e)=>s+e.encode,0)/latencyHistory.length)}ms)`);
        }
    }

    frameImg.onload = () => {
        if (canvas.width !== frameImg.width || canvas.height !== frameImg.height) {
            canvas.width = frameImg.width;
            canvas.height = frameImg.height;
        }
        ctx.drawImage(frameImg, 0, 0);
    };
    frameImg.src = data.image;

    // Update push-up HUD state and overlays
    if (selectedExercise === "pushup") {
        updateHudState(data);
        updateSessionOverlay(data);
    }

    totalFrameCount++;

    repNumber.textContent = data.rep_count || 0;
    fpsDisplay.textContent = `FPS: ${data.fps || "--"}`;

    // Handle active/inactive state
    isUserActive = data.is_active === true;

    if (isUserActive) {
        activeFrameCount++;
        inactiveOverlay.classList.add("hidden");
    } else {
        inactiveOverlay.classList.remove("hidden");
    }

    // Update form score with smoothing
    const rawScore = data.form_score != null ? data.form_score : 0;
    if (isUserActive) {
        scoreBuffer.push(rawScore);
        if (scoreBuffer.length > SCORE_BUFFER_SIZE) scoreBuffer.shift();
    }
    const smoothedScore = scoreBuffer.length > 0
        ? scoreBuffer.reduce((a, b) => a + b, 0) / scoreBuffer.length
        : 0;
    const scorePctLive = Math.round(smoothedScore * 100);

    // Score ring update
    const circumference = 2 * Math.PI * 52;
    if (isUserActive && scoreBuffer.length > 0) {
        scoreRingValue.textContent = scorePctLive;
        const offset = circumference - (circumference * scorePctLive / 100);
        scoreRingFill.style.strokeDashoffset = offset;

        if (scorePctLive >= 70) {
            scoreRingFill.style.stroke = 'var(--good)';
            scoreRingContainer.classList.add('good-form');
            scoreRingContainer.classList.remove('mid-form', 'bad-form');
        } else if (scorePctLive >= 40) {
            scoreRingFill.style.stroke = 'var(--warn)';
            scoreRingContainer.classList.add('mid-form');
            scoreRingContainer.classList.remove('good-form', 'bad-form');
        } else {
            scoreRingFill.style.stroke = 'var(--bad)';
            scoreRingContainer.classList.add('bad-form');
            scoreRingContainer.classList.remove('good-form', 'mid-form');
        }

        scoreHistory.push(smoothedScore);
        if (scoreHistory.length > MAX_SCORE_HISTORY) scoreHistory.shift();
        if (scoreHistoryCtx) drawScoreHistory();
    } else if (!isUserActive) {
        scoreRingValue.textContent = '--';
        scoreRingFill.style.strokeDashoffset = circumference;
        scoreRingContainer.classList.remove('good-form', 'mid-form', 'bad-form');
    }

    // Joint health panel
    updateJointHealth(data.joint_feedback || {}, selectedExercise);

    // Status badge
    if (!isUserActive) {
        statusBadge.textContent = "Not Active";
        statusBadge.className = "status-badge neutral";
    } else if (data.is_correct === true) {
        statusBadge.textContent = "Good Form";
        statusBadge.className = "status-badge correct";
    } else if (data.is_correct === false) {
        statusBadge.textContent = "Check Form";
        statusBadge.className = "status-badge incorrect";
    } else {
        statusBadge.textContent = "Detecting...";
        statusBadge.className = "status-badge neutral";
    }

    // Confidence gauge (mini bar)
    const conf = data.confidence != null ? data.confidence : 0;
    const pct = Math.round(conf * 100);
    gaugeBar.style.setProperty('--gauge-width', pct + '%');
    gaugeText.textContent = pct + "%";
    if (conf >= 0.7) gaugeBar.className = "confidence-mini-bar good";
    else if (conf >= 0.4) gaugeBar.className = "confidence-mini-bar mid";
    else gaugeBar.className = "confidence-mini-bar low";

    if (!isUserActive) {
        feedbackDetails.textContent = "Begin exercise movement to receive feedback.";
        feedbackDetails.className = "feedback-text";
    } else if (data.details && data.details.length > 0) {
        feedbackDetails.textContent = data.details.join(" | ");
        feedbackDetails.className = "feedback-text issue";
    } else if (data.is_correct) {
        feedbackDetails.textContent = "Your form looks good! Keep it up.";
        feedbackDetails.className = "feedback-text";
    } else {
        feedbackDetails.textContent = "Position yourself in frame to begin.";
        feedbackDetails.className = "feedback-text";
    }

    updateFloatingHud();
    sending = false;
}

function handleRepCompleted(repInfo) {
    const dot = document.createElement("div");
    dot.className = "rep-dot";
    dot.title = `Rep ${repInfo.rep_num}: ${Math.round(repInfo.form_score * 100)}%`;

    if (repInfo.form_score >= 0.7) {
        dot.classList.add("good");
    } else if (repInfo.form_score >= 0.4) {
        dot.classList.add("mid");
    } else {
        dot.classList.add("bad");
    }

    dot.textContent = repInfo.rep_num;
    repHistory.appendChild(dot);
    repHistory.scrollLeft = repHistory.scrollWidth;
}

function handleSessionReport(summary) {
    showReport(summary);
    if (socket) {
        socket.disconnect();
        socket = null;
    }
    sessionEnding = false;
}

// ── Pause / Resume ──────────────────────────────────────────────────────

function togglePause() {
    if (isPaused) {
        isPaused = false;
        pauseBtn.textContent = "Pause";
        pauseOverlay.classList.add("hidden");
        sessionStartTime = Date.now() - pausedElapsed;
        sessionTimerInterval = setInterval(updateTimer, 1000);
        sendLoop();
    } else {
        isPaused = true;
        pauseBtn.textContent = "Resume";
        pauseOverlay.classList.remove("hidden");
        pausedElapsed = Date.now() - sessionStartTime;
        if (animFrameId) {
            cancelAnimationFrame(animFrameId);
            animFrameId = null;
        }
        if (sessionTimerInterval) {
            clearInterval(sessionTimerInterval);
            sessionTimerInterval = null;
        }
    }
}

function toggleMirror() {
    isMirrored = !isMirrored;
    canvas.classList.toggle("mirrored", isMirrored);
    mirrorBtn.classList.toggle("active", isMirrored);
}

// ── Session Controls ────────────────────────────────────────────────────

async function endSession() {
    sessionEnding = true;

    // Stop recording first (before stopping capture)
    await stopRecording();

    if (socket && socket.connected) {
        socket.emit("end_session");
    } else {
        showReport({ total_reps: 0, good_reps: 0, avg_form_score: 0, duration_sec: 0, reps: [], common_issues: [] });
    }

    if (sessionTimerInterval) {
        clearInterval(sessionTimerInterval);
        sessionTimerInterval = null;
    }

    stopCapture();
}

function stopCapture() {
    if (animFrameId) {
        cancelAnimationFrame(animFrameId);
        animFrameId = null;
    }

    if (stream) {
        stream.getTracks().forEach((t) => t.stop());
        stream = null;
    }

    video.srcObject = null;
    noCameraOverlay.classList.remove("hidden");
    spinnerOverlay.classList.add("hidden");
    pauseOverlay.classList.add("hidden");
    reconnectBanner.classList.add("hidden");
}

function sendLoop() {
    animFrameId = requestAnimationFrame(sendLoop);

    // Update recording composite every frame (even when not sending to server)
    updateRecordingFrame();

    const now = performance.now();
    if (now - lastSendTime < SEND_INTERVAL_MS) return;
    if (!socket || !socket.connected) return;
    if (!video.videoWidth) return;
    if (isPaused) return;

    if (sending && now - sendStartTime > SEND_TIMEOUT_MS) {
        sending = false;
    }
    if (sending) return;

    lastSendTime = now;

    if (hybridMode && poseLandmarker) {
        // ── HYBRID MODE: Browser-side MediaPipe ──
        let result;
        try {
            result = poseLandmarker.detectForVideo(video, Math.round(now));
        } catch (e) {
            return; // skip frame on detection error
        }
        if (!result || !result.landmarks || result.landmarks.length === 0) return;

        const lm = result.landmarks[0];
        const wlm = result.worldLandmarks[0];

        // Store world landmarks for angle computation
        lastWorldLandmarks = wlm;

        // Draw skeleton + angle labels locally with last known joint feedback
        drawSkeletonOverlay(lm, lastJointFeedback);

        // Send only landmarks to server (~1KB vs ~50KB JPEG)
        sendStartTime = now;
        sending = true;

        const lmFlat = new Float32Array(33 * 4);
        const wlmFlat = new Float32Array(33 * 3);
        for (let i = 0; i < 33; i++) {
            lmFlat[i * 4] = lm[i].x;
            lmFlat[i * 4 + 1] = lm[i].y;
            lmFlat[i * 4 + 2] = lm[i].z;
            lmFlat[i * 4 + 3] = lm[i].visibility;
            wlmFlat[i * 3] = wlm[i].x;
            wlmFlat[i * 3 + 1] = wlm[i].y;
            wlmFlat[i * 3 + 2] = wlm[i].z;
        }

        socket.emit("landmarks", {
            landmarks: Array.from(lmFlat),
            world_landmarks: Array.from(wlmFlat),
            timestamp: Math.round(now),
        });
    } else {
        // ── LEGACY MODE: Server-side MediaPipe ──
        sendStartTime = now;
        sending = true;

        captureCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
        const dataUrl = captureCanvas.toDataURL("image/jpeg", 0.5);

        socket.emit("frame", {
            image: dataUrl,
            timestamp: Math.round(now),
        });
    }
}

function updateTimer() {
    if (!sessionStartTime) return;
    const elapsed = Math.floor((Date.now() - sessionStartTime) / 1000);
    const mins = Math.floor(elapsed / 60).toString().padStart(2, "0");
    const secs = (elapsed % 60).toString().padStart(2, "0");
    sessionTimer.textContent = `${mins}:${secs}`;
}

// ── Session Report ──────────────────────────────────────────────────────

function showReport(summary) {
    if (!summary) summary = { total_reps: 0, good_reps: 0, avg_form_score: 0, duration_sec: 0, reps: [], common_issues: [] };
    const totalReps = summary.total_reps || 0;
    const goodReps = summary.good_reps || 0;
    const avgScore = summary.avg_form_score || 0;
    const duration = summary.duration_sec || 0;

    const scorePct = Math.round(avgScore * 100);
    reportScore.textContent = scorePct + "%";
    const scoreCard = reportScore.closest(".score-card");

    const heroIcon = document.getElementById("report-hero-icon");
    const reportSubtitle = document.getElementById("report-subtitle");
    if (scorePct >= 70) {
        scoreCard.className = "report-card score-card score-good";
        heroIcon.textContent = "\uD83C\uDFC6";
        reportSubtitle.textContent = "Excellent form! Keep up the great work.";
    } else if (scorePct >= 40) {
        scoreCard.className = "report-card score-card score-mid";
        heroIcon.textContent = "\uD83D\uDCAA";
        reportSubtitle.textContent = "Good effort! Focus on the tips below to improve.";
    } else {
        scoreCard.className = "report-card score-card score-bad";
        heroIcon.textContent = "\uD83C\uDFCB\uFE0F";
        reportSubtitle.textContent = "Keep practising \u2014 review the feedback to nail your form.";
    }

    reportReps.textContent = totalReps;
    reportGood.textContent = goodReps;

    if (duration >= 60) {
        const m = Math.floor(duration / 60);
        const s = Math.round(duration % 60);
        reportDuration.textContent = `${m}m ${s}s`;
    } else {
        reportDuration.textContent = Math.round(duration) + "s";
    }

    renderRepChart(summary.reps);

    repTableBody.innerHTML = "";
    if (summary.reps && summary.reps.length > 0) {
        summary.reps.forEach((rep) => {
            const tr = document.createElement("tr");
            const scorePctRep = Math.round(rep.form_score * 100);
            const scoreClass = scorePctRep >= 70 ? "good" : scorePctRep >= 40 ? "mid" : "bad";

            const tdNum = document.createElement("td");
            tdNum.textContent = rep.rep_num;

            const tdScore = document.createElement("td");
            const scoreSpan = document.createElement("span");
            scoreSpan.className = `rep-score ${scoreClass}`;
            scoreSpan.textContent = scorePctRep + "%";
            tdScore.appendChild(scoreSpan);

            const tdIssues = document.createElement("td");
            tdIssues.textContent = rep.issues && rep.issues.length > 0 ? rep.issues.join("; ") : "None";

            tr.appendChild(tdNum);
            tr.appendChild(tdScore);
            tr.appendChild(tdIssues);
            repTableBody.appendChild(tr);
        });
    } else {
        const tr = document.createElement("tr");
        const td = document.createElement("td");
        td.colSpan = 3;
        td.textContent = "No reps recorded.";
        tr.appendChild(td);
        repTableBody.appendChild(tr);
    }

    commonIssuesList.innerHTML = "";
    if (summary.common_issues && summary.common_issues.length > 0) {
        commonIssuesSection.style.display = "block";
        summary.common_issues.forEach((item) => {
            const li = document.createElement("li");
            const strong = document.createElement("strong");
            strong.textContent = item.issue;
            li.appendChild(strong);
            li.appendChild(document.createTextNode(` (${item.count} reps)`));
            commonIssuesList.appendChild(li);
        });
    } else {
        commonIssuesSection.style.display = "none";
    }

    // ── Set breakdown (push-ups) ──
    const setSection = document.getElementById("set-breakdown-section");
    const setList = document.getElementById("set-breakdown-list");
    if (setSection && setList && summary.reps_per_set && summary.reps_per_set.length > 0) {
        setSection.classList.remove("hidden");
        setList.innerHTML = "";
        summary.reps_per_set.forEach((reps, i) => {
            const card = document.createElement("div");
            card.className = "set-card";
            card.innerHTML = `
                <span class="set-card-title">SET ${i + 1}</span>
                <div class="set-card-stats">
                    <span>${reps} reps</span>
                </div>
            `;
            setList.appendChild(card);
        });
    } else if (setSection) {
        setSection.classList.add("hidden");
    }

    // ── Recording download/delete buttons ──
    const dlBtn = document.getElementById("download-recording-btn");
    const delBtn = document.getElementById("delete-recording-btn");
    if (dlBtn && delBtn) {
        if (recordingBlob && recordingBlob.size > 0) {
            dlBtn.classList.remove("hidden");
            delBtn.classList.remove("hidden");
            dlBtn.onclick = () => {
                const url = URL.createObjectURL(recordingBlob);
                const a = document.createElement("a");
                a.href = url;
                a.download = `${selectedExercise}-session-${new Date().toISOString().slice(0, 10)}.webm`;
                a.click();
                URL.revokeObjectURL(url);
            };
            delBtn.onclick = () => {
                recordingBlob = null;
                recordedChunks = [];
                dlBtn.classList.add("hidden");
                delBtn.classList.add("hidden");
                console.log("Recording deleted");
            };
        } else {
            dlBtn.classList.add("hidden");
            delBtn.classList.add("hidden");
        }
    }

    showPage("report");
}

tryAgainBtn.addEventListener("click", () => {
    openGuide(selectedExercise);
});

chooseExerciseBtn.addEventListener("click", () => {
    showPage("exercises");
});

// ── Score History Chart ──────────────────────────────────────────────────

function drawScoreHistory() {
    if (!scoreHistoryCtx) return;
    const c = scoreHistoryCanvas;
    const w = c.width;
    const h = c.height;
    const sctx = scoreHistoryCtx;

    sctx.clearRect(0, 0, w, h);

    if (scoreHistory.length < 2) return;

    // Draw threshold grid lines
    sctx.strokeStyle = "rgba(255,255,255,0.04)";
    sctx.lineWidth = 1;
    [0.4, 0.7].forEach((y) => {
        const py = h - y * h;
        sctx.beginPath();
        sctx.setLineDash([4, 4]);
        sctx.moveTo(0, py);
        sctx.lineTo(w, py);
        sctx.stroke();
    });
    sctx.setLineDash([]);

    // Draw filled area + line
    const step = w / (MAX_SCORE_HISTORY - 1);
    const offset = MAX_SCORE_HISTORY - scoreHistory.length;

    sctx.beginPath();
    sctx.moveTo(offset * step, h);
    for (let i = 0; i < scoreHistory.length; i++) {
        sctx.lineTo((offset + i) * step, h - scoreHistory[i] * h);
    }
    sctx.lineTo((offset + scoreHistory.length - 1) * step, h);
    sctx.closePath();
    sctx.fillStyle = "rgba(212,165,116,0.08)";
    sctx.fill();

    // Line with glow
    sctx.beginPath();
    for (let i = 0; i < scoreHistory.length; i++) {
        const x = (offset + i) * step;
        const y = h - scoreHistory[i] * h;
        if (i === 0) sctx.moveTo(x, y);
        else sctx.lineTo(x, y);
    }
    sctx.shadowBlur = 6;
    sctx.shadowColor = 'rgba(212, 165, 116, 0.5)';
    sctx.strokeStyle = "#D4A574";
    sctx.lineWidth = 2;
    sctx.stroke();
    sctx.shadowBlur = 0;
}

// ── Per-Rep Bar Chart in Report ─────────────────────────────────────────

function renderRepChart(reps) {
    if (!repChartWrap) return;
    repChartWrap.innerHTML = "";

    if (!reps || reps.length === 0) {
        repChartWrap.style.display = "none";
        return;
    }
    repChartWrap.style.display = "flex";

    reps.forEach((rep) => {
        const scorePct = Math.round(rep.form_score * 100);
        const barColor = scorePct >= 70 ? "var(--good)" : scorePct >= 40 ? "var(--warn)" : "var(--bad)";

        const col = document.createElement("div");
        col.className = "rep-chart-col";

        const label = document.createElement("div");
        label.className = "rep-chart-label";
        label.textContent = scorePct + "%";
        label.style.color = barColor;

        const barWrap = document.createElement("div");
        barWrap.className = "rep-chart-bar-wrap";

        const bar = document.createElement("div");
        bar.className = "rep-chart-bar";
        bar.style.height = scorePct + "%";
        bar.style.background = barColor;

        barWrap.appendChild(bar);

        const repLabel = document.createElement("div");
        repLabel.className = "rep-chart-rep";
        repLabel.textContent = rep.rep_num;

        col.appendChild(label);
        col.appendChild(barWrap);
        col.appendChild(repLabel);
        repChartWrap.appendChild(col);
    });
}

// ── Animated Stat Counter (hero stats) ─────────────────────────────────

function animateCounters() {
    const counters = document.querySelectorAll(".hero-stats .stat-number");
    counters.forEach((counter) => {
        const end = parseInt(counter.textContent, 10);
        if (isNaN(end)) return;
        counter.textContent = "0";
        const duration = 1500;
        const start = performance.now();

        function tick(now) {
            const elapsed = now - start;
            const progress = Math.min(elapsed / duration, 1);
            // ease-out cubic
            const eased = 1 - Math.pow(1 - progress, 3);
            counter.textContent = Math.round(eased * end);
            if (progress < 1) requestAnimationFrame(tick);
        }
        requestAnimationFrame(tick);
    });
}

// Trigger counter animation when hero stats scroll into view
const heroStatsEl = document.querySelector(".hero-stats");
if (heroStatsEl) {
    const counterObserver = new IntersectionObserver(
        (entries) => {
            if (entries[0].isIntersecting) {
                animateCounters();
                counterObserver.disconnect();
            }
        },
        { threshold: 0.5 }
    );
    counterObserver.observe(heroStatsEl);
}

// ═══════════════════════════════════════════════════════════════════════
// VISUAL EFFECTS — Subtle and premium only
// ═══════════════════════════════════════════════════════════════════════

// ── AOS Init ────────────────────────────────────────────────────────────

function initAOS() {
    if (typeof AOS === "undefined") return;
    AOS.init({
        duration: 700,
        offset: 80,
        once: true,
        easing: "ease-out-cubic",
    });
}

// ── Vanta.js NET Background (very subtle) ──────────────────────────────

function initVanta() {
    if (vantaEffect) return; // already running
    if (typeof VANTA === "undefined" || typeof THREE === "undefined") return;
    // Don't init on mobile
    if (window.innerWidth < 768) return;

    const heroEl = document.getElementById("hero-section");
    if (!heroEl) return;

    try {
        vantaEffect = VANTA.NET({
            el: heroEl,
            THREE: THREE,
            mouseControls: true,
            touchControls: false,
            gyroControls: false,
            minHeight: 200,
            minWidth: 200,
            scale: 1.0,
            scaleMobile: 1.0,
            color: 0x2a1a0a,
            backgroundColor: 0x0D0D0D,
            points: 4,
            maxDistance: 22,
            spacing: 20,
            showDots: true,
        });
    } catch (e) {
        console.warn("Vanta init failed:", e);
    }
}

function destroyVanta() {
    if (vantaEffect) {
        try { vantaEffect.destroy(); } catch (e) { /* ignore */ }
        vantaEffect = null;
    }
}

// ── Nav scroll effect ───────────────────────────────────────────────────

window.addEventListener("scroll", () => {
    document.getElementById("main-nav").classList.toggle("scrolled", window.scrollY > 50);
}, { passive: true });

// ── Init ────────────────────────────────────────────────────────────────

loadExercises();
initAOS();

// Delay Vanta slightly to not block initial paint
requestAnimationFrame(() => {
    initVanta();
});
