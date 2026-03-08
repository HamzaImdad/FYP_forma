/**
 * ExerVision SPA client.
 *
 * Four views: exercise selection, guide, active session, session report.
 * Communicates with Flask server via Socket.IO for real-time frame processing.
 */

// ── View Management ─────────────────────────────────────────────────────

const views = {
    select: document.getElementById("view-select"),
    guide: document.getElementById("view-guide"),
    session: document.getElementById("view-session"),
    report: document.getElementById("view-report"),
};

let currentView = "select";

function showView(name) {
    Object.values(views).forEach((v) => v.classList.remove("active"));
    views[name].classList.add("active");
    currentView = name;
    window.scrollTo(0, 0);
}

// Header click goes home
document.getElementById("header-title").addEventListener("click", () => {
    if (currentView === "session") return; // don't leave active session
    if (socket) {
        socket.disconnect();
        socket = null;
    }
    stopCapture();
    showView("select");
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
let isMirrored = true; // default selfie mirror
let sessionEnding = false; // true while waiting for session_report after end_session

const SEND_INTERVAL_MS = 100; // ~10 FPS target
const SEND_TIMEOUT_MS = 3000; // Reset sending flag after 3s with no response
let lastSendTime = 0;
let sendStartTime = 0;

// Reusable Image object to avoid memory leak (Phase 3A fix)
const frameImg = new Image();

// ── DOM References ──────────────────────────────────────────────────────

const exerciseGrid = document.getElementById("exercise-grid");
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

const reportScore = document.getElementById("report-score");
const reportReps = document.getElementById("report-reps");
const reportGood = document.getElementById("report-good");
const reportDuration = document.getElementById("report-duration");
const repTableBody = document.getElementById("rep-table-body");
const commonIssuesList = document.getElementById("common-issues-list");
const commonIssuesSection = document.getElementById("common-issues-section");
const tryAgainBtn = document.getElementById("try-again-btn");
const chooseExerciseBtn = document.getElementById("choose-exercise-btn");

// ── Exercise SVG Icons (line-art style) ─────────────────────────────────

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

// ── View 1: Exercise Selection ──────────────────────────────────────────

async function loadExercises() {
    // Show loading state
    exerciseGrid.innerHTML = '<div class="grid-loading">Loading exercises...</div>';
    try {
        const res = await fetch("/api/exercises");
        const data = await res.json();
        exerciseList = data.exercises;
        renderExerciseGrid();
    } catch (err) {
        exerciseGrid.innerHTML = '<div class="grid-loading">Failed to load exercises. Please refresh.</div>';
        console.error("Failed to load exercises:", err);
    }
}

function renderExerciseGrid() {
    exerciseGrid.innerHTML = "";
    exerciseList.forEach((ex) => {
        const card = document.createElement("div");
        card.className = "exercise-card";
        const musclePills = ex.muscles.map(m => `<span class="muscle-pill">${m}</span>`).join("");
        card.innerHTML = `
            <div class="card-icon">${EXERCISE_ICONS[ex.id] || EXERCISE_ICONS.squat}</div>
            <div class="card-name">${ex.display_name}</div>
            <div class="card-muscles">${musclePills}</div>
        `;
        card.addEventListener("click", () => openGuide(ex.id));
        exerciseGrid.appendChild(card);
    });
}

// ── View 2: Exercise Guide ──────────────────────────────────────────────

async function openGuide(exerciseId) {
    selectedExercise = exerciseId;

    try {
        const res = await fetch(`/api/exercise_guide/${exerciseId}`);
        const data = await res.json();

        guideTitle.textContent = data.display_name;
        // Exercise data is developer-controlled (exercise_data.json), safe for innerHTML
        guideMuscles.innerHTML = data.muscles
            .map((m) => `<span class="muscle-tag">${m}</span>`)
            .join("");
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

    showView("guide");
}

guideBackBtn.addEventListener("click", () => showView("select"));

// Classifier radio buttons — with :has() CSS fallback (Phase 2C)
document.querySelectorAll('input[name="classifier"]').forEach((radio) => {
    radio.addEventListener("change", () => {
        selectedClassifier = radio.value;
        // JS fallback for CSS :has() (older Firefox)
        document.querySelectorAll(".radio-card").forEach((card) => card.classList.remove("selected"));
        radio.closest(".radio-card").classList.add("selected");
    });
});
// Set initial selected state
document.querySelector('input[name="classifier"]:checked')?.closest(".radio-card")?.classList.add("selected");

// ── View 3: Active Session ──────────────────────────────────────────────

beginSessionBtn.addEventListener("click", startSession);
endSessionBtn.addEventListener("click", endSession);
pauseBtn.addEventListener("click", togglePause);
mirrorBtn.addEventListener("click", toggleMirror);

async function startSession() {
    // Reset state
    isPaused = false;
    pausedElapsed = 0;
    sessionEnding = false;
    pauseBtn.textContent = "Pause";
    pauseOverlay.classList.add("hidden");
    reconnectBanner.classList.add("hidden");

    // Reset UI
    repNumber.textContent = "0";
    repHistory.innerHTML = "";
    feedbackDetails.textContent = "Perform the exercise to receive feedback.";
    statusBadge.textContent = "Waiting";
    statusBadge.className = "status-badge neutral";
    gaugeBar.style.width = "0%";
    gaugeText.textContent = "--";

    // Set exercise label & classifier label
    const classifierNames = {
        rule_based: "Rule-Based",
        ml: "Random Forest",
        bilstm: "BiLSTM",
    };
    sessionExerciseLabel.textContent =
        selectedExercise.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
    classifierLabel.textContent = classifierNames[selectedClassifier] || "Rule-Based";

    // Apply mirror state
    if (isMirrored) {
        canvas.classList.add("mirrored");
        mirrorBtn.classList.add("active");
    } else {
        canvas.classList.remove("mirrored");
        mirrorBtn.classList.remove("active");
    }

    showView("session");

    // Show spinner while initialising
    spinnerOverlay.classList.remove("hidden");
    noCameraOverlay.classList.add("hidden");

    // Start camera
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: "user",
                width: { ideal: 640 },
                height: { ideal: 480 },
            },
            audio: false,
        });

        video.srcObject = stream;
        await video.play();

        // Use smaller capture size for faster encoding/transmission
        const vw = video.videoWidth || 640;
        const vh = video.videoHeight || 480;
        const maxDim = 320;
        const scale = Math.min(maxDim / Math.max(vw, vh), 1.0);
        captureCanvas.width = Math.round(vw * scale);
        captureCanvas.height = Math.round(vh * scale);
        // Display canvas matches video for smooth rendering
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

    // Connect socket
    connectSocket();

    // Start session timer
    sessionStartTime = Date.now();
    sessionTimerInterval = setInterval(updateTimer, 1000);
    updateTimer();
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
        // Only start session if we haven't ended it (avoids ghost session on reconnect)
        if (!sessionEnding && currentView === "session") {
            socket.emit("start_session", {
                exercise: selectedExercise,
                classifier: selectedClassifier,
            });
        }
    });

    socket.on("disconnect", () => {
        console.log("Disconnected from server");
        if (currentView === "session") {
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
    socket.on("rep_completed", handleRepCompleted);
    socket.on("session_report", handleSessionReport);
}

function handleProcessedFrame(data) {
    if (!data.image) { sending = false; return; }

    // Draw annotated frame using reusable Image (Phase 3A: no memory leak)
    frameImg.onload = () => {
        // Only resize canvas when dimensions change (avoids flicker from canvas clear)
        if (canvas.width !== frameImg.width || canvas.height !== frameImg.height) {
            canvas.width = frameImg.width;
            canvas.height = frameImg.height;
        }
        ctx.drawImage(frameImg, 0, 0);
    };
    frameImg.src = data.image;

    // Update HUD
    repNumber.textContent = data.rep_count || 0;
    fpsDisplay.textContent = `FPS: ${data.fps || "--"}`;

    // Status badge
    if (data.is_correct === true) {
        statusBadge.textContent = "Good Form";
        statusBadge.className = "status-badge correct";
    } else if (data.is_correct === false) {
        statusBadge.textContent = "Check Form";
        statusBadge.className = "status-badge incorrect";
    } else {
        statusBadge.textContent = "Detecting...";
        statusBadge.className = "status-badge neutral";
    }

    // Confidence gauge
    const conf = data.confidence != null ? data.confidence : 0;
    const pct = Math.round(conf * 100);
    gaugeBar.style.width = pct + "%";
    gaugeText.textContent = pct + "%";

    if (conf >= 0.7) {
        gaugeBar.className = "gauge-bar good";
    } else if (conf >= 0.4) {
        gaugeBar.className = "gauge-bar mid";
    } else {
        gaugeBar.className = "gauge-bar low";
    }

    // Feedback details — reuse text to avoid DOM churn
    if (data.details && data.details.length > 0) {
        feedbackDetails.textContent = data.details.join(" | ");
        feedbackDetails.className = "feedback-details issue";
    } else if (data.is_correct) {
        feedbackDetails.textContent = "Your form looks good! Keep it up.";
        feedbackDetails.className = "feedback-details";
    } else if (data.is_correct === false && (!data.details || data.details.length === 0)) {
        feedbackDetails.textContent = "Begin exercise movement to receive feedback.";
        feedbackDetails.className = "feedback-details";
    } else {
        feedbackDetails.textContent = "Position yourself in frame to begin.";
        feedbackDetails.className = "feedback-details";
    }

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
    // Clean up socket after report is received
    if (socket) {
        socket.disconnect();
        socket = null;
    }
    sessionEnding = false;
}

// ── Pause / Resume ──────────────────────────────────────────────────────

function togglePause() {
    if (isPaused) {
        // Resume
        isPaused = false;
        pauseBtn.textContent = "Pause";
        pauseOverlay.classList.add("hidden");
        // Adjust start time to account for paused duration
        sessionStartTime = Date.now() - pausedElapsed;
        sessionTimerInterval = setInterval(updateTimer, 1000);
        sendLoop();
    } else {
        // Pause
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

// ── Mirror Toggle ───────────────────────────────────────────────────────

function toggleMirror() {
    isMirrored = !isMirrored;
    canvas.classList.toggle("mirrored", isMirrored);
    mirrorBtn.classList.toggle("active", isMirrored);
}

// ── Session Controls ────────────────────────────────────────────────────

function endSession() {
    sessionEnding = true;

    if (socket && socket.connected) {
        socket.emit("end_session");
    } else {
        // Socket lost before we could request report — show empty report
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

    const now = performance.now();
    if (now - lastSendTime < SEND_INTERVAL_MS) return;
    if (!socket || !socket.connected) return;
    if (!video.videoWidth) return;
    if (isPaused) return;

    // Timeout protection: if server hasn't responded in 3s, reset flag
    if (sending && now - sendStartTime > SEND_TIMEOUT_MS) {
        sending = false;
    }
    if (sending) return;

    lastSendTime = now;
    sendStartTime = now;
    sending = true;

    captureCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
    const dataUrl = captureCanvas.toDataURL("image/jpeg", 0.5);

    socket.emit("frame", {
        image: dataUrl,
        timestamp: Math.round(now),
    });
}

function updateTimer() {
    if (!sessionStartTime) return;
    const elapsed = Math.floor((Date.now() - sessionStartTime) / 1000);
    const mins = Math.floor(elapsed / 60).toString().padStart(2, "0");
    const secs = (elapsed % 60).toString().padStart(2, "0");
    sessionTimer.textContent = `${mins}:${secs}`;
}

// ── View 4: Session Report ──────────────────────────────────────────────

function showReport(summary) {
    const totalReps = summary.total_reps || 0;
    const goodReps = summary.good_reps || 0;
    const avgScore = summary.avg_form_score || 0;
    const duration = summary.duration_sec || 0;

    const scorePct = Math.round(avgScore * 100);
    reportScore.textContent = scorePct + "%";
    const scoreCard = reportScore.closest(".score-card");

    // Motivational hero
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

    // Rep table — use textContent for issue data to prevent XSS (Phase 2A)
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

    // Common issues — use textContent for issue data (Phase 2A)
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

    showView("report");
}

tryAgainBtn.addEventListener("click", () => {
    openGuide(selectedExercise);
});

chooseExerciseBtn.addEventListener("click", () => {
    showView("select");
});

// ── Init ────────────────────────────────────────────────────────────────

loadExercises();
