import {
  FilesetResolver,
  PoseLandmarker,
  type PoseLandmarkerResult,
} from "@mediapipe/tasks-vision";

const WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.34/wasm";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task";

let landmarkerPromise: Promise<PoseLandmarker> | null = null;

export async function getPoseLandmarker(): Promise<PoseLandmarker> {
  if (landmarkerPromise) return landmarkerPromise;
  landmarkerPromise = (async () => {
    const vision = await FilesetResolver.forVisionTasks(WASM_URL);
    const landmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
      runningMode: "VIDEO",
      numPoses: 1,
      minPoseDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });
    return landmarker;
  })();
  return landmarkerPromise;
}

// 33 landmarks × 4 (x, y, z, visibility) flat
export function flattenLandmarks(result: PoseLandmarkerResult): {
  landmarks: number[];
  worldLandmarks: number[];
} | null {
  if (!result.landmarks || result.landmarks.length === 0) return null;
  const lms = result.landmarks[0];
  if (!lms || lms.length !== 33) return null;

  const lmFlat = new Array<number>(33 * 4);
  for (let i = 0; i < 33; i++) {
    const p = lms[i];
    lmFlat[i * 4] = p.x;
    lmFlat[i * 4 + 1] = p.y;
    lmFlat[i * 4 + 2] = p.z;
    lmFlat[i * 4 + 3] = p.visibility ?? 1;
  }

  const wlmFlat = new Array<number>(33 * 3);
  const wlms = result.worldLandmarks?.[0];
  if (wlms && wlms.length === 33) {
    for (let i = 0; i < 33; i++) {
      const p = wlms[i];
      wlmFlat[i * 3] = p.x;
      wlmFlat[i * 3 + 1] = p.y;
      wlmFlat[i * 3 + 2] = p.z;
    }
  }

  return { landmarks: lmFlat, worldLandmarks: wlmFlat };
}

// Minimal skeleton draw: green by default, red for joints in jointFeedback[name] === "incorrect"
export const POSE_CONNECTIONS: Array<[number, number]> = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], // arms + shoulders
  [11, 23], [12, 24], [23, 24],                      // torso
  [23, 25], [25, 27], [27, 29], [27, 31],            // left leg
  [24, 26], [26, 28], [28, 30], [28, 32],            // right leg
];

const LANDMARK_NAMES: Record<number, string> = {
  11: "left_shoulder", 12: "right_shoulder",
  13: "left_elbow", 14: "right_elbow",
  15: "left_wrist", 16: "right_wrist",
  23: "left_hip", 24: "right_hip",
  25: "left_knee", 26: "right_knee",
  27: "left_ankle", 28: "right_ankle",
};

const JOINT_INDICES = Object.keys(LANDMARK_NAMES).map((s) => parseInt(s, 10));

export function drawSkeleton(
  ctx: CanvasRenderingContext2D,
  result: PoseLandmarkerResult,
  w: number,
  h: number,
  jointFeedback: Record<string, string> = {},
): void {
  ctx.clearRect(0, 0, w, h);
  const lms = result.landmarks?.[0];
  if (!lms) return;

  const badLandmarks = new Set<number>();
  for (const [idx, name] of Object.entries(LANDMARK_NAMES)) {
    if (jointFeedback[name] === "incorrect") badLandmarks.add(parseInt(idx, 10));
  }

  // Bones
  ctx.lineWidth = 3;
  ctx.lineCap = "round";
  for (const [a, b] of POSE_CONNECTIONS) {
    const pa = lms[a];
    const pb = lms[b];
    if (!pa || !pb) continue;
    if ((pa.visibility ?? 1) < 0.5 || (pb.visibility ?? 1) < 0.5) continue;
    const bad = badLandmarks.has(a) || badLandmarks.has(b);
    ctx.strokeStyle = bad ? "#F87171" : "#AEE710";
    ctx.beginPath();
    ctx.moveTo(pa.x * w, pa.y * h);
    ctx.lineTo(pb.x * w, pb.y * h);
    ctx.stroke();
  }

  // Joints
  for (const idx of JOINT_INDICES) {
    const p = lms[idx];
    if (!p || (p.visibility ?? 1) < 0.5) continue;
    const bad = badLandmarks.has(idx);
    ctx.fillStyle = bad ? "#F87171" : "#AEE710";
    ctx.beginPath();
    ctx.arc(p.x * w, p.y * h, 5, 0, Math.PI * 2);
    ctx.fill();
  }
}
