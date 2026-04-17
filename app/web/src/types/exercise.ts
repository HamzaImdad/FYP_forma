export type ExerciseSlug =
  | "pushup"
  | "squat"
  | "lunge"
  | "deadlift"
  | "pullup"
  | "plank"
  | "bicep_curl"
  | "tricep_dip"
  | "crunch"
  | "lateral_raise"
  | "side_plank";

export type CameraView = "side" | "front" | "three_quarter";

export type CameraGuidance = {
  view: CameraView;
  /** One-line headline — "Place the camera on your side". */
  headline: string;
  /** Supporting detail — framing, height, distance. */
  detail: string;
};

export type Exercise = {
  slug: ExerciseSlug;
  name: string;
  tagline: string;
  primary: boolean;
  /** Weight-based exercises prompt for a kg load before starting the session. */
  isWeighted: boolean;
  /** Pre-session camera placement instructions the user confirms before the
   *  session connects. Each exercise has a recommended view. */
  cameraGuidance: CameraGuidance;
};

export const EXERCISES: Exercise[] = [
  {
    slug: "pushup",
    name: "Push-Up",
    tagline: "The default. Start here.",
    primary: true,
    isWeighted: false,
    cameraGuidance: {
      view: "side",
      headline: "Place the camera on your side",
      detail: "Side view at roughly shoulder height, 2–3 m away. Your whole body from head to feet must be visible in profile.",
    },
  },
  {
    slug: "squat",
    name: "Squat",
    tagline: "Depth, torso, knees.",
    primary: false,
    isWeighted: true,
    cameraGuidance: {
      view: "front",
      headline: "Place the camera front or side",
      detail: "Front or side view at hip height, 2–3 m away. Either works — just keep your whole body from head to feet in frame through the full squat.",
    },
  },
  {
    slug: "lunge",
    name: "Lunge",
    tagline: "Front knee, back knee, upright torso.",
    primary: false,
    isWeighted: false,
    cameraGuidance: {
      view: "side",
      headline: "Place the camera on your side",
      detail: "Side view at hip height, 2–3 m away. The full lunge travel — front knee, back knee, torso — must stay in frame.",
    },
  },
  {
    slug: "deadlift",
    name: "Deadlift",
    tagline: "Hip hinge, flat back.",
    primary: false,
    isWeighted: true,
    cameraGuidance: {
      view: "front",
      headline: "Place the camera in front of you",
      detail: "Front view at hip height, 2–3 m away. Shoulders, hips, knees, and ankles must all be visible from setup to lockout.",
    },
  },
  {
    slug: "crunch",
    name: "V-Up Crunch",
    tagline: "Legs up, arms forward.",
    primary: false,
    isWeighted: false,
    cameraGuidance: {
      view: "side",
      headline: "Place the camera on your side",
      detail: "Side view at floor level, 2–3 m away. Full body from head to feet must be visible. Ensure enough space to see legs and arms fully extended.",
    },
  },
  {
    slug: "lateral_raise",
    name: "Lateral Raise",
    tagline: "Elbows lead, no shrug.",
    primary: false,
    isWeighted: false,
    cameraGuidance: {
      view: "front",
      headline: "Place the camera in front of you",
      detail: "Front view at waist height, 2–3 m away. Full upper body including both arms and hips must be visible.",
    },
  },
  {
    slug: "side_plank",
    name: "Side Plank",
    tagline: "Hips stacked, hold it.",
    primary: false,
    isWeighted: false,
    cameraGuidance: {
      view: "side",
      headline: "Place the camera facing the front of your body",
      detail: "Side view at floor level, 2–3 m away. Full body from head to feet must be visible in a straight line.",
    },
  },
  {
    slug: "pullup",
    name: "Pull-Up",
    tagline: "Chin above bar, no swing.",
    primary: false,
    isWeighted: false,
    cameraGuidance: {
      view: "front",
      headline: "Place the camera in front of you",
      detail: "Front view, far enough back that the bar and your full arm extension stay in frame at the bottom of every rep.",
    },
  },
  {
    slug: "plank",
    name: "Plank",
    tagline: "One line, hold it.",
    primary: false,
    isWeighted: false,
    cameraGuidance: {
      view: "side",
      headline: "Place the camera on your side",
      detail: "Side view at body height. Shoulders, hips, and ankles must all be visible to track the body line.",
    },
  },
  {
    slug: "bicep_curl",
    name: "Bicep Curl",
    tagline: "Upper arm still.",
    primary: false,
    isWeighted: false,
    cameraGuidance: {
      view: "side",
      headline: "Place the camera on your side",
      detail: "Side view at chest height. Shoulder, elbow, and wrist on the curling arm must stay visible through full extension.",
    },
  },
  {
    slug: "tricep_dip",
    name: "Tricep Dip",
    tagline: "Elbow depth, chest up.",
    primary: false,
    isWeighted: false,
    cameraGuidance: {
      view: "side",
      headline: "Place the camera on your side",
      detail: "Side view, far enough back to see full dip travel. Shoulders, elbows, and hips must stay in frame.",
    },
  },
];

const SLUG_SET = new Set<string>(EXERCISES.map((e) => e.slug));

export function isExerciseSlug(value: string | undefined): value is ExerciseSlug {
  return !!value && SLUG_SET.has(value);
}

export function exerciseBySlug(slug: ExerciseSlug): Exercise {
  return EXERCISES.find((e) => e.slug === slug) ?? EXERCISES[0];
}
