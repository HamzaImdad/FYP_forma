// Session 4: PlansPage is now the real plan-creator UI.
// Split screen with the plan-architect chatbot + live preview + active plan.
// Protected by <ProtectedRoute> in App.tsx.

import { PlanShell } from "@/components/plans/PlanShell";

export function PlansPage() {
  return <PlanShell />;
}
