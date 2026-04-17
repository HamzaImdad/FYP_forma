// PlanCreatorChat — thin wrapper that mounts ChatShell in plan mode with
// a sidebar-less layout and an onTurnDone callback so PlanShell can
// refetch the live plan draft after every assistant turn.

import type { MutableRefObject } from "react";
import { ChatShell } from "@/components/chat/ChatShell";

type Props = {
  onTurnDone?: (conversationId: number | null) => void;
  sendRef?: MutableRefObject<
    | { send: (text: string) => void; pending: boolean }
    | null
  >;
};

export function PlanCreatorChat({ onTurnDone, sendRef }: Props) {
  return (
    <ChatShell
      mode="plan"
      showSidebar={false}
      authed={true}
      title="PLAN ARCHITECT"
      tagline="Tell me how you train and I'll build an adaptive workout plan."
      inputPlaceholder="Days per week, time, injuries, goals…"
      onTurnDone={onTurnDone}
      sendRef={sendRef}
    />
  );
}
