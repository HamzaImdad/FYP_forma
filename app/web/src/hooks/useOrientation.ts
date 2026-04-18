import { useEffect, useState } from "react";

import type { PhoneOrientation } from "../types/exercise";

/** Tracks the current viewport orientation via matchMedia. Updates on rotation. */
export function useOrientation(): PhoneOrientation {
  const [orientation, setOrientation] = useState<PhoneOrientation>(() =>
    typeof window !== "undefined" && window.matchMedia("(orientation: landscape)").matches
      ? "landscape"
      : "portrait",
  );

  useEffect(() => {
    if (typeof window === "undefined") return;
    const mql = window.matchMedia("(orientation: landscape)");
    const handler = (event: MediaQueryListEvent) => {
      setOrientation(event.matches ? "landscape" : "portrait");
    };
    if (mql.addEventListener) {
      mql.addEventListener("change", handler);
    } else {
      mql.addListener(handler);
    }
    setOrientation(mql.matches ? "landscape" : "portrait");
    return () => {
      if (mql.removeEventListener) {
        mql.removeEventListener("change", handler);
      } else {
        mql.removeListener(handler);
      }
    };
  }, []);

  return orientation;
}
