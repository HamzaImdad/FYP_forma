import { useEffect, useState } from "react";

/** True when the current device is a phone — a coarse-pointer viewport under
 *  1024px wide. Lets orientation-gating skip laptops / desktops entirely. */
export function usePhone(): boolean {
  const [isPhone, setIsPhone] = useState<boolean>(() => {
    if (typeof window === "undefined") return false;
    return window.matchMedia("(pointer: coarse) and (max-width: 1024px)").matches;
  });

  useEffect(() => {
    if (typeof window === "undefined") return;
    const mql = window.matchMedia("(pointer: coarse) and (max-width: 1024px)");
    const handler = (event: MediaQueryListEvent) => setIsPhone(event.matches);
    if (mql.addEventListener) {
      mql.addEventListener("change", handler);
    } else {
      mql.addListener(handler);
    }
    setIsPhone(mql.matches);
    return () => {
      if (mql.removeEventListener) {
        mql.removeEventListener("change", handler);
      } else {
        mql.removeListener(handler);
      }
    };
  }, []);

  return isPhone;
}
