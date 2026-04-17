import { useEffect } from "react";
import { useLocation } from "react-router-dom";
import Lenis from "lenis";
import { gsap, ScrollTrigger, registerGsap } from "@/lib/gsap";

let lenisInstance: Lenis | null = null;

export function getLenis() {
  return lenisInstance;
}

export function SmoothScroll() {
  const { pathname, hash } = useLocation();

  useEffect(() => {
    registerGsap();
    const lenis = new Lenis({
      duration: 1.2,
      easing: (t: number) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
      smoothWheel: true,
    });
    lenisInstance = lenis;

    // Bridge Lenis ↔ ScrollTrigger so reveal animations and pinned sections
    // fire with the smooth-scrolled position, not the raw native scroll.
    lenis.on("scroll", ScrollTrigger.update);
    const tickerCallback = (time: number) => lenis.raf(time * 1000);
    gsap.ticker.add(tickerCallback);
    gsap.ticker.lagSmoothing(0);

    return () => {
      gsap.ticker.remove(tickerCallback);
      lenis.destroy();
      lenisInstance = null;
    };
  }, []);

  // Reset scroll on route change, jump to hash target if present
  useEffect(() => {
    const lenis = lenisInstance;
    if (!lenis) return;
    if (hash) {
      const target = document.querySelector(hash) as HTMLElement | null;
      if (target) {
        lenis.scrollTo(target, { offset: -80, immediate: false });
        return;
      }
    }
    lenis.scrollTo(0, { immediate: true });
  }, [pathname, hash]);

  return null;
}
