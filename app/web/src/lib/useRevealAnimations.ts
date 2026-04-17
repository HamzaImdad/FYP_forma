import { useEffect, type RefObject } from "react";
import { useGSAP } from "@gsap/react";
import { gsap, registerGsap, ScrollTrigger } from "@/lib/gsap";

registerGsap();

export function useRevealAnimations(scope: RefObject<HTMLElement | null>) {
  useGSAP(
    () => {
      gsap.utils.toArray<HTMLElement>("[data-reveal]").forEach((el) => {
        gsap.from(el, {
          y: 24,
          opacity: 0,
          duration: 0.9,
          ease: "power3.out",
          immediateRender: false,
          scrollTrigger: {
            trigger: el,
            start: "top 88%",
            toggleActions: "play none none none",
          },
        });
      });
    },
    { scope },
  );

  useEffect(() => {
    const t = setTimeout(() => ScrollTrigger.refresh(), 400);
    return () => clearTimeout(t);
  }, []);
}
