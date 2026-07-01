import { useEffect, useRef } from "react";
import { gsap } from "gsap";

export function useWorkbenchMotion() {
  const rootRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    const root = rootRef.current;
    if (!root) return undefined;

    gsap.defaults({ duration: 0.34, ease: "power2.out" });
    const media = gsap.matchMedia();

    media.add(
      {
        reduceMotion: "(prefers-reduced-motion: reduce)",
        active: "(min-width: 1px)"
      },
      (context) => {
        const conditions = context.conditions as { reduceMotion?: boolean } | undefined;
        const sections = root.querySelectorAll(
          ".toolbar, .input-grid, .control-row, .advanced-prompts, .generation-settings"
        );

        if (conditions?.reduceMotion) {
          gsap.set(sections, { autoAlpha: 1, y: 0, clearProps: "all" });
          return undefined;
        }

        const tween = gsap.fromTo(
          sections,
          { y: 14, autoAlpha: 0 },
          {
            y: 0,
            autoAlpha: 1,
            duration: 0.38,
            stagger: 0.045,
            ease: "power2.out",
            clearProps: "transform,visibility,opacity"
          }
        );

        return () => {
          tween.kill();
        };
      }
    );

    return () => media.revert();
  }, []);

  return rootRef;
}
