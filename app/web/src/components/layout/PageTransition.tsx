import { AnimatePresence, motion } from "framer-motion";
import { useLocation } from "react-router-dom";
import type { PropsWithChildren } from "react";

export function PageTransition({ children }: PropsWithChildren) {
  const { pathname } = useLocation();
  return (
    <AnimatePresence mode="wait" initial={false}>
      <motion.div
        key={pathname}
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -8 }}
        transition={{ duration: 0.4, ease: [0.2, 0.7, 0.1, 1] }}
      >
        {children}
      </motion.div>
    </AnimatePresence>
  );
}
