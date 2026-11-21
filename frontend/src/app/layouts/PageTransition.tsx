import { AnimatePresence, motion } from 'framer-motion';
import type { PropsWithChildren } from 'react';
import { useLocation } from 'react-router-dom';

// Animation policy: page transitions only, minimal and fast. No
// parallax, no floating loops, no decorative motion.
export function PageTransition({ children }: PropsWithChildren) {
  const location = useLocation();

  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={location.pathname}
        initial={{ opacity: 0, y: 4 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0 }}
        transition={{ duration: 0.15, ease: 'easeOut' }}
      >
        {children}
      </motion.div>
    </AnimatePresence>
  );
}
