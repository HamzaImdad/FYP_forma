import { useEffect, useRef, useState } from "react";
import { NavLink, Link, useLocation, useNavigate } from "react-router-dom";
import { ChevronDown, Menu, X } from "lucide-react";
import { EXERCISES, type Exercise } from "../../types/exercise";
import { useAuth } from "../../context/AuthContext";

type NavItem = { to: string; label: string };

const DEFAULT_WEIGHT_KG = 60;
const weightStorageKey = (slug: string) => `forma:last_weight:${slug}`;

function readLastWeight(slug: string): number {
  try {
    const raw = window.localStorage.getItem(weightStorageKey(slug));
    if (raw) {
      const n = parseFloat(raw);
      if (Number.isFinite(n) && n > 0) return n;
    }
  } catch {
    // ignore (SSR / disabled storage)
  }
  return DEFAULT_WEIGHT_KG;
}

function writeLastWeight(slug: string, weight: number): void {
  try {
    window.localStorage.setItem(weightStorageKey(slug), String(weight));
  } catch {
    // ignore
  }
}

const PRIMARY: NavItem[] = [
  { to: "/", label: "Home" },
  { to: "/exercises", label: "Exercises" },
  { to: "/dashboard", label: "Dashboard" },
];

const MORE: NavItem[] = [
  { to: "/voice-coaching", label: "Voice Coaching" },
  { to: "/chatbot", label: "AI Chatbot" },
  { to: "/plans", label: "Plans & Goals" },
  { to: "/milestones", label: "Milestones" },
];

export function Nav() {
  const [moreOpen, setMoreOpen] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [pickerOpen, setPickerOpen] = useState(false);
  const [weightStep, setWeightStep] = useState<Exercise | null>(null);
  const [weightInput, setWeightInput] = useState<string>("");
  const [scrolled, setScrolled] = useState(false);
  const [userMenuOpen, setUserMenuOpen] = useState(false);
  const moreRef = useRef<HTMLLIElement>(null);
  const userMenuRef = useRef<HTMLLIElement>(null);
  const { pathname } = useLocation();
  const navigate = useNavigate();
  const { user, logout } = useAuth();

  useEffect(() => {
    setMoreOpen(false);
    setMobileOpen(false);
    setPickerOpen(false);
    setWeightStep(null);
    setUserMenuOpen(false);
  }, [pathname]);

  useEffect(() => {
    if (!userMenuOpen) return;
    const onClick = (e: MouseEvent) => {
      if (!userMenuRef.current?.contains(e.target as Node)) {
        setUserMenuOpen(false);
      }
    };
    document.addEventListener("mousedown", onClick);
    return () => document.removeEventListener("mousedown", onClick);
  }, [userMenuOpen]);

  const handleLogout = async () => {
    setUserMenuOpen(false);
    await logout();
    navigate("/", { replace: true });
  };

  const userInitial = user?.display_name?.trim().charAt(0).toUpperCase() ?? "";

  useEffect(() => {
    if (!pickerOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        if (weightStep) {
          setWeightStep(null);
        } else {
          setPickerOpen(false);
        }
      }
    };
    document.addEventListener("keydown", onKey);
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = prev;
    };
  }, [pickerOpen, weightStep]);

  const closePicker = () => {
    setPickerOpen(false);
    setWeightStep(null);
  };

  const pickExercise = (ex: Exercise) => {
    if (ex.isWeighted) {
      setWeightInput(String(readLastWeight(ex.slug)));
      setWeightStep(ex);
    } else {
      closePicker();
      navigate(`/workout/${ex.slug}`);
    }
  };

  const confirmWeight = () => {
    if (!weightStep) return;
    const parsed = parseFloat(weightInput);
    if (!Number.isFinite(parsed) || parsed <= 0) return;
    writeLastWeight(weightStep.slug, parsed);
    const slug = weightStep.slug;
    closePicker();
    navigate(`/workout/${slug}?weight=${parsed}`);
  };

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 8);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    if (!moreOpen) return;
    const onClick = (e: MouseEvent) => {
      if (!moreRef.current?.contains(e.target as Node)) setMoreOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setMoreOpen(false);
    };
    document.addEventListener("mousedown", onClick);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onClick);
      document.removeEventListener("keydown", onKey);
    };
  }, [moreOpen]);

  const onDark = !scrolled;
  return (
    <header
      data-on-dark={onDark ? "true" : "false"}
      className={
        "fixed inset-x-0 top-0 z-[500] transition-all duration-300 " +
        (scrolled
          ? "backdrop-blur-md bg-[color:var(--color-page)]/85 border-b border-[color:var(--rule)]"
          : "bg-transparent border-b border-transparent")
      }
    >
      <div className="mx-auto max-w-[1440px] px-6 md:px-10 h-[72px] flex items-center justify-between">
        <Link
          to="/"
          className={
            "font-[family-name:var(--font-display)] text-2xl tracking-[0.24em] transition-colors " +
            (onDark
              ? "text-[color:var(--color-ink-on-dark)] hover:text-[color:var(--color-gold-soft)]"
              : "text-[color:var(--color-ink)] hover:text-[color:var(--color-gold)]")
          }
          aria-label="FORMA — Train With Form"
        >
          F
          <em
            className={
              "font-[family-name:var(--font-serif)] italic " +
              (onDark
                ? "text-[color:var(--color-gold-soft)]"
                : "text-[color:var(--color-gold)]")
            }
          >
            O
          </em>
          RMA
        </Link>

        <nav aria-label="Primary" className="hidden md:block">
          <ul className="flex items-center gap-1">
            {PRIMARY.map((item) => (
              <li key={item.to}>
                <NavLinkItem {...item} onDark={onDark} />
              </li>
            ))}

            <li ref={moreRef} className="relative">
              <button
                type="button"
                onClick={() => setMoreOpen((v) => !v)}
                aria-haspopup="true"
                aria-expanded={moreOpen}
                className={
                  "inline-flex items-center gap-1.5 px-4 py-2 text-sm tracking-[0.1em] uppercase transition-colors " +
                  (onDark
                    ? "text-[color:var(--color-ink-on-dark-2)] hover:text-[color:var(--color-ink-on-dark)]"
                    : "text-[color:var(--color-ink-2)] hover:text-[color:var(--color-ink)]")
                }
              >
                More
                <ChevronDown
                  size={14}
                  className={
                    "transition-transform duration-200 " +
                    (moreOpen ? "rotate-180" : "rotate-0")
                  }
                />
              </button>
              <div
                className={
                  "absolute right-0 top-full mt-2 min-w-[260px] origin-top-right rounded-[6px] border border-[color:var(--rule)] bg-[color:var(--color-raised)] shadow-[0_24px_64px_rgba(26,26,26,0.11)] transition-all duration-200 " +
                  (moreOpen
                    ? "opacity-100 translate-y-0 pointer-events-auto"
                    : "opacity-0 -translate-y-1 pointer-events-none")
                }
              >
                <ul className="py-2">
                  {MORE.map((item) => (
                    <li key={item.to}>
                      <NavLink
                        to={item.to}
                        className={({ isActive }) =>
                          "block px-5 py-2.5 text-sm transition-colors " +
                          (isActive
                            ? "text-[color:var(--color-gold)]"
                            : "text-[color:var(--color-ink-2)] hover:text-[color:var(--color-ink)] hover:bg-[color:var(--color-sunken)]")
                        }
                      >
                        {item.label}
                      </NavLink>
                    </li>
                  ))}
                </ul>
              </div>
            </li>

            <li>
              <NavLinkItem to="/about" label="About" onDark={onDark} />
            </li>

            <li className="ml-3">
              <button
                type="button"
                onClick={() => setPickerOpen(true)}
                className={
                  "inline-flex items-center gap-2 px-5 py-2.5 text-xs uppercase tracking-[0.14em] font-medium rounded-[2px] transition-colors " +
                  (onDark
                    ? "bg-[color:var(--color-ink-on-dark)] text-[color:var(--color-ink)] hover:bg-[color:var(--color-gold-soft)]"
                    : "bg-[color:var(--color-ink)] text-[color:var(--color-ink-on-dark)] hover:bg-[color:var(--color-orange)]")
                }
              >
                Start Workout
              </button>
            </li>

            {user ? (
              <li className="ml-2 relative" ref={userMenuRef}>
                <button
                  type="button"
                  onClick={() => setUserMenuOpen((v) => !v)}
                  className="flex items-center justify-center w-9 h-9 rounded-full text-[0.85rem] font-[family-name:var(--font-display)] tracking-wider bg-[color:var(--color-gold)] text-[color:var(--color-page)] hover:bg-[color:var(--color-gold-hover)] transition-colors"
                  aria-label={`Account menu — ${user.display_name}`}
                  aria-expanded={userMenuOpen}
                >
                  {userInitial}
                </button>
                <div
                  className={
                    "absolute right-0 top-full mt-2 min-w-[220px] bg-[color:var(--color-page)] border border-[color:var(--rule)] shadow-[0_16px_40px_rgba(26,26,26,0.18)] rounded-[2px] transition-all duration-200 " +
                    (userMenuOpen
                      ? "opacity-100 translate-y-0 pointer-events-auto"
                      : "opacity-0 -translate-y-1 pointer-events-none")
                  }
                >
                  <div className="px-5 py-4 border-b border-[color:var(--rule)]">
                    <div className="text-[10px] uppercase tracking-[0.22em] text-[color:var(--color-ink-2)]">
                      Signed in
                    </div>
                    <div className="mt-1 text-sm text-[color:var(--color-ink)] truncate">
                      {user.display_name}
                    </div>
                    <div className="text-xs text-[color:var(--color-ink-2)] truncate">
                      {user.email}
                    </div>
                  </div>
                  <ul className="py-1">
                    <li>
                      <NavLink
                        to="/dashboard"
                        className="block px-5 py-2.5 text-sm text-[color:var(--color-ink-2)] hover:text-[color:var(--color-ink)] hover:bg-[color:var(--color-sunken)] transition-colors"
                      >
                        Dashboard
                      </NavLink>
                    </li>
                    <li>
                      <button
                        type="button"
                        onClick={handleLogout}
                        className="block w-full text-left px-5 py-2.5 text-sm text-[color:var(--color-ink-2)] hover:text-[color:var(--color-bad)] hover:bg-[color:var(--color-sunken)] transition-colors"
                      >
                        Log out
                      </button>
                    </li>
                  </ul>
                </div>
              </li>
            ) : (
              <>
                <li className="ml-2">
                  <Link
                    to="/login"
                    className={
                      "inline-flex items-center px-4 py-2.5 text-xs uppercase tracking-[0.14em] transition-colors " +
                      (onDark
                        ? "text-[color:var(--color-ink-on-dark)] hover:text-[color:var(--color-gold-soft)]"
                        : "text-[color:var(--color-ink-2)] hover:text-[color:var(--color-ink)]")
                    }
                  >
                    Log in
                  </Link>
                </li>
                <li>
                  <Link
                    to="/signup"
                    className="inline-flex items-center px-5 py-2.5 text-xs uppercase tracking-[0.14em] font-medium rounded-[2px] bg-[color:var(--color-gold)] text-[color:var(--color-page)] hover:bg-[color:var(--color-gold-hover)] transition-colors"
                  >
                    Sign up
                  </Link>
                </li>
              </>
            )}
          </ul>
        </nav>

        <button
          type="button"
          className={
            "md:hidden p-2 " +
            (onDark
              ? "text-[color:var(--color-ink-on-dark)]"
              : "text-[color:var(--color-ink)]")
          }
          onClick={() => setMobileOpen((v) => !v)}
          aria-label={mobileOpen ? "Close menu" : "Open menu"}
        >
          {mobileOpen ? <X size={22} /> : <Menu size={22} />}
        </button>
      </div>

      <div
        className={
          "md:hidden overflow-hidden border-t border-[color:var(--rule)] bg-[color:var(--color-page)] transition-[max-height] duration-300 ease-[var(--ease-out-editorial)] " +
          (mobileOpen ? "max-h-[480px]" : "max-h-0")
        }
      >
        <ul className="px-6 py-4 space-y-1">
          {[...PRIMARY, { to: "/about", label: "About" }, ...MORE].map((item) => (
            <li key={item.to}>
              <NavLink
                to={item.to}
                className={({ isActive }) =>
                  "block py-2.5 text-base " +
                  (isActive
                    ? "text-[color:var(--color-gold)]"
                    : "text-[color:var(--color-ink-2)]")
                }
              >
                {item.label}
              </NavLink>
            </li>
          ))}
          <li className="pt-3">
            <button
              type="button"
              onClick={() => setPickerOpen(true)}
              className="block w-full text-center px-5 py-3 bg-[color:var(--color-ink)] text-[color:var(--color-ink-on-dark)] text-xs uppercase tracking-[0.14em] rounded-[2px]"
            >
              Start Workout
            </button>
          </li>
          {user ? (
            <li className="pt-2">
              <button
                type="button"
                onClick={handleLogout}
                className="block w-full text-center px-5 py-3 border border-[color:var(--rule)] text-[color:var(--color-ink-2)] text-xs uppercase tracking-[0.14em] rounded-[2px]"
              >
                Log out · {user.display_name}
              </button>
            </li>
          ) : (
            <li className="pt-2 grid grid-cols-2 gap-2">
              <Link
                to="/login"
                className="block text-center px-5 py-3 border border-[color:var(--rule)] text-[color:var(--color-ink-2)] text-xs uppercase tracking-[0.14em] rounded-[2px]"
              >
                Log in
              </Link>
              <Link
                to="/signup"
                className="block text-center px-5 py-3 bg-[color:var(--color-gold)] text-[color:var(--color-page)] text-xs uppercase tracking-[0.14em] rounded-[2px]"
              >
                Sign up
              </Link>
            </li>
          )}
        </ul>
      </div>

      {pickerOpen && (
        <div
          className="fixed inset-0 z-[1000] bg-[color:var(--color-ink)]/70 backdrop-blur-sm flex items-start justify-center p-6 overflow-y-auto"
          onClick={closePicker}
        >
          <div
            className="relative mt-24 mb-12 w-full max-w-3xl bg-[color:var(--color-page)] border border-[color:var(--rule)] shadow-[0_32px_80px_rgba(26,26,26,0.28)] rounded-[4px]"
            onClick={(e) => e.stopPropagation()}
            role="dialog"
            aria-modal="true"
            aria-label={weightStep ? "Enter weight" : "Choose an exercise"}
          >
            <button
              type="button"
              onClick={closePicker}
              className="absolute top-4 right-4 p-2 text-[color:var(--color-ink-2)] hover:text-[color:var(--color-ink)]"
              aria-label="Close"
            >
              <X size={18} />
            </button>

            {weightStep ? (
              <div className="px-8 pt-10 pb-10">
                <div className="text-[10px] uppercase tracking-[0.24em] text-[color:var(--color-gold)] mb-3">
                  {weightStep.name}
                </div>
                <h2 className="font-[family-name:var(--font-display)] text-4xl md:text-5xl text-[color:var(--color-ink)] tracking-[0.02em]">
                  How much are you lifting?
                </h2>
                <p className="mt-3 font-[family-name:var(--font-serif)] italic text-lg text-[color:var(--color-ink-2)]">
                  Enter today's working weight. We'll log it with your session.
                </p>

                <div className="mt-10 flex items-end gap-4">
                  <label className="flex-1">
                    <span className="block text-[10px] uppercase tracking-[0.24em] text-[color:var(--color-ink-2)] mb-2">
                      Weight
                    </span>
                    <div className="flex items-baseline gap-3 border-b-2 border-[color:var(--color-ink)]/60 focus-within:border-[color:var(--color-gold)] transition-colors">
                      <input
                        type="number"
                        inputMode="decimal"
                        min={0}
                        step={0.5}
                        autoFocus
                        value={weightInput}
                        onChange={(e) => setWeightInput(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === "Enter") confirmWeight();
                        }}
                        className="flex-1 bg-transparent font-[family-name:var(--font-display)] text-5xl md:text-6xl tabular-nums text-[color:var(--color-ink)] focus:outline-none py-2 tracking-[0.02em]"
                      />
                      <span className="font-[family-name:var(--font-display)] text-2xl text-[color:var(--color-ink-2)] tracking-[0.1em]">
                        KG
                      </span>
                    </div>
                  </label>
                </div>

                <div className="mt-10 flex items-center justify-between gap-4">
                  <button
                    type="button"
                    onClick={() => setWeightStep(null)}
                    className="inline-flex items-center gap-2 px-5 py-2.5 border border-[color:var(--color-ink)]/20 text-[10px] uppercase tracking-[0.18em] text-[color:var(--color-ink-2)] hover:text-[color:var(--color-ink)] hover:border-[color:var(--color-ink)]/40 transition-colors"
                  >
                    ← Back
                  </button>
                  <button
                    type="button"
                    onClick={confirmWeight}
                    disabled={!(parseFloat(weightInput) > 0)}
                    className="inline-flex items-center gap-2 px-8 py-3 bg-[color:var(--color-ink)] text-[color:var(--color-ink-on-dark)] text-xs uppercase tracking-[0.14em] font-medium rounded-[2px] hover:bg-[color:var(--color-orange)] disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                  >
                    Start Session →
                  </button>
                </div>
              </div>
            ) : (
              <>
                <div className="px-8 pt-10 pb-6">
                  <div className="text-[10px] uppercase tracking-[0.24em] text-[color:var(--color-gold)] mb-3">
                    Start Workout
                  </div>
                  <h2 className="font-[family-name:var(--font-display)] text-4xl md:text-5xl text-[color:var(--color-ink)] tracking-[0.02em]">
                    Choose an exercise
                  </h2>
                  <p className="mt-3 font-[family-name:var(--font-serif)] italic text-lg text-[color:var(--color-ink-2)]">
                    Train with form. Push-ups are the default.
                  </p>
                </div>
                <ul className="grid grid-cols-1 sm:grid-cols-2 gap-px bg-[color:var(--rule)] border-t border-[color:var(--rule)]">
                  {EXERCISES.map((ex) => (
                    <li key={ex.slug}>
                      <button
                        type="button"
                        onClick={() => pickExercise(ex)}
                        className="w-full text-left px-6 py-5 bg-[color:var(--color-page)] hover:bg-[color:var(--color-sunken)] transition-colors group"
                      >
                        <div className="flex items-baseline justify-between gap-3">
                          <span className="font-[family-name:var(--font-display)] text-2xl text-[color:var(--color-ink)] tracking-[0.04em]">
                            {ex.name}
                          </span>
                          <div className="flex items-center gap-2">
                            {ex.isWeighted && (
                              <span className="text-[9px] uppercase tracking-[0.2em] text-[color:var(--color-ink-2)] border border-[color:var(--color-ink-2)]/30 px-1.5 py-0.5">
                                Weighted
                              </span>
                            )}
                            {ex.primary && (
                              <span className="text-[9px] uppercase tracking-[0.2em] text-[color:var(--color-gold)]">
                                Default
                              </span>
                            )}
                          </div>
                        </div>
                        <div className="mt-1 text-sm text-[color:var(--color-ink-2)] group-hover:text-[color:var(--color-ink)]">
                          {ex.tagline}
                        </div>
                      </button>
                    </li>
                  ))}
                </ul>
              </>
            )}
          </div>
        </div>
      )}
    </header>
  );
}

function NavLinkItem({ to, label, onDark }: NavItem & { onDark: boolean }) {
  return (
    <NavLink
      to={to}
      end={to === "/"}
      className={({ isActive }) => {
        const base = "inline-flex px-4 py-2 text-sm uppercase tracking-[0.1em] transition-colors";
        if (onDark) {
          return (
            base +
            " " +
            (isActive
              ? "text-[color:var(--color-ink-on-dark)]"
              : "text-[color:var(--color-ink-on-dark-2)] hover:text-[color:var(--color-ink-on-dark)]")
          );
        }
        return (
          base +
          " " +
          (isActive
            ? "text-[color:var(--color-ink)] [text-shadow:0_1px_0_rgba(184,134,74,0.18)]"
            : "text-[color:var(--color-ink-2)] hover:text-[color:var(--color-ink)]")
        );
      }}
    >
      {label}
    </NavLink>
  );
}
