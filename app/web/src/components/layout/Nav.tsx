import { useEffect, useRef, useState } from "react";
import { NavLink, Link, useLocation, useNavigate } from "react-router-dom";
import { Menu, X } from "lucide-react";
import { useAuth } from "../../context/AuthContext";

type NavItem = { to: string; label: string; gated?: boolean };

// `gated: true` = hidden when logged-out. Route is still behind ProtectedRoute
// in App.tsx — this flag only controls nav visibility so unauthenticated
// visitors don't click into a silent redirect.
const NAV_ITEMS: NavItem[] = [
  { to: "/about", label: "About" },
  { to: "/exercises", label: "Exercises" },
  { to: "/how-it-works", label: "How It Works" },
  { to: "/features", label: "Features" },
  { to: "/dashboard", label: "Dashboard", gated: true },
  { to: "/plans", label: "Plans & Goals", gated: true },
  { to: "/milestones", label: "Milestones", gated: true },
];

export function Nav() {
  const [menuOpen, setMenuOpen] = useState(false);
  const [userMenuOpen, setUserMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);
  const userMenuRef = useRef<HTMLDivElement>(null);
  const { pathname } = useLocation();
  const navigate = useNavigate();
  const { user, logout } = useAuth();

  const visibleNav = NAV_ITEMS.filter((i) => !i.gated || user);

  // Close on route change
  useEffect(() => {
    setMenuOpen(false);
    setUserMenuOpen(false);
  }, [pathname]);

  // Close on escape
  useEffect(() => {
    if (!menuOpen && !userMenuOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setMenuOpen(false);
        setUserMenuOpen(false);
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [menuOpen, userMenuOpen]);

  // Close on outside click
  useEffect(() => {
    if (!menuOpen && !userMenuOpen) return;
    const onClick = (e: MouseEvent) => {
      const target = e.target as Node;
      if (menuOpen && menuRef.current && !menuRef.current.contains(target)) {
        setMenuOpen(false);
      }
      if (
        userMenuOpen &&
        userMenuRef.current &&
        !userMenuRef.current.contains(target)
      ) {
        setUserMenuOpen(false);
      }
    };
    document.addEventListener("mousedown", onClick);
    return () => document.removeEventListener("mousedown", onClick);
  }, [menuOpen, userMenuOpen]);

  const handleLogout = async () => {
    setUserMenuOpen(false);
    await logout();
    navigate("/", { replace: true });
  };

  const userInitial = user?.display_name?.trim().charAt(0).toUpperCase() ?? "";

  return (
    <header className="fixed inset-x-0 top-0 z-[500] bg-[color:var(--color-page)]/95 backdrop-blur-md border-b border-[color:var(--rule)]">
      <div className="mx-auto max-w-[1440px] px-6 md:px-10 h-[72px] flex items-center justify-between gap-6">
        {/* Logo */}
        <Link
          to="/"
          className="font-[family-name:var(--font-display)] text-2xl tracking-[0.24em] text-[color:var(--color-ink)] hover:text-[color:var(--color-gold)] transition-colors shrink-0"
          aria-label="FORMA — Train With Form"
        >
          F
          <em className="font-[family-name:var(--font-serif)] italic text-[color:var(--color-gold)]">
            O
          </em>
          RMA
        </Link>

        {/* Inline navbar links — desktop only. Mobile falls back to the menu button. */}
        <nav
          aria-label="Primary"
          className="hidden lg:flex items-center gap-8 flex-1 justify-center"
        >
          {visibleNav.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === "/"}
              className={({ isActive }) =>
                "relative text-[11px] uppercase tracking-[0.2em] transition-colors " +
                (isActive
                  ? "text-[color:var(--color-ink)] after:absolute after:left-0 after:right-0 after:-bottom-2 after:h-[2px] after:bg-[color:var(--color-gold)]"
                  : "text-[color:var(--color-ink-2)] hover:text-[color:var(--color-ink)]")
              }
            >
              {item.label}
            </NavLink>
          ))}
        </nav>

        {/* Right side: CTA + menu + user */}
        <div className="flex items-center gap-2 md:gap-3 shrink-0">
          {/* Start Workout CTA (logged in) / Sign In (logged out) */}
          {user ? (
            <Link
              to="/exercises"
              className="hidden sm:inline-flex items-center px-5 py-2.5 text-xs uppercase tracking-[0.14em] font-medium rounded-[2px] bg-[color:var(--color-gold)] text-[color:var(--color-page)] hover:bg-[color:var(--color-gold-hover)] transition-colors"
            >
              Start Workout
            </Link>
          ) : (
            <Link
              to="/login"
              className="hidden sm:inline-flex items-center px-5 py-2.5 text-xs uppercase tracking-[0.14em] font-medium rounded-[2px] bg-[color:var(--color-gold)] text-[color:var(--color-page)] hover:bg-[color:var(--color-gold-hover)] transition-colors"
            >
              Sign In
            </Link>
          )}

          {/* User avatar (logged in) */}
          {user && (
            <div className="relative" ref={userMenuRef}>
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
                  "absolute right-0 top-full mt-2 min-w-[220px] bg-[color:var(--color-raised)] border border-[rgba(174,231,16,0.08)] shadow-[0_16px_40px_rgba(0,0,0,0.5)] rounded-[2px] transition-all duration-200 " +
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
                      onClick={() => setUserMenuOpen(false)}
                      className="block px-5 py-2.5 text-sm text-[color:var(--color-ink-2)] hover:text-[color:var(--color-ink)] hover:bg-[color:var(--color-sunken)] transition-colors"
                    >
                      Dashboard
                    </NavLink>
                  </li>
                  <li>
                    <NavLink
                      to="/profile"
                      onClick={() => setUserMenuOpen(false)}
                      className="block px-5 py-2.5 text-sm text-[color:var(--color-ink-2)] hover:text-[color:var(--color-ink)] hover:bg-[color:var(--color-sunken)] transition-colors"
                    >
                      Profile
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
            </div>
          )}

          {/* Menu button — mobile only (< lg). Desktop uses the inline nav above. */}
          <div className="relative lg:hidden" ref={menuRef}>
            <button
              type="button"
              onClick={() => setMenuOpen((v) => !v)}
              className="flex items-center gap-2 px-3 py-2.5 text-xs uppercase tracking-[0.14em] text-[color:var(--color-ink)] hover:text-[color:var(--color-gold)] transition-colors"
              aria-label={menuOpen ? "Close menu" : "Open menu"}
              aria-expanded={menuOpen}
            >
              {menuOpen ? <X size={20} /> : <Menu size={20} />}
              <span className="hidden sm:inline">Menu</span>
            </button>

            {/* Dropdown panel */}
            <div
              className={
                "absolute right-0 top-full mt-2 w-[280px] bg-[color:var(--color-raised)] border border-[rgba(174,231,16,0.08)] shadow-[0_24px_64px_rgba(0,0,0,0.55)] rounded-[2px] transition-all duration-200 origin-top-right " +
                (menuOpen
                  ? "opacity-100 translate-y-0 scale-100 pointer-events-auto"
                  : "opacity-0 -translate-y-2 scale-[0.98] pointer-events-none")
              }
              role="menu"
              aria-label="Primary navigation"
            >
              <div className="px-5 pt-5 pb-3 border-b border-[color:var(--rule)]">
                <span className="text-[9px] uppercase tracking-[0.24em] text-[color:var(--color-ink-2)]">
                  Navigate
                </span>
              </div>
              <ul className="py-2">
                {visibleNav.map((item) => (
                  <li key={item.to}>
                    <NavLink
                      to={item.to}
                      end={item.to === "/"}
                      className={({ isActive }) =>
                        "block px-5 py-3 text-sm uppercase tracking-[0.12em] transition-colors " +
                        (isActive
                          ? "text-[color:var(--color-gold)] bg-[color:var(--color-sunken)]"
                          : "text-[color:var(--color-ink-2)] hover:text-[color:var(--color-ink)] hover:bg-[color:var(--color-sunken)]")
                      }
                      role="menuitem"
                    >
                      {item.label}
                    </NavLink>
                  </li>
                ))}
              </ul>
              {/* Auth footer inside menu for small screens */}
              <div className="sm:hidden border-t border-[color:var(--rule)] p-3 space-y-2">
                {user ? (
                  <Link
                    to="/exercises"
                    className="block text-center px-5 py-3 bg-[color:var(--color-gold)] text-[color:var(--color-page)] text-xs uppercase tracking-[0.14em] rounded-[2px]"
                  >
                    Start Workout
                  </Link>
                ) : (
                  <Link
                    to="/login"
                    className="block text-center px-5 py-3 bg-[color:var(--color-gold)] text-[color:var(--color-page)] text-xs uppercase tracking-[0.14em] rounded-[2px]"
                  >
                    Sign In
                  </Link>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}
