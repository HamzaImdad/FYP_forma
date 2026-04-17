import { useState, type FormEvent } from "react";
import {
  Navigate,
  useLocation,
  useNavigate,
  useSearchParams,
} from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { ApiError } from "../lib/api";
import {
  AuthShell,
  AuthField,
  AuthSubmit,
  AuthError,
} from "../components/auth/AuthShell";

const ERROR_COPY_LOGIN: Record<string, string> = {
  invalid_credentials: "That email or password didn't match. Try again.",
  unauthorized: "Session expired — log in again.",
};

const ERROR_COPY_SIGNUP: Record<string, string> = {
  invalid_email: "That doesn't look like a valid email.",
  password_too_short: "Password needs at least 8 characters.",
  display_name_required: "Tell us what to call you.",
  email_exists: "An account with that email already exists.",
};

type LocationState = { from?: { pathname?: string } } | null;
type Tab = "signin" | "signup";

export function LoginPage() {
  const { login, signup, user, loading } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const [searchParams, setSearchParams] = useSearchParams();

  const state = location.state as LocationState;
  const redirectTo = state?.from?.pathname ?? "/dashboard";

  const tabParam = searchParams.get("tab");
  const tab: Tab = tabParam === "signup" ? "signup" : "signin";

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  if (!loading && user) {
    return <Navigate to={redirectTo} replace />;
  }

  const switchTab = (next: Tab) => {
    setError(null);
    const nextParams = new URLSearchParams(searchParams);
    if (next === "signup") nextParams.set("tab", "signup");
    else nextParams.delete("tab");
    setSearchParams(nextParams, { replace: true });
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError(null);

    if (tab === "signup" && password.length < 8) {
      setError(ERROR_COPY_SIGNUP.password_too_short);
      return;
    }

    setSubmitting(true);
    try {
      if (tab === "signup") {
        await signup(email.trim(), password, displayName.trim());
      } else {
        await login(email.trim(), password);
      }
      navigate(redirectTo, { replace: true });
    } catch (err) {
      const code = err instanceof ApiError ? err.code : undefined;
      const copy = tab === "signup" ? ERROR_COPY_SIGNUP : ERROR_COPY_LOGIN;
      setError(copy[code ?? ""] ?? "Something went wrong. Try again.");
    } finally {
      setSubmitting(false);
    }
  };

  const header =
    tab === "signup"
      ? { kicker: "Begin the discipline", title: "SIGN\u00A0UP", tagline: "Every great athlete keeps a log" }
      : { kicker: "Welcome back", title: "LOG\u00A0IN", tagline: "Train with the eye of a coach" };

  const submitLabel = tab === "signup" ? "Open your training log" : "Enter the session";
  const submitDisabled =
    tab === "signup" ? !email || !password || !displayName : !email || !password;

  return (
    <AuthShell>
      {/* Tab bar */}
      <div
        role="tablist"
        aria-label="Authentication"
        className="flex items-stretch border-b border-[color:var(--color-ink)]/12 mb-10"
      >
        <TabButton active={tab === "signin"} onClick={() => switchTab("signin")}>
          Sign&nbsp;In
        </TabButton>
        <TabButton active={tab === "signup"} onClick={() => switchTab("signup")}>
          Create&nbsp;Account
        </TabButton>
      </div>

      {/* Header */}
      <div>
        <span className="block text-[10px] uppercase tracking-[0.32em] text-[color:var(--color-gold)]">
          {header.kicker}
        </span>
        <h1
          className="mt-4 font-[family-name:var(--font-display)] text-[clamp(2.6rem,6vw,3.8rem)] leading-[0.88] text-[color:var(--color-ink)] tracking-tight"
          dangerouslySetInnerHTML={{ __html: header.title }}
        />
        <p className="mt-3 font-[family-name:var(--font-serif)] italic text-[1.1rem] text-[color:var(--color-ink-2)]">
          {header.tagline}
        </p>
      </div>

      {/* Form */}
      <form onSubmit={handleSubmit} noValidate className="mt-10">
        <AuthError message={error} />

        {tab === "signup" && (
          <AuthField
            label="Name"
            name="display_name"
            value={displayName}
            onChange={setDisplayName}
            autoComplete="name"
            placeholder="Alex Rivera"
            disabled={submitting}
          />
        )}

        <AuthField
          label="Email"
          name="email"
          type="email"
          value={email}
          onChange={setEmail}
          autoComplete="email"
          placeholder="you@forma.app"
          disabled={submitting}
        />

        <AuthField
          label="Password"
          name="password"
          type="password"
          value={password}
          onChange={setPassword}
          autoComplete={tab === "signup" ? "new-password" : "current-password"}
          placeholder={tab === "signup" ? "minimum 8 characters" : "••••••••"}
          disabled={submitting}
        />

        <AuthSubmit label={submitLabel} loading={submitting} disabled={submitDisabled} />
      </form>

      {/* Cross-tab footer */}
      <div className="mt-10 font-[family-name:var(--font-serif)] italic text-[0.95rem] text-[color:var(--color-ink-2)]">
        {tab === "signup" ? (
          <>
            Already training?{" "}
            <button
              type="button"
              onClick={() => switchTab("signin")}
              className="not-italic uppercase tracking-[0.18em] text-[10px] text-[color:var(--color-gold)] hover:text-[color:var(--color-gold-hover)] transition-colors border-b border-[color:var(--color-gold)]/40 pb-0.5"
            >
              Sign in
            </button>
          </>
        ) : (
          <>
            New to FORMA?{" "}
            <button
              type="button"
              onClick={() => switchTab("signup")}
              className="not-italic uppercase tracking-[0.18em] text-[10px] text-[color:var(--color-gold)] hover:text-[color:var(--color-gold-hover)] transition-colors border-b border-[color:var(--color-gold)]/40 pb-0.5"
            >
              Create an account
            </button>
          </>
        )}
      </div>
    </AuthShell>
  );
}

function TabButton({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      role="tab"
      aria-selected={active}
      onClick={onClick}
      className={
        "flex-1 px-3 py-4 text-[11px] uppercase tracking-[0.22em] font-[family-name:var(--font-sans)] transition-colors " +
        (active
          ? "text-[color:var(--color-ink)] border-b-2 border-[color:var(--color-gold)] -mb-px"
          : "text-[color:var(--color-ink-2)] hover:text-[color:var(--color-ink)] border-b-2 border-transparent -mb-px")
      }
    >
      {children}
    </button>
  );
}
