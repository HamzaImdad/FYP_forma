import { useState, type FormEvent } from "react";
import { Link, Navigate, useLocation, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { ApiError } from "../lib/api";
import {
  AuthShell,
  AuthField,
  AuthSubmit,
  AuthError,
} from "../components/auth/AuthShell";

const ERROR_COPY: Record<string, string> = {
  invalid_credentials: "That email or password didn't match. Try again.",
  unauthorized: "Session expired — log in again.",
};

type LocationState = { from?: { pathname?: string } } | null;

export function LoginPage() {
  const { login, user, loading } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const state = location.state as LocationState;
  const redirectTo = state?.from?.pathname ?? "/dashboard";

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  if (!loading && user) {
    return <Navigate to={redirectTo} replace />;
  }

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      await login(email.trim(), password);
      navigate(redirectTo, { replace: true });
    } catch (err) {
      const code = err instanceof ApiError ? err.code : undefined;
      setError(ERROR_COPY[code ?? ""] ?? "Something went wrong. Try again.");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <AuthShell
      kicker="Welcome back"
      title="LOG&nbsp;IN"
      tagline="Train with the eye of a coach"
      footer={
        <>
          New to FORMA?{" "}
          <Link
            to="/signup"
            className="not-italic uppercase tracking-[0.18em] text-[10px] text-[color:var(--color-gold)] hover:text-[color:var(--color-gold-hover)] transition-colors border-b border-[color:var(--color-gold)]/40 pb-0.5"
          >
            Sign up
          </Link>
        </>
      }
    >
      <form onSubmit={handleSubmit} noValidate>
        <AuthError message={error} />

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
          autoComplete="current-password"
          placeholder="••••••••"
          disabled={submitting}
        />

        <AuthSubmit
          label="Enter the session"
          loading={submitting}
          disabled={!email || !password}
        />
      </form>
    </AuthShell>
  );
}
