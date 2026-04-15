import { useState, type FormEvent } from "react";
import { Link, Navigate, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { ApiError } from "../lib/api";
import {
  AuthShell,
  AuthField,
  AuthSubmit,
  AuthError,
} from "../components/auth/AuthShell";

const ERROR_COPY: Record<string, string> = {
  invalid_email: "That doesn't look like a valid email.",
  password_too_short: "Password needs at least 8 characters.",
  display_name_required: "Tell us what to call you.",
  email_exists: "An account with that email already exists.",
};

export function SignupPage() {
  const { signup, user, loading } = useAuth();
  const navigate = useNavigate();

  const [displayName, setDisplayName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  if (!loading && user) {
    return <Navigate to="/dashboard" replace />;
  }

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError(null);

    if (password.length < 8) {
      setError(ERROR_COPY.password_too_short);
      return;
    }

    setSubmitting(true);
    try {
      await signup(email.trim(), password, displayName.trim());
      navigate("/dashboard", { replace: true });
    } catch (err) {
      const code = err instanceof ApiError ? err.code : undefined;
      setError(ERROR_COPY[code ?? ""] ?? "Something went wrong. Try again.");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <AuthShell
      kicker="Begin the discipline"
      title="SIGN&nbsp;UP"
      tagline="Every great athlete keeps a log"
      footer={
        <>
          Already training?{" "}
          <Link
            to="/login"
            className="not-italic uppercase tracking-[0.18em] text-[10px] text-[color:var(--color-gold)] hover:text-[color:var(--color-gold-hover)] transition-colors border-b border-[color:var(--color-gold)]/40 pb-0.5"
          >
            Log in
          </Link>
        </>
      }
    >
      <form onSubmit={handleSubmit} noValidate>
        <AuthError message={error} />

        <AuthField
          label="Name"
          name="display_name"
          value={displayName}
          onChange={setDisplayName}
          autoComplete="name"
          placeholder="Alex Rivera"
          disabled={submitting}
        />

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
          autoComplete="new-password"
          placeholder="minimum 8 characters"
          disabled={submitting}
        />

        <AuthSubmit
          label="Open your training log"
          loading={submitting}
          disabled={!email || !password || !displayName}
        />
      </form>
    </AuthShell>
  );
}
