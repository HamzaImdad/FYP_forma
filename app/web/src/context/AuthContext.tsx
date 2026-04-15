import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { api, ApiError } from "../lib/api";

export type User = {
  id: number;
  email: string;
  display_name: string;
  avatar_url: string | null;
  height_cm: number | null;
  weight_kg: number | null;
  age: number | null;
  experience_level: "beginner" | "intermediate" | "advanced";
  training_goal: "strength" | "size" | "endurance" | "skill";
  coaching_tone: "gentle" | "neutral" | "drill_sergeant";
  created_at: string;
  last_login: string | null;
};

type AuthEnvelope = { user: User };

type AuthContextValue = {
  user: User | null;
  loading: boolean;
  signup: (email: string, password: string, displayName: string) => Promise<void>;
  login: (email: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  refresh: () => Promise<void>;
};

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    try {
      const { user } = await api<AuthEnvelope>("/api/auth/me");
      setUser(user);
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) {
        setUser(null);
      } else {
        // Network / server error — don't nuke the user; keep whatever we had.
        console.warn("auth refresh failed:", err);
      }
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const signup = useCallback(
    async (email: string, password: string, displayName: string) => {
      const { user } = await api<AuthEnvelope>("/api/auth/signup", {
        method: "POST",
        body: JSON.stringify({ email, password, display_name: displayName }),
      });
      setUser(user);
    },
    [],
  );

  const login = useCallback(async (email: string, password: string) => {
    const { user } = await api<AuthEnvelope>("/api/auth/login", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    });
    setUser(user);
  }, []);

  const logout = useCallback(async () => {
    try {
      await api("/api/auth/logout", { method: "POST" });
    } finally {
      setUser(null);
    }
  }, []);

  const value = useMemo<AuthContextValue>(
    () => ({ user, loading, signup, login, logout, refresh }),
    [user, loading, signup, login, logout, refresh],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within an AuthProvider");
  return ctx;
}
