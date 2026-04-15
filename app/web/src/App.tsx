import { Route, Routes, useLocation } from "react-router-dom";
import { Nav } from "./components/layout/Nav";
import { Footer } from "./components/layout/Footer";
import { SmoothScroll } from "./components/layout/SmoothScroll";
import { PageTransition } from "./components/layout/PageTransition";
import { ProtectedRoute } from "./components/auth/ProtectedRoute";
import { Home } from "./pages/Home";
import { Stub } from "./pages/Stub";
import { VoiceCoachingPage } from "./pages/VoiceCoachingPage";
import { ChatbotPage } from "./pages/ChatbotPage";
import { PlansPage } from "./pages/PlansPage";
import { MilestonesPage } from "./pages/MilestonesPage";
import { DeveloperModePage } from "./pages/DeveloperModePage";
import { AboutPage } from "./pages/AboutPage";
import { WorkoutPage } from "./pages/WorkoutPage";
import { LoginPage } from "./pages/LoginPage";
import { SignupPage } from "./pages/SignupPage";

export function App() {
  const { pathname } = useLocation();
  // Session, workout, and auth pages are fullscreen (no nav/footer)
  const fullscreen =
    pathname === "/session" ||
    pathname === "/login" ||
    pathname === "/signup" ||
    pathname.startsWith("/workout/");

  return (
    <>
      <SmoothScroll />
      {!fullscreen && <Nav />}
      <main className="relative z-[2]">
        <PageTransition>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route
              path="/exercises"
              element={
                <Stub
                  phase="Phase 03"
                  title="Exercises"
                  description="Ten exercises, one index. Arriving with the session-core port."
                />
              }
            />
            <Route
              path="/guide"
              element={
                <Stub
                  phase="Phase 03"
                  title="Camera Setup"
                  description="Per-exercise guide — framing, distance, lighting."
                />
              }
            />
            <Route
              path="/session"
              element={
                <Stub
                  phase="Phase 03"
                  title="Live Session"
                  description="Socket.IO video feed + real-time form score. Coming next."
                />
              }
            />
            <Route
              path="/report"
              element={
                <Stub
                  phase="Phase 03"
                  title="Session Report"
                  description="Per-rep form scores, common issues, set breakdown."
                />
              }
            />
            <Route
              path="/dashboard"
              element={
                <ProtectedRoute>
                  <Stub
                    phase="Phase 04"
                    title="Dashboard"
                    description="Form trend, push-up focus, session history, streak heatmap."
                  />
                </ProtectedRoute>
              }
            />
            <Route path="/login" element={<LoginPage />} />
            <Route path="/signup" element={<SignupPage />} />
            <Route path="/about" element={<AboutPage />} />
            <Route path="/voice-coaching" element={<VoiceCoachingPage />} />
            <Route
              path="/chatbot"
              element={
                <ProtectedRoute>
                  <ChatbotPage />
                </ProtectedRoute>
              }
            />
            <Route
              path="/plans"
              element={
                <ProtectedRoute>
                  <PlansPage />
                </ProtectedRoute>
              }
            />
            <Route
              path="/milestones"
              element={
                <ProtectedRoute>
                  <MilestonesPage />
                </ProtectedRoute>
              }
            />
            <Route path="/_dev" element={<DeveloperModePage />} />
            <Route
              path="/workout/:exercise"
              element={
                <ProtectedRoute>
                  <WorkoutPage />
                </ProtectedRoute>
              }
            />
            <Route
              path="*"
              element={
                <Stub
                  phase="404"
                  title="Not Found"
                  description="That route doesn't exist — yet."
                />
              }
            />
          </Routes>
        </PageTransition>
      </main>
      {!fullscreen && <Footer />}
    </>
  );
}
