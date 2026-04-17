import { Route, Routes, useLocation } from "react-router-dom";
import { Nav } from "./components/layout/Nav";
import { Footer } from "./components/layout/Footer";
import { SmoothScroll } from "./components/layout/SmoothScroll";
import { PageTransition } from "./components/layout/PageTransition";
import { ProtectedRoute } from "./components/auth/ProtectedRoute";
import { Home } from "./pages/Home";
import { Stub } from "./pages/Stub";
import { ExercisesPage } from "./pages/ExercisesPage";
import { VoiceCoachingPage } from "./pages/VoiceCoachingPage";
import { ChatbotPage } from "./pages/ChatbotPage";
import { PlansPage } from "./pages/PlansPage";
import { MilestonesPage } from "./pages/MilestonesPage";
import { DeveloperModePage } from "./pages/DeveloperModePage";
import { AboutPage } from "./pages/AboutPage";
import { HowItWorksPage } from "./pages/HowItWorksPage";
import { FeaturesPage } from "./pages/FeaturesPage";
import { WorkoutPage } from "./pages/WorkoutPage";
import { LoginPage } from "./pages/LoginPage";
import { SignupPage } from "./pages/SignupPage";
import { DashboardPage } from "./pages/DashboardPage";
import { ProfilePage } from "./pages/ProfilePage";
import { SessionDetailPanel } from "./components/dashboard/SessionDetailPanel";
import { CelebrationToast } from "./components/milestones/CelebrationToast";
import { PublicChatWidget } from "./components/chat/PublicChatWidget";
import { PersonalCoachPanel } from "./components/dashboard/PersonalCoachPanel";
import { useAuth } from "./context/AuthContext";

export function App() {
  const { pathname } = useLocation();
  const { user } = useAuth();
  // Workout pages are fullscreen (no nav/footer). Auth pages now show nav.
  const fullscreen = pathname.startsWith("/workout/");

  return (
    <>
      <SmoothScroll />
      {user && <CelebrationToast />}
      {!fullscreen && <Nav />}
      <main className="relative z-[2]">
        <PageTransition>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/exercises" element={<ExercisesPage />} />
            <Route
              path="/dashboard"
              element={
                <ProtectedRoute>
                  <DashboardPage />
                </ProtectedRoute>
              }
            />
            <Route
              path="/dashboard/session/:id"
              element={
                <ProtectedRoute>
                  <SessionDetailPanel />
                </ProtectedRoute>
              }
            />
            <Route path="/login" element={<LoginPage />} />
            <Route path="/signup" element={<SignupPage />} />
            <Route path="/about" element={<AboutPage />} />
            <Route path="/how-it-works" element={<HowItWorksPage />} />
            <Route path="/features" element={<FeaturesPage />} />
            <Route path="/voice-coaching" element={<VoiceCoachingPage />} />
            <Route path="/chatbot" element={<ChatbotPage />} />
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
            <Route
              path="/profile"
              element={
                <ProtectedRoute>
                  <ProfilePage />
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
      {!user && !fullscreen && pathname !== "/login" && <PublicChatWidget />}
      {user
        && !fullscreen
        && !pathname.startsWith("/plans")
        && pathname !== "/login"
        && <PersonalCoachPanel />}
    </>
  );
}
