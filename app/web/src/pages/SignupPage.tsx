import { Navigate } from "react-router-dom";

// Sign-up now lives inside LoginPage as a tab. Keep this route for external
// links that still point to /signup — redirect them to the merged page with
// the signup tab preselected.
export function SignupPage() {
  return <Navigate to="/login?tab=signup" replace />;
}
