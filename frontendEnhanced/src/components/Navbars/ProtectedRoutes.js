// src/components/ProtectedRoute.js
import { Navigate } from "react-router-dom";
import { isAuthenticated } from "layouts/Auth";

const ProtectedRoute = ({ element }) => {
  return isAuthenticated() ? element : <Navigate to="/auth/login" replace />;
};

export default ProtectedRoute;
