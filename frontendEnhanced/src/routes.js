import Index from "views/Index.js";
import Profile from "views/examples/Profile.js";
import Maps from "views/examples/Maps.js";
import Register from "views/examples/Register.js";
import Login from "views/examples/Login.js";
import Tables from "views/examples/Tables.js";
import Icons from "views/examples/Icons.js";
import { Navigate } from "react-router-dom";
import { isAuthenticated } from "layouts/Auth";
import Logout from "views/examples/Logout"


// 🔐 Wrapper inline pentru protejarea componentelor
const protectedRoute = (Component) => {
  return isAuthenticated() ? Component : <Navigate to="/auth/login" replace />;
};

var routes = [
  {
    path: "/index",
    name: "Job Search",
    icon: "ni ni-briefcase-24 text-primary",
    component: protectedRoute(<Index />), // 🔒 protejat
    layout: "/admin",
  },
  {
    path: "/icons",
    name: "Icons",
    icon: "ni ni-planet text-blue",
    component: protectedRoute(<Icons />), // 🔒
    layout: "/admin",
  },
  {
    path: "/maps",
    name: "Maps",
    icon: "ni ni-pin-3 text-orange",
    component: protectedRoute(<Maps />), // 🔒
    layout: "/admin",
  },
  {
    path: "/user-profile",
    name: "Profile",
    icon: "ni ni-single-02 text-yellow",
    component: protectedRoute(<Profile />), // 🔒
    layout: "/admin",
  },
  {
    path: "/tables",
    name: "Documents",
    icon: "ni ni-archive-2 text-red",
    component: protectedRoute(<Tables />), // 🔒
    layout: "/admin",
  },
  {
    path: "/login",
    name: "Login",
    icon: "ni ni-key-25 text-info",
    component: <Login />, // ✅ Public
    layout: "/auth",
  },
  {
    path: "/register",
    name: "Register",
    icon: "ni ni-circle-08 text-green",
    component: <Register />, // ✅ Public
    layout: "/auth",
  },
  {
    path: "/logout",
    name: "Logout",
    icon: "ni ni-user-run text-danger",
    component: <Logout />, // 🔄 Redirecționează automat
    layout: "/auth",
  },
];

export default routes;
