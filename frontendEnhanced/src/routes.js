import Index from "views/Index.js";
import Profile from "views/examples/Profile.js";
import Maps from "views/examples/Maps.js";
import Register from "views/examples/Register.js";
import Login from "views/examples/Login.js";
import Tables from "views/examples/Tables.js";
import Icons from "views/examples/Icons.js";
import Logout from "views/examples/Logout"
import ProtectedRoute from "components/Navbars/ProtectedRoutes";


var routes = [
  {
    path: "/index",
    name: "Job Search",
    icon: "ni ni-briefcase-24 text-primary",
    component: <ProtectedRoute element={<Index />} />,
    layout: "/admin",
  },
  {
    path: "/user-profile",
    name: "Profile",
    icon: "ni ni-single-02 text-yellow",
    component: <ProtectedRoute element={<Profile/>}/>,
    layout: "/admin",
  },
  {
    path: "/tables",
    name: "Documents",
    icon: "ni ni-archive-2 text-red",
    component: <ProtectedRoute element={<Tables/>}/>,
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
