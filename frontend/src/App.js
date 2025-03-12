import React from "react";
import { BrowserRouter as Router, Route, Switch } from "react-router-dom";
import Navbar from "./components/Navbar";
import HeroSection from "./components/HeroSection";
import Features from "./components/Features";
import HomePage from "./pages/HomePage";
import UploadPage from "./pages/UploadPage";
import "./App.css";

function App() {
  return (
    <Router>
      <Navbar />
      <Switch>
        <Route path="/" exact component={HomePage} />
        <Route path="/upload" component={UploadPage} />
      </Switch>
    </Router>
  );
}

export default App;
