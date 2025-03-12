// src/components/HeroSection.js
import React from "react";

const HeroSection = () => {
  return (
    <section className="hero-section">
      <div className="hero-content">
        <h1>Welcome to the Resume Matcher</h1>
        <p>Upload your resume and match it with the best job opportunities!</p>
        <a href="/upload" className="btn">Upload Your Resume</a>
      </div>
    </section>
  );
};

export default HeroSection;
