// src/pages/UploadPage.js
import React, { useState } from "react";

const UploadPage = () => {
  const [selectedFile, setSelectedFile] = useState(null);

  const handleFileChange = (e) => {
    setSelectedFile(e.target.files[0]);
  };

  const handleFileUpload = () => {
    if (selectedFile) {
      alert(`File uploaded: ${selectedFile.name}`);
    } else {
      alert("Please select a file first!");
    }
  };

  return (
    <div className="upload-container">
      <h2>Upload Your Resume</h2>
      <input type="file" onChange={handleFileChange} accept=".pdf, .docx" />
      <button onClick={handleFileUpload}>Upload</button>
      {selectedFile && <p>Selected file: {selectedFile.name}</p>}
    </div>
  );
};

export default UploadPage;
