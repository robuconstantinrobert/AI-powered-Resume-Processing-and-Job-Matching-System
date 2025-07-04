import { useNavigate } from "react-router-dom";
import React, { useState, useEffect } from "react";
import { 
  Card,
  CardHeader,
  Media,
  Table,
  Container,
  Row,
  Button,
  CardBody,
  Spinner
} from "reactstrap";
import Header from "components/Headers/Header.js";
import EditModal from "./EditModal.js";

const ResumeJobMatching = () => {
  const navigate = useNavigate();
  const [selectedFile, setSelectedFile] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const [useEsco, setUseEsco] = useState(false);
  const [uploadStatus, setUploadStatus] = useState("");
  const [selectedEmb, setSelectedEmb] = useState("minilm");
  const [selectedModel, setSelectedModel] = useState("tinyllama");
  const [documents, setDocuments] = useState([]);
  const [editModalOpen, setEditModalOpen] = useState(false);
  const [currentDocId, setCurrentDocId] = useState(null);
  const [searchLoading,  setSearchLoading]  = useState(false);
  const [dashRefresh, setDashRefresh] = useState(0);
  

  useEffect(() => {
    fetchDocuments();
  }, []);

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
  };

  const handleDragOver = (event) => {
    event.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = () => {
    setDragOver(false);
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setDragOver(false);
    if (event.dataTransfer.files.length) {
      setSelectedFile(event.dataTransfer.files[0]);
    }
  };

  const fetchDocuments = async () => {
    const userId = localStorage.getItem("user_id");
    if (!userId) return;

    try {
      const response = await fetch(`http://localhost:5000/api/documents/${userId}`);
      if (!response.ok) throw new Error("Failed to fetch documents");
      const data = await response.json();
      setDocuments(data);
    } catch (err) {
      console.error("Error fetching documents:", err);
    }
  };


  const handleUpload = async () => {
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("emb", selectedEmb);
    formData.append("model", selectedModel);
    formData.append("user_id", localStorage.getItem("user_id"));
    formData.append("file_name", selectedFile.name);


    if (useEsco) {
      formData.append("top", 3);
    }


    const endpoint = useEsco
      ? "http://localhost:5000/api/process_cv_with_esco"
      : "http://localhost:5000/api/process_cv";

    try {
      setUploadStatus("Uploading...");
      const response = await fetch(endpoint, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error("Upload failed");

      const data = await response.json();
      console.log("Upload result:", data);
      setUploadStatus("Upload successful!");
      setDashRefresh(k => k + 1);
    } catch (err) {
      console.error(err);
      setUploadStatus("Upload failed. Please try again.");
    }

    fetchDocuments();

  };

  const handleDelete = async (documentId) => {
    try {
      const response = await fetch(`http://localhost:5000/api/documents/${documentId}`, {
        method: "DELETE",
      });

      if (!response.ok) throw new Error("Failed to delete document");

      setDocuments((prevDocs) => prevDocs.filter((doc) => doc._id !== documentId));
      setDashRefresh(k => k + 1);
    } catch (err) {
      console.error("Error deleting document:", err);
    }
  };

  const openEditModal = (docId) => {
    setCurrentDocId(docId);
    setEditModalOpen(true);
  };

  const handleSearch = async (docId) => {
    const userId = localStorage.getItem("user_id");
    if (!docId || !userId) return;

    try {
      setSearchLoading(true);
      const res = await fetch("http://localhost:5000/api/linkedin/search-jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ _id: docId, user_id: userId })
      });
      if (!res.ok) {
        const { error } = await res.json();
        throw new Error(error || "Search failed");
      }
      const { count, results } = await res.json();
      console.log(`Found ${count} jobs for doc ${docId}`, results);

      navigate(`/jobs?docId=${docId}&userId=${userId}`);

    } catch (err) {
      console.error("LinkedIn search error:", err);
      setSearchLoading(false);
    }
  };

  


  return (
    <>
      <Header refreshKey={dashRefresh} />
      {searchLoading && (
        <div
          className="position-fixed w-100 h-100 d-flex justify-content-center align-items-center"
          style={{ top: 0, left: 0, background: "rgba(255,255,255,0.6)", zIndex: 1050 }}
        >
          <Spinner style={{ width: "3rem", height: "3rem" }} />
        </div>
      )}
      <Container className="mt--7" fluid>
        <Row>
          <div className="col">
            <Card className="shadow">
              <CardHeader className="border-0 d-flex justify-content-between align-items-center flex-wrap">
                <h3 className="mb-0">Resume Manager</h3>

                <div className="d-flex align-items-center">
                  {/* toggle button for enhanced processing */}
                  <Button
                    color={useEsco ? "info" : "secondary"}
                    size="sm"
                    className="mr-4 px-3"
                    onClick={() => setUseEsco(!useEsco)}
                  >
                    Enhanced processing
                  </Button>

                  {/* Embedding selector */}
                  <div className="d-flex align-items-center mr-4">
                    <label className="text-muted mb-0 mr-2">Embedding:</label>
                    <select
                      className="form-control form-control-alternative form-control-sm"
                      value={selectedEmb}
                      onChange={e => setSelectedEmb(e.target.value)}
                    >
                      <option value="minilm">MiniLM</option>
                      <option value="mpnet">MPNet</option>
                      <option value="gtr">GTR T5</option>
                    </select>
                  </div>

                  {/* Model selector */}
                  <div className="d-flex align-items-center">
                    <label className="text-muted mb-0 mr-2">Chat Completion:</label>
                    <select
                      className="form-control form-control-alternative form-control-sm"
                      value={selectedModel}
                      onChange={e => setSelectedModel(e.target.value)}
                    >
                      <option value="tinyllama">TinyLlama</option>
                      <option value="zephyr">Zephyr</option>
                      <option value="qwen">Qwen</option>
                    </select>
                  </div>
                </div>
              </CardHeader>

              <Table className="align-items-center table-flush" responsive>
                <thead className="thead-light">
                  <tr>
                    <th scope="col">Document</th>
                    <th scope="col">Skills Match</th>
                    <th scope="col">Job Recommendations</th>
                    <th scope="col">Seniority</th>
                    <th scope="col">Processing Status</th>
                    <th scope="col" />
                  </tr>
                </thead>
                <tbody>
                  {documents.length === 0 ? (
                    <tr>
                      <td colSpan="5" className="text-center text-muted">No documents found</td>
                    </tr>
                  ) : (
                    documents.map((doc) => (
                      <tr key={doc._id}>
                        <th scope="row">
                          <Media className="align-items-center">
                            <Media>
                              <span className="mb-0 text-sm">{doc.file_name || "Unnamed Document"}</span>
                            </Media>
                          </Media>
                        </th>
                        <td>
                          <ul className="mb-0">
                            {(doc?.date_extrase?.competente || []).map((title, idx) => (
                              <li key={idx}>{title}</li>
                            ))}
                          </ul>
                        </td>
                        <td>
                          <ul className="mb-0">
                            {(doc?.date_extrase?.job_titles || []).map((title, idx) => (
                              <li key={idx}>{title}</li>
                            ))}
                          </ul>
                        </td>
                        <td>
                          <span>{doc?.date_extrase?.work_experience || "N/A"}</span>
                        </td>
                        <td>
                          <div className="d-flex align-items-center">
                            <span className="mr-2">Processed</span>
                          </div>
                        </td>
                        <td className="text-left">
                          <Button
                            color="danger"
                            size="sm"
                            className="mr-2"
                            onClick={() => {
                              handleDelete(doc._id);
                            }}
                          >
                            <i className="fas fa-trash" />
                          </Button>
                          <Button
                            color="info"
                            size="sm"
                            className="mr-2"
                            onClick={() => handleSearch(doc._id)}
                          >
                            <i className="fas fa-search" />
                          </Button>
                          <Button
                            color="warning"
                            size="sm"
                            className="mr-2"
                            onClick={() => openEditModal(doc._id)}
                          >
                            <i className="fas fa-edit" />
                          </Button>
                        </td>
                      </tr>
                    ))  
                  )}
                </tbody>

              </Table>
            </Card>
          </div>
        </Row>
        
        <EditModal
          isOpen={editModalOpen}
          toggle={() => setEditModalOpen(false)}
          documentId={currentDocId}
          onUpdate={() => {
            fetchDocuments();
            setEditModalOpen(false);
          }}
        />
        
        <Row className="mt-5 justify-content-center">
          <div className="col-lg-6">
            <Card className="shadow">
              <CardHeader className="bg-white border-0 text-center">
                <h4 className="mb-0">Upload Resume</h4>
                {uploadStatus && (
                  <p className={`mt-2 ${uploadStatus.includes("failed") ? "text-danger" : "text-success"}`}>
                    {uploadStatus}
                  </p>
                )}
              </CardHeader>
              <CardBody className="text-center">
                <div 
                  className={`upload-zone p-5 border ${dragOver ? "border-primary" : "border-secondary"}`} 
                  onDragOver={handleDragOver} 
                  onDragLeave={handleDragLeave} 
                  onDrop={handleDrop} 
                  onClick={() => document.getElementById("fileInput").click()}
                  style={{ cursor: "pointer" }}
                >
                  <p className="text-muted">Drag & Drop your file here or click to upload</p>
                  {selectedFile && <p className="text-primary">Selected File: {selectedFile.name}</p>}
                </div>
                <input 
                  type="file" 
                  id="fileInput" 
                  className="d-none" 
                  onChange={handleFileChange} 
                />
                <Button color="primary" className="mt-3 w-100" onClick={handleUpload} disabled={!selectedFile}>
                  Upload Resume
                </Button>
              </CardBody>
            </Card>
          </div>
        </Row>
      </Container>
    </>
  );
};

export default ResumeJobMatching;
