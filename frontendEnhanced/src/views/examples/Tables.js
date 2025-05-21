import React, { useState } from "react";
import { 
  Badge,
  Card,
  CardHeader,
  DropdownMenu,
  DropdownItem,
  UncontrolledDropdown,
  DropdownToggle,
  Media,
  Progress,
  Table,
  Container,
  Row,
  Button,
  CardBody
} from "reactstrap";
import Header from "components/Headers/Header.js";

const ResumeJobMatching = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const [useEsco, setUseEsco] = useState(false);
  const [uploadStatus, setUploadStatus] = useState("");
  const [selectedEmb, setSelectedEmb] = useState("minilm");
  const [selectedModel, setSelectedModel] = useState("tinyllama");


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

  const handleUpload = async () => {
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("emb", selectedEmb);
    formData.append("model", selectedModel);
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
    } catch (err) {
      console.error(err);
      setUploadStatus("Upload failed. Please try again.");
    }
  };


  return (
    <>
      <Header />
      {/* Page content */}
      <Container className="mt--7" fluid>
        {/* Table */}
        <Row>
          <div className="col">
            <Card className="shadow">
              <CardHeader className="border-0 d-flex justify-content-between align-items-center">
                <h3 className="mb-0">Resume Manager</h3>
                <div className="mt-3">
                  <label className="d-flex align-items-center justify-content-center">
                    <input
                      type="checkbox"
                      checked={useEsco}
                      onChange={() => setUseEsco(!useEsco)}
                      style={{ marginRight: "8px" }}
                    />
                    Enhanced processing
                  </label>
                </div>
                <div className="mt-2">
                  <label className="text-muted">Embedding:</label>
                  <select
                    className="form-control"
                    value={selectedEmb}
                    onChange={(e) => setSelectedEmb(e.target.value)}
                  >
                    <option value="minilm">MiniLM</option>
                    <option value="mpnet">MPNet</option>
                    <option value="gtr">GTR T5</option>
                  </select>
                </div>

                <div className="mt-2">
                  <label className="text-muted">Chat Completion:</label>
                  <select
                    className="form-control"
                    value={selectedModel}
                    onChange={(e) => setSelectedModel(e.target.value)}
                  >
                    <option value="tinyllama">TinyLlama</option>
                    <option value="zephyr">Zephyr</option>
                    <option value="qwen">Qwen</option>
                  </select>
                </div>
              </CardHeader>
              <Table className="align-items-center table-flush" responsive>
                <thead className="thead-light">
                  <tr>
                    <th scope="col">Document</th>
                    <th scope="col">Skills Match</th>
                    <th scope="col">Job Recommendations</th>
                    <th scope="col">Processing Status</th>
                    <th scope="col" />
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <th scope="row">
                      <Media className="align-items-center">
                        <Media>
                          <span className="mb-0 text-sm">Resume_Jhon_Doe.pdf</span>
                        </Media>
                      </Media>
                    </th>
                    <td>
                      <Badge color="success" className="badge-dot mr-4">
                        <i className="bg-success" /> 85% Match
                      </Badge>
                    </td>
                    <td>
                      <ul className="mb-0">
                        <li>Software Engineer</li>
                        <li>Backend Developer</li>
                      </ul>
                    </td>
                    <td>
                      <div className="d-flex align-items-center">
                        <span className="mr-2">Completed</span>
                        <Progress max="100" value="100" barClassName="bg-success" style={{ width: "120px" }} />
                      </div>
                    </td>
                    <td className="text-left">
                      <Button color="danger" size="sm" className="mr-2">
                        <i className="fas fa-trash" />
                      </Button>
                      <Button color="info" size="sm" className="mr-2">
                        <i className="fas fa-search" />
                      </Button>
                      <Button color="warning" size="sm" className="mr-2">
                        <i className="fas fa-edit" />
                      </Button>
                      <UncontrolledDropdown>
                        <DropdownToggle className="btn-icon-only text-light" size="sm" color="">
                          <i className="fas fa-ellipsis-v" />
                        </DropdownToggle>
                        <DropdownMenu className="dropdown-menu-arrow" right>
                          <DropdownItem href="#pablo">View Resume</DropdownItem>
                          <DropdownItem href="#pablo">Edit Details</DropdownItem>
                          <DropdownItem href="#pablo">Remove Document</DropdownItem>
                        </DropdownMenu>
                      </UncontrolledDropdown>
                    </td>
                  </tr>
                </tbody>
              </Table>
            </Card>
          </div>
        </Row>
        
        {/* Drag and Drop Upload Zone */}
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


