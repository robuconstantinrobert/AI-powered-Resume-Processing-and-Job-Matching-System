// import React from "react";
// import { 
//   Badge,
//   Card,
//   CardHeader,
//   DropdownMenu,
//   DropdownItem,
//   UncontrolledDropdown,
//   DropdownToggle,
//   Media,
//   Progress,
//   Table,
//   Container,
//   Row,
//   Button, 
// } from "reactstrap";
// import Header from "components/Headers/Header.js";

// const ResumeJobMatching = () => {
//   return (
//     <>
//       <Header />
//       {/* Page content */}
//       <Container className="mt--7" fluid>
//         {/* Table */}
//         <Row>
//           <div className="col">
//             <Card className="shadow">
//               <CardHeader className="border-0">
//                 <h3 className="mb-0">Resume manager</h3>
//               </CardHeader>
//               <Table className="align-items-center table-flush" responsive>
//                 <thead className="thead-light">
//                   <tr>
//                     <th scope="col">Document</th>
//                     <th scope="col">Skills Match</th>
//                     <th scope="col">Job Recommendations</th>
//                     <th scope="col">Processing Status</th>
//                     <th scope="col" />
//                   </tr>
//                 </thead>
//                 <tbody>
//                   <tr>
//                     <th scope="row">
//                       <Media className="align-items-center">
//                         <Media>
//                           <span className="mb-0 text-sm">Resume_Jhon_Doe.pdf</span>
//                         </Media>
//                       </Media>
//                     </th>
//                     <td>
//                       <Badge color="success" className="badge-dot mr-4">
//                         <i className="bg-success" /> 85% Match
//                       </Badge>
//                     </td>
//                     <td>
//                       <ur>
//                         <li>Software Engineer</li>
//                         <li>Backend Developer</li>
//                       </ur>
//                     </td>
//                     <td>
//                       <div className="d-flex align-items-center">
//                         <span className="mr-2">Completed</span>
//                         <Progress max="100" value="100" barClassName="bg-success" />
//                       </div>
//                     </td>
//                     <td className="text-left">
//                       {/* Buttons with icons and hover text */}
//                       <Button 
//                         color="danger" 
//                         size="sm" 
//                         className="mr-2"
//                         //title="Delete"
//                       >
//                         <i className="fas fa-trash" /> 
//                       </Button>
//                       <Button 
//                         color="info" 
//                         size="sm" 
//                         className="mr-2"
//                         //title="Search Jobs"
//                       >
//                         <i className="fas fa-search" /> 
//                       </Button>
//                       <Button 
//                         color="warning" 
//                         size="sm" 
//                         className="mr-2"
//                         //title="Edit Document"
//                       >
//                         <i className="fas fa-edit" /> 
//                       </Button>

//                       <UncontrolledDropdown>
//                         <DropdownToggle
//                           className="btn-icon-only text-light"
//                           href="#pablo"
//                           role="button"
//                           size="sm"
//                           color=""
//                           onClick={(e) => e.preventDefault()}
//                         >
//                           <i className="fas fa-ellipsis-v" />
//                         </DropdownToggle>
//                         <DropdownMenu className="dropdown-menu-arrow" right>
//                           <DropdownItem href="#pablo" onClick={(e) => e.preventDefault()}>
//                             View Resume
//                           </DropdownItem>
//                           <DropdownItem href="#pablo" onClick={(e) => e.preventDefault()}>
//                             Edit Details
//                           </DropdownItem>
//                           <DropdownItem href="#pablo" onClick={(e) => e.preventDefault()}>
//                             Remove Document
//                           </DropdownItem>
//                         </DropdownMenu>
//                       </UncontrolledDropdown>
//                     </td>
//                   </tr>
//                 </tbody>
//               </Table>
//             </Card>
//           </div>
//         </Row>
//       </Container>
//     </>
//   );
// };

// export default ResumeJobMatching;



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

  const handleUpload = () => {
    if (selectedFile) {
      console.log("Uploading file:", selectedFile.name);
      // Implement upload logic here
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
                <Button color="primary" size="sm">Add New Resume</Button>
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
                <h4 className="mb-0">Upload New Resume</h4>
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


