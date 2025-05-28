// // // import { useState, useEffect } from "react";
// // // import { useLocation, useSearchParams } from "react-router-dom";
// // // import axios from "axios";
// // // import {
// // //   Card,
// // //   CardHeader,
// // //   CardBody,
// // //   Table,
// // //   Container,
// // //   Row,
// // //   Col,
// // //   Button
// // // } from "reactstrap";
// // // import Header from "components/Headers/Header.js";
// // // import { Spinner } from "reactstrap";

// // // function useQuery() {
// // //   return new URLSearchParams(useLocation().search);
// // // }

// // // const Index = () => {
// // //   const [jobs, setJobs]       = useState([]);
// // //   const [loading, setLoading] = useState(true);
// // //   const [error, setError]     = useState();
// // //   const [searchParams] = useSearchParams();
// // //   const docId   = searchParams.get("docId");
// // //   const userId  = searchParams.get("userId");


// // //     useEffect(() => {
// // //       if (!docId || !userId) return;

// // //       const fetchRecommendations = async () => {
// // //         try {
// // //           console.log("Fetching jobs for", { docId, userId });
// // //           // use full localhost URL to eliminate proxy mis‐routing
// // //           const resp = await axios.get(
// // //             "http://localhost:5000/api/jobs/recommendations",
// // //             {
// // //               params: { doc_id: docId, user_id: userId },
// // //               timeout: 10000,         // give it up to 10s
// // //             }
// // //           );
// // //           console.log("Raw GET response:", resp);
// // //           console.log("resp.data:", resp.data);

// // //           if (Array.isArray(resp.data)) {
// // //             setJobs(resp.data);
// // //           } else if (resp.data.error) {
// // //             throw new Error(resp.data.error);
// // //           } else {
// // //             console.warn("Unexpected shape:", resp.data);
// // //             setJobs([]);
// // //           }
// // //         } catch (err) {
// // //           console.error("Error in fetchRecommendations:", err);
// // //           setError(err.message || "Unknown error");
// // //           setJobs([]);  // stop spinner
// // //         }
// // //       };

// // //       fetchRecommendations();
// // //     }, [docId, userId]);

// // //     if (loading) {
// // //       return (
// // //         <>
// // //           <Header />
// // //           <Container className="mt--7 text-center">
// // //             <Spinner style={{ width: "3rem", height: "3rem" }} />
// // //           </Container>
// // //         </>
// // //       );
// // //     }

// // //   return (
// // //     <>
// // //       <Header />
// // //       <Container className="mt--7" fluid>
// // //         {error && <p className="text-danger">{error}</p>}

// // //         <Table responsive>
// // //           <thead>
// // //             <tr>
// // //               <th>Job Title</th>
// // //               <th>Company</th>
// // //               <th>Location</th>
// // //               <th>Apply</th>
// // //             </tr>
// // //           </thead>
// // //           <tbody>
// // //             {jobs.length === 0 ? (
// // //               <tr>
// // //                 <td colSpan="4" className="text-center">
// // //                   No jobs found
// // //                 </td>
// // //               </tr>
// // //             ) : (
// // //               jobs.map((job, i) => (
// // //                 <tr key={i}>
// // //                   <td>{job.title}</td>
// // //                   <td>{job.company}</td>
// // //                   <td>{job.location}</td>
// // //                   <td>
// // //                     <a
// // //                       href={job.url}
// // //                       target="_blank"
// // //                       rel="noopener noreferrer"
// // //                       className="btn btn-sm btn-primary"
// // //                     >
// // //                       Apply
// // //                     </a>
// // //                   </td>
// // //                 </tr>
// // //               ))
// // //             )}
// // //           </tbody>
// // //         </Table>
// // //       </Container>
// // //     </>
// // //   );
// // // };

// // // export default Index;

// // import { useState, useEffect } from "react";
// // import { useSearchParams } from "react-router-dom";
// // import axios from "axios";
// // import { Table, Container, Spinner } from "reactstrap";
// // import Header from "components/Headers/Header.js";

// // const Index = () => {
// //   const [jobs, setJobs]       = useState([]);       // NEVER null
// //   const [loading, setLoading] = useState(true);
// //   const [error, setError]     = useState(null);

// //   const [searchParams] = useSearchParams();
// //   const docId  = searchParams.get("doc_Id");
// //   const userId = searchParams.get("user_Id");

// //   useEffect(() => {
// //     // If we don't even have the query params, stop loading and bail
// //     if (!docId || !userId) {
// //       setLoading(false);
// //       return;
// //     }

// //     const fetchJobs = async () => {
// //       try {
// //         const resp = await axios.get(
// //           "http://localhost:5000/api/jobs/recommendations",
// //           { params: { doc_id: docId, user_id: userId } }
// //         );
// //         setJobs(Array.isArray(resp.data) ? resp.data : []);
// //       } catch (err) {
// //         console.error("Error fetching jobs:", err);
// //         setError(err.response?.data?.error || err.message);
// //       } finally {
// //         setLoading(false);
// //       }
// //     };

// //     fetchJobs();
// //   }, [docId, userId]);

// //   // 1) Loading state
// //   if (loading) {
// //     return (
// //       <>
// //         <Header />
// //         <Container className="mt--7 text-center">
// //           <Spinner style={{ width: "3rem", height: "3rem" }} />
// //         </Container>
// //       </>
// //     );
// //   }

// //   // 2) Loaded: safe to render table with jobs always an array
// //   return (
// //     <>
// //       <Header />
// //       <Container className="mt--7" fluid>
// //         {error && <p className="text-danger">{error}</p>}

// //         <Table responsive>
// //           <thead>
// //             <tr>
// //               <th>Job Title</th>
// //               <th>Company</th>
// //               <th>Location</th>
// //               <th>Apply</th>
// //             </tr>
// //           </thead>
// //           <tbody>
// //             {!(jobs?.length) ? (
// //               <tr>
// //                 <td colSpan="4" className="text-center">
// //                   No jobs found
// //                 </td>
// //               </tr>
// //             ) : (
// //               jobs.map((job, idx) => (
// //                 <tr key={idx}>
// //                   <td>{job.title}</td>
// //                   <td>{job.company}</td>
// //                   <td>{job.location}</td>
// //                   <td>
// //                     <a
// //                       href={job.url}
// //                       target="_blank"
// //                       rel="noopener noreferrer"
// //                       className="btn btn-sm btn-primary"
// //                     >
// //                       Apply
// //                     </a>
// //                   </td>
// //                 </tr>
// //               ))
// //             )}
// //           </tbody>
// //         </Table>
// //       </Container>
// //     </>
// //   );
// // };

// // export default Index;


// import { useState, useEffect } from "react";
// import { useSearchParams } from "react-router-dom";
// import axios from "axios";
// import { Table, Container, Spinner } from "reactstrap";
// import Header from "components/Headers/Header.js";

// export default function JobRecommendations() {
//   const [jobs, setJobs]       = useState([]);
//   const [loading, setLoading] = useState(true);
//   const [error, setError]     = useState(null);

//   const [searchParams] = useSearchParams();
//   const docId  = searchParams.get("doc_id");   // <-- fixed
//   const userId = searchParams.get("user_id");  // <-- fixed

//   useEffect(() => {
//     if (!docId || !userId) { setLoading(false); return; }

//     const fetchJobs = async () => {
//       setLoading(true);
//       try {
//         const { data } = await axios.get(
//           "http://localhost:5000/api/jobs/recommendations?doc_id={doc_id}&user_id={user_id}",
//           { params: { doc_id: docId, user_id: userId }, timeout: 10000 }
//         );
//         setJobs(Array.isArray(data) ? data : []);
//         setError(null);
//       } catch (err) {
//         console.error(err);
//         setError(err.response?.data?.error || err.message);
//         setJobs([]);
//       } finally {
//         setLoading(false);
//       }
//     };

//     fetchJobs();
//   }, [docId, userId]);

//   if (loading) {
//     return (
//       <>
//         <Header />
//         <Container className="mt--7 text-center">
//           <Spinner style={{ width: "3rem", height: "3rem" }} />
//         </Container>
//       </>
//     );
//   }

//   return (
//     <>
//       <Header />
//       <Container className="mt--7" fluid>
//         {error && <p className="text-danger">{error}</p>}

//         <Table responsive hover>
//           <thead>
//             <tr>
//               <th>Job Title</th>
//               <th>Company</th>
//               <th>Location</th>
//               <th style={{ width: 90 }}>Apply</th>
//             </tr>
//           </thead>
//           <tbody>
//             {jobs.length === 0 ? (
//               <tr>
//                 <td colSpan="4" className="text-center">
//                   No jobs found
//                 </td>
//               </tr>
//             ) : (
//               jobs.map(job => (
//                 <tr key={job._id /* backend made this a string */}>
//                   <td>{job.title}</td>
//                   <td>{job.company}</td>
//                   <td>{job.location}</td>
//                   <td>
//                     <a
//                       href={job.url}
//                       target="_blank"
//                       rel="noopener noreferrer"
//                       className="btn btn-sm btn-primary"
//                     >
//                       Apply
//                     </a>
//                   </td>
//                 </tr>
//               ))
//             )}
//           </tbody>
//         </Table>
//       </Container>
//     </>
//   );
// }


import { useState, useEffect } from "react";
import { useSearchParams } from "react-router-dom";
import axios from "axios";
import {
  Container, Spinner, Table, Input, Label, Row, Col
} from "reactstrap";
import Header from "components/Headers/Header.js";

export default function JobRecommendations() {
  /* ------------------------- url parameters ------------------------- */
  const [searchParams] = useSearchParams();
  const userId = localStorage.getItem("user_id");

  /* ---------------------------- state ------------------------------- */
  const [docs,          setDocs]   = useState([]);      // list of CVs
  const [selectedDocId, setDocId]  = useState("");      // the chosen one
  const [jobs,          setJobs]   = useState([]);
  const [loading,       setLoad]   = useState(false);
  const [error,         setError]  = useState(null);

  /* 1️⃣  load the user’s documents once -------------------------------- */
  useEffect(() => {
    if (!userId) return;

    (async () => {
      try {
        const { data } = await axios.get(
          "http://localhost:5000/api/cvs",
          { params: { user_id: userId }, timeout: 10000 }
        );
        setDocs(data);
        // optionally pre-select the first CV
        if (data.length) setDocId(data[0].id);
      } catch (err) {
        console.error(err);
        setError("Couldn’t load your documents");
      }
    })();
  }, [userId]);

  /* 2️⃣  every time selectedDocId changes → fetch recommendations ------- */
  useEffect(() => {
    if (!userId || !selectedDocId) return;

    (async () => {
      setLoad(true);
      try {
        const { data } = await axios.get(
          "http://localhost:5000/api/jobs/recommendations",
          {
            params : { doc_id: selectedDocId, user_id: userId },
            timeout: 10000
          }
        );
        setJobs(Array.isArray(data) ? data : []);
        setError(null);
      } catch (err) {
        console.error(err);
        setError(err.response?.data?.error || err.message);
        setJobs([]);
      } finally {
        setLoad(false);
      }
    })();
  }, [selectedDocId, userId]);

  /* --------------------------- render ------------------------------- */
  return (
    <>
      <Header />
      <Container className="mt--7" fluid>

        {/* pick the CV -------------------------------------------------- */}
        <Row className="mb-4">
          <Col md="4">
            <Label for="cvSelect">Choose a résumé / CV</Label>
            <Input
              id="cvSelect"
              type="select"
              value={selectedDocId}
              onChange={e => setDocId(e.target.value)}
            >
              {docs.map(doc => (
                <option key={doc.id} value={doc.id}>
                  {doc.name}
                </option>
              ))}
            </Input>
          </Col>
        </Row>

        {/* results ------------------------------------------------------ */}
        {loading ? (
          <div className="text-center">
            <Spinner style={{ width: "3rem", height: "3rem" }} />
          </div>
        ) : (
          <>
            {error && <p className="text-danger">{error}</p>}

            <Table responsive hover>
              <thead>
                <tr>
                  <th>Job Title</th>
                  <th>Company</th>
                  <th>Location</th>
                  <th style={{ width: 90 }}>Apply</th>
                </tr>
              </thead>
              <tbody>
                {jobs.length === 0 ? (
                  <tr>
                    <td colSpan="4" className="text-center">
                      No jobs found
                    </td>
                  </tr>
                ) : (
                  jobs.map(job => (
                    <tr key={job._id}>
                      <td>{job.title}</td>
                      <td>{job.company}</td>
                      <td>{job.location}</td>
                      <td>
                        <a
                          href={job.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="btn btn-sm btn-primary"
                        >
                          Apply
                        </a>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </Table>
          </>
        )}
      </Container>
    </>
  );
}
