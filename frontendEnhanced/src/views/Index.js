import { useState, useEffect } from "react";
import axios from "axios";
import {
  Container, Spinner, Table, Input, Label, Row, Col
} from "reactstrap";
import Header from "components/Headers/Header.js";

export default function JobRecommendations() {
  const userId = localStorage.getItem("user_id");

  const [docs,          setDocs]   = useState([]);      
  const [selectedDocId, setDocId]  = useState("");     
  const [jobs,          setJobs]   = useState([]);
  const [loading,       setLoad]   = useState(false);
  const [error,         setError]  = useState(null);

  useEffect(() => {
    if (!userId) return;

    (async () => {
      try {
        const { data } = await axios.get(
          "http://localhost:5000/api/cvs",
          { params: { user_id: userId }, timeout: 10000 }
        );
        setDocs(data);
        if (data.length) setDocId(data[0].id);
      } catch (err) {
        console.error(err);
        setError("Couldn’t load your documents");
      }
    })();
  }, [userId]);

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

  return (
    <>
      <Header />
      <Container className="mt--7" fluid>

        
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
