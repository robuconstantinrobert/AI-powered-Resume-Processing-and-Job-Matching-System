import { useState, useEffect } from "react";
import axios from "axios";
import {
  Container,
  Spinner,
  Table,
  Input,
  Label,
  Row,
  Col
} from "reactstrap";
import Header from "components/Headers/Header.js";

export default function JobRecommendations() {
  const userId = localStorage.getItem("user_id");

  const [docs,          setDocs]      = useState([]);
  const [selectedDocId, setDocId]     = useState("");
  const [jobs,          setJobs]      = useState([]);
  const [loading,       setLoad]      = useState(false);
  const [error,         setError]     = useState(null);
  const [deleting,      setDeleting]  = useState(false);
  const [dashRefresh, setDashRefresh] = useState(0);

  useEffect(() => {
    if (!userId) return;

    (async () => {
      try {
        const { data } = await axios.get(
          "http://localhost:5000/api/cvs",
          { params: { user_id: userId }, timeout: 10_000 }
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
            timeout: 10_000
          }
        );

        const prepared = Array.isArray(data)
          ? data.map(j => ({ applied_status: false, ...j }))
          : [];

        setJobs(prepared);
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

  const handleApply = async (job) => {
    window.open(job.url, "_blank", "noopener,noreferrer");

    try {
      await axios.put(
        `http://localhost:5000/api/jobs/${job._id}/apply`,
        { user_id: userId }
      );

      setJobs(prev =>
        prev.map(j =>
          j._id === job._id ? { ...j, applied_status: true } : j
        )
      );
      setDashRefresh(k => k+1);
    } catch (err) {
      console.error(err);
      alert("Couldn’t mark job as applied. Try again?");
    }
  };

  const handleDeleteApplied = async () => {
    if (jobs.every(j => !j.applied_status)) return; 


    try {
      setDeleting(true);
      await axios.delete("http://localhost:5000/api/jobs/cleanup", {
        params : { doc_id: selectedDocId, user_id: userId },
        timeout: 10_000
      });

      setJobs(prev => prev.filter(j => !j.applied_status));
      setDashRefresh(k => k + 1);
    } catch (err) {
      console.error(err);
      setError("Couldn’t delete applied jobs. Try again?");
    } finally {
      setDeleting(false);
    }
  };

  return (
    <>
      <Header refreshKey={dashRefresh} />

      <Container className="mt--7" fluid>
        <Row className="mb-4 align-items-end">
          <Col md="4">
            <Label for="cvSelect">Choose Document</Label>
            <Input
              id="cvSelect"
              type="select"
              value={selectedDocId}
              onChange={e => setDocId(e.target.value)}
            >
              {docs.map(doc => (
                <option key={doc.id} value={doc.id}>{doc.name}</option>
              ))}
            </Input>
          </Col>

          <Col className="text-right">
            <button
              className="btn btn-danger"
              onClick={handleDeleteApplied}
              disabled={deleting || jobs.every(j => !j.applied_status)}
            >
              {deleting ? "Deleting…" : "Delete applied jobs"}
            </button>
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
                  <th style={{ width: 10 }}>Apply</th>
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
                    <tr
                      key={job._id}
                      className={job.applied_status ? "table-success" : ""}
                    >
                      <td>{job.title}</td>
                      <td>{job.company}</td>
                      <td>{job.location}</td>
                      <td>
                        <button
                          className="btn btn-sm btn-primary"
                          onClick={() => handleApply(job)}
                          disabled={job.applied_status}
                        >
                          {job.applied_status ? "Applied ✓" : "Apply"}
                        </button>
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
