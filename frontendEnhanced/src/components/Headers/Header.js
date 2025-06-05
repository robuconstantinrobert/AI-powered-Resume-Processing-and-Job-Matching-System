import { useEffect, useState } from "react";
import {
  Card, CardBody, CardTitle,
  Container, Row, Col, Spinner
} from "reactstrap";

const Header = ({ refreshKey = 0 }) => {          // ← NEW
  const userId = localStorage.getItem("user_id");
  const [stats, setStats] = useState(null);
  const [error, setError] = useState(null);

  /* refetch whenever refreshKey changes */
  useEffect(() => {
    if (!userId) return;

    const fetchStats = async () => {
      try {
        const res = await fetch(
          `http://localhost:5000/api/dashboard/stats?user_id=${userId}`
        );
        if (!res.ok) throw new Error("Failed to load stats");
        setStats(await res.json());
        setError(null);
      } catch (err) {
        console.error(err);
        setError("Stats unavailable");
      }
    };

    fetchStats();
  }, [userId, refreshKey]);                       // ← include key

  const StatCard = ({ color, icon, label, value }) => (
    <Col lg="6" xl="3">
      <Card className="card-stats mb-4 mb-xl-0">
        <CardBody>
          <Row>
            <div className="col">
              <CardTitle tag="h5" className="text-uppercase text-muted mb-0">
                {label}
              </CardTitle>
              <span className="h2 font-weight-bold mb-0">
                {value ?? <Spinner size="sm" />}
              </span>
            </div>
            <Col className="col-auto">
              <div className={`icon icon-shape bg-${color} text-white rounded-circle shadow`}>
                <i className={icon} />
              </div>
            </Col>
          </Row>
        </CardBody>
      </Card>
    </Col>
  );

  return (
    <div className="header bg-gradient-info pb-8 pt-5 pt-md-8">
      <Container fluid>
        <div className="header-body">
          <Row>
            <Col>
              <h2 className="text-white">Resume Job Matching</h2>
              <p className="text-white mt-3">
                Monitor and manage resumes and job matches
              </p>
            </Col>
          </Row>

          <Row className="mt-4">
            <StatCard
              color="info"
              icon="fas fa-file-alt"
              label="Total Resumes"
              value={stats?.total_resumes}
            />
            <StatCard
              color="success"
              icon="fas fa-briefcase"
              label="Jobs Found"
              value={stats?.total_jobs}
            />
            <StatCard
              color="warning"
              icon="fas fa-hourglass-half"
              label="Pending"
              value={stats?.pending_jobs}
            />
            <StatCard
              color="success"
              icon="fas fa-check-circle"
              label="Applied"
              value={stats?.applied_jobs}
            />
          </Row>

          {error && (
            <Row className="mt-3">
              <Col><p className="text-white">{error}</p></Col>
            </Row>
          )}
        </div>
      </Container>
    </div>
  );
};

export default Header;
