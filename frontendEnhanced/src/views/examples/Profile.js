import {
  Button,
  Card,
  CardHeader,
  CardBody,
  FormGroup,
  Input,
  Container,
  Row,
  Col,
} from "reactstrap";
// core components
import UserHeader from "components/Headers/UserHeader.js";
import { useState } from "react";

const Profile = () => {
  const [linkedinEmail, setLinkedinEmail] = useState("");
  const [linkedinPassword, setLinkedinPassword] = useState("");
  const [linkedinStatus, setLinkedinStatus] = useState("");
  const [isLinkedInLinked, setIsLinkedInLinked] = useState(false);

  const handleLinkedInSave = async () => {
    const user_id = localStorage.getItem("user_id");

    if (!linkedinEmail || !linkedinPassword) {
      setLinkedinStatus("Please enter both email and password.");
      return;
    }

    try {
      setLinkedinStatus("Linking account...");

      const res = await fetch("http://localhost:5000/api/linkedin/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          email: linkedinEmail,
          password: linkedinPassword,
          user_id: user_id || linkedinEmail.split("@")[0],
        }),
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.error || "Failed to link LinkedIn account.");
      }

      setLinkedinStatus("LinkedIn account linked successfully!");
      setIsLinkedInLinked(true);
      localStorage.setItem("linkedin_linked", "true"); // store for later use
    } catch (err) {
      console.error(err);
      setLinkedinStatus("Failed to link account: " + err.message);
      setIsLinkedInLinked(false);
      localStorage.setItem("linkedin_linked", "false");
    }
  };

  return (
    <>
      <UserHeader />
      {/* Page content */}
      <Container className="mt--7" fluid>
        <Row>
          <Col className="order-xl-2 mb-5 mb-xl-0" xl="4">
            <Card className="card-profile shadow">
              <Row className="justify-content-center">
                <Col className="order-lg-2" lg="3">
                  <div className="card-profile-image">
                    <a href="#pablo" onClick={(e) => e.preventDefault()}>
                      <img
                        alt="..."
                        className="rounded-circle"
                        src={require("../../assets/img/theme/team-4-800x800.jpg")}
                      />
                    </a>
                  </div>
                </Col>
              </Row>
              <CardHeader className="text-center border-0 pt-8 pt-md-4 pb-0 pb-md-4">
                <div className="d-flex justify-content-between">
                  <Button
                    className="mr-4"
                    color="info"
                    href="#pablo"
                    onClick={(e) => e.preventDefault()}
                    size="sm"
                  >
                    Connect
                  </Button>
                  <Button
                    className="float-right"
                    color="default"
                    href="#pablo"
                    onClick={(e) => e.preventDefault()}
                    size="sm"
                  >
                    Message
                  </Button>
                </div>
              </CardHeader>
              <CardBody className="pt-0 pt-md-4">
                <Row>
                  <div className="col">
                    <div className="card-profile-stats d-flex justify-content-center mt-md-5">
                      <div>
                        <span className="heading">22</span>
                        <span className="description">Friends</span>
                      </div>
                      <div>
                        <span className="heading">10</span>
                        <span className="description">Photos</span>
                      </div>
                      <div>
                        <span className="heading">89</span>
                        <span className="description">Comments</span>
                      </div>
                    </div>
                  </div>
                </Row>
                <div className="text-center">
                  <h3>
                    Jessica Jones
                    <span className="font-weight-light">, 27</span>
                  </h3>
                  <div className="h5 font-weight-300">
                    <i className="ni location_pin mr-2" />
                    Bucharest, Romania
                  </div>
                  <div className="h5 mt-4">
                    <i className="ni business_briefcase-24 mr-2" />
                    Solution Manager - Creative Tim Officer
                  </div>
                  <div>
                    <i className="ni education_hat mr-2" />
                    University of Computer Science
                  </div>
                  <hr className="my-4" />
                  <p>
                    Ryan — the name taken by Melbourne-raised, Brooklyn-based
                    Nick Murphy — writes, performs and records all of his own
                    music.
                  </p>
                  <a href="#pablo" onClick={(e) => e.preventDefault()}>
                    Show more
                  </a>
                </div>
              </CardBody>
            </Card>
          </Col>
          <Col className="order-xl-1" xl="8">
            <Card className="bg-secondary shadow">
              <CardHeader className="bg-white border-0">
                <Row className="align-items-center">
                  <Col xs="8">
                    <h3 className="mb-0">My account</h3>
                  </Col>
                  <Col className="text-right" xs="4">
                    <Button
                      color="primary"
                      href="#pablo"
                      onClick={(e) => e.preventDefault()}
                      size="sm"
                    >
                      Settings
                    </Button>
                  </Col>
                </Row>
              </CardHeader>
              <CardBody>
                <hr className="my-4" />
                  <h6 className="heading-small text-muted mb-4">LinkedIn Integration</h6>
                  <div className="pl-lg-4">
                    <FormGroup>
                      <label>LinkedIn Email</label>
                      <Input
                        className="form-control-alternative"
                        placeholder="LinkedIn Email"
                        type="email"
                        value={linkedinEmail}
                        onChange={(e) => setLinkedinEmail(e.target.value)}
                      />
                    </FormGroup>
                    <FormGroup>
                      <label>LinkedIn Password</label>
                      <Input
                        className="form-control-alternative"
                        placeholder="LinkedIn Password"
                        type="password"
                        value={linkedinPassword}
                        onChange={(e) => setLinkedinPassword(e.target.value)}
                      />
                    </FormGroup>
                    <Button color="success" onClick={handleLinkedInSave}>
                      Save
                    </Button>
                    {linkedinStatus && (
                      <p className={`mt-2 ${linkedinStatus.includes("success") ? "text-success" : "text-danger"}`}>
                        {linkedinStatus}
                      </p>
                    )}
                  </div>

              </CardBody>
            </Card>
          </Col>
        </Row>
      </Container>
    </>
  );
};

export default Profile;
