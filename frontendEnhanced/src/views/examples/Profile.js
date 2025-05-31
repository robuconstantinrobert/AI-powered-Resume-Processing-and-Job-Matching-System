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
        <Row className="justify-content-center">
          <Col lg="6" md="8" sm="10">
            <Card className="bg-secondary shadow">
              <CardHeader className="bg-white border-0">
                <Row className="align-items-center">
                  <Col xs="8">
                    <h3 className="mb-0">My account</h3>
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
