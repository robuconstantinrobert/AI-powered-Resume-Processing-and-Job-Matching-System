import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  Button,
  Card,
  CardHeader,
  CardBody,
  FormGroup,
  Form,
  Input,
  InputGroupAddon,
  InputGroupText,
  InputGroup,
  Row,
  Col,
  Alert
} from "reactstrap";
import axios from "axios";

const Login = () => {
  const [email, setEmail] = useState("");
  const [parola, setParola] = useState("");
  const [feedback, setFeedback] = useState({ type: "", message: "" });

  const navigate = useNavigate();

  const handleLogin = async () => {
    try {
      const response = await axios.post("http://localhost:5000/api/users/login", {
        email,
        parola
      });

      if (response.status === 200) {
        const { token, user_id } = response.data;
        localStorage.setItem("token", token);
        localStorage.setItem("user_id", response.data.user_id);
        setFeedback({ type: "success", message: "Autentificare reușită!" });

        setTimeout(() => {
            navigate("/admin/index");
          }, 1000
        );

        // aici poți face redirect sau alt comportament
      }
    } catch (err) {
      const msg = err.response?.data?.error || "Eroare la autentificare.";
      setFeedback({ type: "danger", message: msg });
    }
  };

  return (
    <>
      <Col lg="5" md="7">
        <Card className="bg-secondary shadow border-0">
          <CardBody className="px-lg-5 py-lg-5">
            <div className="text-center text-muted mb-4">
              <small>Sign in using your credentials</small>
            </div>

            {feedback.message && (
              <Alert color={feedback.type}>{feedback.message}</Alert>
            )}

            <Form role="form">
              <FormGroup className="mb-3">
                <InputGroup className="input-group-alternative">
                  <InputGroupAddon addonType="prepend">
                    <InputGroupText>
                      <i className="ni ni-email-83" />
                    </InputGroupText>
                  </InputGroupAddon>
                  <Input
                    placeholder="Email"
                    type="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                  />
                </InputGroup>
              </FormGroup>
              <FormGroup>
                <InputGroup className="input-group-alternative">
                  <InputGroupAddon addonType="prepend">
                    <InputGroupText>
                      <i className="ni ni-lock-circle-open" />
                    </InputGroupText>
                  </InputGroupAddon>
                  <Input
                    placeholder="Password"
                    type="password"
                    value={parola}
                    onChange={(e) => setParola(e.target.value)}
                  />
                </InputGroup>
              </FormGroup>
              <div className="text-center">
                <Button className="my-4" color="primary" type="button" onClick={handleLogin}>
                  Sign in
                </Button>
              </div>
            </Form>
          </CardBody>
        </Card>
      </Col>
    </>
  );
};

export default Login;
