import { useNavigate } from "react-router-dom";
import React, { useState } from "react";
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
  Col,
  Alert
} from "reactstrap";
import axios from "axios";

const Register = () => {
  const [nume, setNume] = useState("");
  const [email, setEmail] = useState("");
  const [parola, setParola] = useState("");
  const [feedback, setFeedback] = useState({ type: "", message: "" });

  const navigate = useNavigate();

  const handleRegister = async () => {
    try {
      const response = await axios.post("http://localhost:5000/api/users/register", {
        nume,
        email,
        parola,
        preferinte: {
          locatie: "remote",
          domeniu: ["backend", "AI"]
        }
      });

      if (response.status === 201) {
        setFeedback({ type: "success", message: "Cont creat cu succes!" });
      }
      
      setTimeout(() => {
            navigate("/admin/index");
          }, 1000
        );

    } catch (err) {
      const msg = err.response?.data?.error || "Eroare la înregistrare.";
      setFeedback({ type: "danger", message: msg });
    }
  };

  return (
    <>
      <Col lg="6" md="8">
        <Card className="bg-secondary shadow border-0">
          <CardHeader className="bg-transparent pb-5">
          </CardHeader>

          <CardBody className="px-lg-5 py-lg-5">
            <div className="text-center text-muted mb-4">
              <small>Here you can register with your email address, username and password</small>
            </div>

            {feedback.message && (
              <Alert color={feedback.type}>{feedback.message}</Alert>
            )}

            <Form role="form">
              <FormGroup>
                <InputGroup className="input-group-alternative mb-3">
                  <InputGroupAddon addonType="prepend">
                    <InputGroupText>
                      <i className="ni ni-hat-3" />
                    </InputGroupText>
                  </InputGroupAddon>
                  <Input
                    placeholder="Name"
                    type="text"
                    value={nume}
                    onChange={(e) => setNume(e.target.value)}
                  />
                </InputGroup>
              </FormGroup>

              <FormGroup>
                <InputGroup className="input-group-alternative mb-3">
                  <InputGroupAddon addonType="prepend">
                    <InputGroupText>
                      <i className="ni ni-email-83" />
                    </InputGroupText>
                  </InputGroupAddon>
                  <Input
                    placeholder="Email"
                    type="email"
                    autoComplete="new-email"
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
                    autoComplete="new-password"
                    value={parola}
                    onChange={(e) => setParola(e.target.value)}
                  />
                </InputGroup>
              </FormGroup>

              <div className="text-center">
                <Button className="mt-4" color="primary" type="button" onClick={handleRegister}>
                  Create account
                </Button>
              </div>
            </Form>
          </CardBody>
        </Card>
      </Col>
    </>
  );
};

export default Register;
