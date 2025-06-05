import { Nav, Container, Row, Col } from "reactstrap";

const Login = () => {
  return (
    <>
      <footer className="py-5">
        <Container>
          <Row className="align-items-center justify-content-xl-between">
            <Col xl="6">
              <Nav className="nav-footer justify-content-center justify-content-xl-end">
              </Nav>
            </Col>
          </Row>
        </Container>
      </footer>
    </>
  );
};

export default Login;
