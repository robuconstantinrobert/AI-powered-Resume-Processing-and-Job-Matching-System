import { Row, Col, } from "reactstrap";

const Footer = () => {
  return (
    <footer className="footer">
      <Row className="align-items-center justify-content-xl-between">
        <Col xl="6">
          <div className="copyright text-center text-xl-left text-muted">
            © {new Date().getFullYear()}{" "}
          </div>
        </Col>

        <Col xl="6">
          
        </Col>
      </Row>
    </footer>
  );
};

export default Footer;
