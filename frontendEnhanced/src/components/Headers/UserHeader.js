import { Container, Row, Col } from "reactstrap";

const UserHeader = () => (
  <div
    className="header pb-8 pt-5 pt-lg-8 d-flex align-items-center
               bg-gradient-info"       
    style={{
      minHeight: "250px",
      background: "linear-gradient(87deg,#11cdef 0,#1171ef 100%)",
    }}
  >
    <span className="mask bg-gradient-default opacity-6" />
  </div>
);

export default UserHeader;
