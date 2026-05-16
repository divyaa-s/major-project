import React from "react";
import { Link, useNavigate } from "react-router-dom";
import "../styles/Header.css";

const Header = () => {
  const navigate = useNavigate();

  const handleSignOut = () => {
    localStorage.removeItem("isLoggedIn");
    navigate("/");
  };

  return (
    <nav className="header">
      <h1 className="header-title">Deepfake Detection</h1>
      <ul className="header-links">
        <li><Link to="/dashboard">Dashboard</Link></li>
        <li><Link to="/upload">Upload Video</Link></li>
        
        <li><Link to="/deepfake-videos">Deepfake Videos</Link></li>
        <li><Link to="/report">Report</Link></li>
        <li><Link to="/profile">Profile</Link></li>
        <li><Link to="/awareness">Awareness</Link></li>
        <li><Link to="/" onClick={handleSignOut} className="logout-btn">Sign Out</Link></li>
      </ul>
    </nav>
  );
};

export default Header;
