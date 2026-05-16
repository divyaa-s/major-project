import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import "../styles/auth.css";  

const sampleUsers = {
  "admin@example.com": "admin123",
  "user@example.com": "user123",
  "test@example.com": "password"
};

const Login = ({ setIsAuthenticated }) => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const navigate = useNavigate();

  const handleSubmit = (e) => {
    e.preventDefault();
    
    if (sampleUsers[email] && sampleUsers[email] === password) {
      setIsAuthenticated(true);  // Set authentication state to true
      navigate("/dashboard");     // Redirect to dashboard after successful login
    } else {
      alert("Invalid email or password!");
    }
  };

  return (
    <div className="signin-container">
      <h2>Login</h2>
      <form onSubmit={handleSubmit}>
        <input 
          type="email" 
          placeholder="Email" 
          value={email} 
          onChange={(e) => setEmail(e.target.value)} 
          required 
        />
        <input 
          type="password" 
          placeholder="Password" 
          value={password} 
          onChange={(e) => setPassword(e.target.value)} 
          required 
        />
        <button type="submit">Login</button>
      </form>
      <p>Don't have an account? <a href="/signup">Sign Up</a></p>
    </div>
  );
};

export default Login;
