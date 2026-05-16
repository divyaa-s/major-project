import React, { useState } from "react";
import { Routes, Route } from "react-router-dom";
import Header from "./components/Header";
import Dashboard from "./pages/Dashboard";
import Upload from "./pages/Upload";
import History from "./pages/History";
import DeepfakeVideos from "./pages/deepfakevideos";
import Report from "./pages/Report";
import Profile from "./pages/profile";
import Login from "./pages/Login";
import Signup from "./pages/Signup";
import Awareness from "./pages/Awareness";
import Results from "./pages/Results";

const App = () => {
  const [isAuthenticated, setIsAuthenticated] = useState(false);

  return (
    <>
      {/* Conditionally render the header based on authentication */}
      {isAuthenticated && <Header />}
      
      <Routes>
        <Route path="/" element={<Login setIsAuthenticated={setIsAuthenticated} />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/upload" element={<Upload />} />
        <Route path="/history" element={<History />} />
        <Route path="/deepfake-videos" element={<DeepfakeVideos />} />
        <Route path="/results" element={<Results />} />
        <Route path="/report" element={<Report />} />
        <Route path="/profile" element={<Profile />} />
        <Route path="/signup" element={<Signup />} />
        <Route path="/awareness" element={<Awareness />} />
      </Routes>
    </>
  );
};

export default App;
