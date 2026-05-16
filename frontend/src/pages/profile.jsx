import React, { useState } from "react";
import "../styles/profile.css";

const Profile = () => {
  const [isEditing, setIsEditing] = useState(false);
  const [isChangingPassword, setIsChangingPassword] = useState(false);

  const handleEditProfile = () => {
    setIsEditing(true); // Set to true when editing starts
  };

  const handleChangePassword = () => {
    setIsChangingPassword(true); // Show password change form
  };

  const handleSavePassword = () => {
    console.log("Password changed");
    // You can add your password change logic here (e.g., API call to change the password)
    setIsChangingPassword(false); // Hide password change form after saving
  };

  return (
    <div className="profile-container">
      <h2>User Profile</h2>
      
      {/* Conditionally render either profile details or edit form */}
      {isEditing ? (
        <div className="edit-form">
          <label>Edit Name:</label>
          <input type="text" defaultValue="Admin" />
          <label>Edit Email:</label>
          <input type="email" defaultValue="Admin@example.com" />
          <button onClick={() => setIsEditing(false)}>Save</button>
        </div>
      ) : isChangingPassword ? (
        <div className="change-password-form">
          <label>New Password:</label>
          <input type="password" placeholder="Enter new password" />
          <label>Confirm New Password:</label>
          <input type="password" placeholder="Confirm new password" />
          <button onClick={handleSavePassword}>Save Password</button>
          <button onClick={() => setIsChangingPassword(false)}>Cancel</button>
        </div>
      ) : (
        <div className="profile-details">
          <p><strong>Name:</strong> Admin</p>
          <p><strong>Email:</strong> Admin@example.com</p>
        </div>
      )}

      <div className="profile-actions">
        <button onClick={handleEditProfile}>Edit Profile</button>
        <button onClick={handleChangePassword}>Change Password</button>
      </div>
    </div>
  );
};

export default Profile;
