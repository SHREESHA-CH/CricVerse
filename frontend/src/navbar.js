import React from 'react';
import './Navbar.css';

const Navbar = () => {
  return (
    <nav className="navbar">
      <div className="navbar-logo">
        Cricverse
      </div>
      <ul className="navbar-links">
        <li><a href="#contact">CRICCHATBOT</a></li>
      </ul>
      <div className="navbar-user">
        <div className="dropdown">
          <button className="dropbtn">User</button>
          <div className="dropdown-content">
            <a href="#profile">Profile</a>
            <a href="#signout">Sign Out</a>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;