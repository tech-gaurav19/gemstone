// Footer.js

import React from 'react';

import './css/Footer.css';
import logo from '../logo1.png';
import { Link } from "react-router-dom";
import '../pages/Homepage';
const Footer = () => {
  return (
    <footer className="footer">
      <div className="footerContainer">
      <span className='footerLogo'>
  <img className='footerLogoImg' src={logo} alt="Logo" />
  
</span>
        <div className="footer-content">
          
          <div className="footer-links footer-section links">
            <h2>Quick Links</h2>
            <ul>
            
              <li><Link to="/About">About</Link></li>
             
              <li><Link to="/Contact">Contact</Link></li>
            </ul>
          </div>

          <div className="footer-links footer-section links">
            <h2>Our Services</h2>
            <ul>
             
              <li> <a
               href="https://price-forecasting-commodities.streamlit.app/"
               target="_self"
               //  rel="noopener noreferrer"
              >
               Markets
              </a></li>
              <li><a  
                  href="https://portfoliooptimization-dmjxbnte45lclf9ix2pk8p.streamlit.app/"
                  target="_self" >
                   Portfolio Suggestion
              </a></li>
            
            </ul>
          </div>

          <div className="footer-links footer-section links">
            <h2>Support</h2>
            <ul>
             
             
             <li><a 
                  href="https://fjpt9zhjrfcphmmmd9s5n2.streamlit.app/"
                  target="_self"
                  >Help Center
            </a></li>

            <li><Link to="/Contact">Submit a Request</Link></li>
           
            </ul>
          </div>

         
          
        </div>
      </div>

      <div className="footer-bottom">
        <h4>&copy; 2024 GEMSTONE PRICE PREDICTION, All Rights Reserved.</h4>
      </div>
    </footer>
  );
}

export default Footer;
