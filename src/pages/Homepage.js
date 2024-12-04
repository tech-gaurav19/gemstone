import React from "react";
import "../components/css/HomePage.css"
import Footer from "../components/Footer";


const HomePage = () => {
  return (
 <div>
    
    <div className="HomePage">
   
        <h4 style={{marginTop:"10px", padding:"5px 10px",position:"absolute",right:"50px",color: "#1eb2a6"  ,border:"1px solid white",borderRadius:"5px"}}><a 
                  href="https://fjpt9zhjrfcphmmmd9s5n2.streamlit.app/"
                  target="_self"
                  >Chat
            </a>
        </h4>

        <div class="page-item">
        <h1>GEMSTONE PRICE PREDICTION</h1>
        <h4>WELCOME TO THE FUTURE OF INVESTING!</h4>
        <h3 style={{color: "#1eb2a6" }}>
        <a  
                  href="https://price-forecasting-commodities.streamlit.app/"
                  target="_self"
                  >Explore Now
            </a>
        
        </h3>


        </div>

        <div class="protfilio_help">
          <h1>WHERE TO INVEST</h1>
          <h5>WE GIVE SOME PORTFOLIO OPTIMIZATION ADVICE </h5>
          <h3 style={{color: "#1eb2a6" }}>
            <a  
                  href="https://portfoliooptimization-dmjxbnte45lclf9ix2pk8p.streamlit.app/"
                  target="_self"
                  >Check Here
            </a>
        </h3>


        </div>

      
      <Footer/>
    </div>

    </div>
  )
}

export default HomePage