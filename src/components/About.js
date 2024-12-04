import React from "react";
import Footer from "../components/Footer"


const About =() =>{
      return(
        <>
        <div title = {"About us - Gstone"}>
      <div className="row contactus ">
        <div className="col-md-6 ">
          <img
             src="/images/financial-chart.jpg"
            alt="aboutus"
            style={{ width: "100%" }}
          />
        </div>
        <div className="col-md-6" style={{background:"white"}}>
            <h1 >Easy</h1>

            <h1 >Fast</h1>
          <p className="text-justify mt-2" >
             The Dataset of the gemstones was obtained from Investing.com.By using past data we predict the future price of different Gemstone and we also give some
              portfolio optimization Advice 
          </p>
        </div>
      </div>
    </div>
    <Footer/>
    </>
      );
};

export default About;