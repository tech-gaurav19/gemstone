import React from "react";
import Footer from "./Footer";
const Contact =() =>{
      return(
        <div title={"Contact us"}>
        <div className="row contactus ">
          <div className="col-md-6 ">
            <img
              src="/images/Stock.png"
              alt="contactus"
              style={{ width: "100%",margin:"10px",padding:"20px" }}
            />
          </div>
          <div className="col-md-6" style={{backgroundColor:"white",marginTop:"10px"}}>
            <h1 className=" text-center">CONTACT US</h1>
            <p className="text-justify mt-2">
            Get in touch for future price of Gemstone 
            and portfolio optimization Suggestion.
            </p>
            <p className="mt-3" style={{cursor:"pointer"}}>
                <h6>Email : <a href="" style={{color:"blue"}}>www.help@gstone.com </a>  </h6>
             
            </p>
            <p className="mt-3" style={{cursor:"pointer"}}>
             <h6> PhoneCall  : 091-3456789</h6>
            </p>
            <p className="mt-3" style={{cursor:"pointer"}}>
           <h6>Toll free : 1800-1200-0202 </h6>
            </p>
          </div>
        </div>
        <Footer/>
      </div>
     
      );
};

export default Contact;