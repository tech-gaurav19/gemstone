import React from 'react';
import Footer from './Footer';
const Courses = () => {
  return (
    <>
   
   
        <div className="row Courses " style={{margin:"10px",textAlign:"left",
                                        }}>
          <div className="col-md-12 " style={{border:" solid white",color:"white",height:"200px",padding:"20px"}}>
             <h6>For Beginners</h6>
            <h1>Basics of Stocks</h1>
          </div>
          <div className="col-md-12" style={{border:" solid white",color:"white",height:"200px",padding:"20px"}}>
          <h6>Explore in detail what is stock pricing Concepts.</h6>
            <h1>Stocks Pricing Concepts</h1>
          </div>
          <div className="col-md-12" style={{border:" solid white",color:"white",height:"200px",padding:"20px"}}>
             <h6>For Beginners</h6>
            <h1>Stocks Strategies</h1>
          </div>
         
         </div>

      <Footer/>
      </>
  );
}

export default Courses;
