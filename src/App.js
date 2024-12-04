import './App.css';
import Navbar from './components/Navbar';
import Login from './pages/login';
import 'bootstrap/dist/css/bootstrap.min.css'; 
import SignUp from './pages/register';
import Homepage from './pages/Homepage'
import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import About from './components/About';
import Contact from './components/contact';


function App() {
  const isLoggedIn = window.localStorage.getItem("loggedIn");
  return (
    
    <Router>
      <div className="App">
        <Navbar />
        
        <Routes>
         
            
          <Route path="/sign-in" element={<Login />} />
          <Route path="/sign-up" element={<SignUp />} />
          <Route path="/" element={<Homepage />} />
         
          <Route path="/About" element={<About />}/>
          <Route path="/Contact" element={<Contact />}/>
          
        </Routes>
         
         
      </div>
    </Router>
    
  );
}

export default App;
