import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';
import reportWebVitals from './reportWebVitals';
// import { ChakraProvider } from '@chakra-ui/react';

import { ChakraProvider, extendTheme } from '@chakra-ui/react';
const theme = extendTheme({
  colors: {
    brand: {
      50: '#f9fafb',
    },
    text: '#FFFFFF', // Set font color to white
  },
  fonts: {
    body: 'Helvetica Neue, sans-serif',
    heading: 'Helvetica Neue, sans-serif',
  },
  styles: {
    global: {
      // Set background color to white
      body: {
        bg: '#FFFFFF',
      },
    },
    table: {
      // Set font color for table container and caption
      
     
    },
    th: {
      // Set font color for table header
      color: 'white',
    },
  },
});




const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(

  <ChakraProvider  theme={theme} >
    <App />
    </ChakraProvider>

    );

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
