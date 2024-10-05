import React,{ Suspense, useState } from 'react';
import Home from './pages/home/Home';
const Team = React.lazy(() => import('./pages/team/Team'));
const Events=React.lazy(()=>import('./pages/events/Events'));
import Footer from './Footer';
import './App.css';
export default function App() {
  const [currentPage, setCurrentPage] = useState("HOME");
  const [toggle, setToggle] = useState(false);
  const [btnChar, setBtnChar] = useState('☰');
  const handleClick = (page) => {
    setToggle(!toggle);
    btnChar === '☰' ? setBtnChar('X') : setBtnChar('☰');
    if (page !== '') {
      setCurrentPage(page);
      setBtnChar('☰');
    }
  };
  return (
    <>
      <header id="headernav">
        <button onClick={() => handleClick('')}>
          {btnChar}
        </button>
      </header>
      <nav id="menunavbar" className={toggle ? 'visible' : ''}>
        <section id="images">
        <img src="./PCCOEnobg.png" alt='pccoenobg'/>
        <img src="./AIMSAnobg.png" alt='aimsanobg'/>
        </section>
        <button onClick={() => handleClick('HOME')}>
          HOME
        </button>
        <button onClick={() => handleClick('TEAM')}>
          TEAM
        </button>
        <button onClick={() => handleClick('EVENTS')}>
          EVENTS
        </button>
      </nav>
      <section id="main">
        {currentPage === 'HOME' ?
         <Suspense fallback={<section className='loading-screen'>loading</section>}><Home /></Suspense>:
          currentPage === 'TEAM' ? 
          <Suspense fallback={<section className='loading-screen'>loading</section>}><Team /></Suspense> :
            currentPage === 'EVENTS' ? 
            <Suspense fallback={<section className='loading-screen'>loading</section>}><Events /></Suspense>: null
        }
      </section>
      <Footer />
    </>
  )
}
