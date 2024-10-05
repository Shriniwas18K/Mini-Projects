import './App.css';
import Cards from './Visitor/Card';
const properties=[
  ["a1-pg","santkrupa-colony,near dharmaraj-chowk,akurdi-411435",
  "300sqft","3000Rs/person","8999883840","description here",
  "./img1.jpg","./img2.jpeg",""],
  [],
  [],


  // abhishek fill here



  // asd
]

function App() {
  let content=properties.map((item,inasd)=>(
    <Cards addrs={item[1]} key={inasd} pgname={item[0]}
    phone={item[4]} rent={item[3]} area={item[2]}
    im1={item[6]} im2={item[7]} im3={item[8]} className="cardone"
    />
  ));
  return (
    <>
      <div className="w-full bg-black text-white hidden md:block ">
        <div className='h-20'>
          <h1 className='font-mono text-3xl text-center font-bold py-5'>Gharseek</h1>
          <form action="#" method="get" className='search-form bg-black py-5'>
            <input type="text" name="search" placeholder="Search for PG" 
            className='search-box mx-2 w-10/12 rounded-lg px-5 py-2 text-black '/>
            <button type="submit" className='bg-black w-9 h-9 text-2xl rounded'>⫸</button>
          </form>
        </div>
      </div>
      <div className="font-sans  lg:flex card-list bg-black mt-16">
        {content}
      </div>
    </>
  )
}
export default App