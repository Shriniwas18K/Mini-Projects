import { useState } from "react"
import OutputCardfield from "./OutputCardfield"
export default function Cards(props:any) {
    const [tog,asdtog]=useState('none');
    const [btn,btncon]=useState('view-details')
    return (
        <div className="cardone flex justify-center items-center h-fit ">
            <div className="max-w-4xl w-full px-6 py-10 " >
                <div>
                    <dl className="">
                        <OutputCardfield  value={props.pgname} />
                        <OutputCardfield value={props.addrs} />
                        {tog==='visible' ? 
                            <section>
                            <OutputCardfield  value={props.phone} />
                            <OutputCardfield value={props.rent} />
                            <OutputCardfield  value={props.area} />
                            <div className="px-4 py-6 sm:grid sm:grid-cols-2 sm:gap-4 sm:px-0">
                            <div className="mt-1 text-sm leading-6 text-gray-700 sm:col-span-2 sm:mt-0">
                                <div className="flex justify-center">
                                    <div className="flex flex-wrap gap-1">
                                        <img src={props.im1} alt="Image 1" className="w-72 md:w-32 h-min  rounded-md" />
                                        <img src={props.im2} alt="Image 2" className="w-72 md:w-32 h-min  rounded-md" />
                                        {/* <img src={props.im3} alt="Image 3" className="w-72 md:w-32 h-min rounded-md" /> */}
                                    </div>
                                </div>
                            </div>
                            </div> 
                            <OutputCardfield value={props.description}/>
                        </section>: undefined }
                        {tog===('none') ? 
                        <img src={props.im1} alt="Image 1" className="w-72 md:w-32 h-min  rounded-md mt-1 mx-1 my-1" />:undefined}
                        <button onClick={()=>{
                            tog==='visible' ? 
                            asdtog('none'): asdtog('visible');
                            btn==='view-details' ? 
                            btncon('hide-details') : btncon('view-details');
                        }} className="px-5 text-white rounded">{btn}</button>
                    </dl>
                </div>
            </div>
        </div>
    )
}


// async function processData() {
//     fetch('https://smpg.onrender.com/retrieveProperties/412434').then(data => {
//         return data.json()
//     }).then(data => {
//         let content=data.map((item:any) => {
            
//         })
//         return content;
//     })
// }

// processData();
