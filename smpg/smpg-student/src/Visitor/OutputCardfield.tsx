export default function OutputCardfield(props :any){
    return(<>
        <div className="text-xl font-mono px-1 rounded  text-white
        bg-inherit mx-2 w-min ">
        {props.value}
        </div>
    </>)
}