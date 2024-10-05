import {director,Flat} from './Builder.js';
function client():void{
	let Director=new director();
	let twobhk=Director.get2bhkEthnicFlat();
	console.log(twobhk.style);
};
client();
