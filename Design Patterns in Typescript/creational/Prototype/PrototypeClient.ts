import {Prototype,Button,GoldenButton} from "./Prototype.js"
function Client():void{
	let button=new Button();
	button.height=10;
	button.width=10;
	let gb1=new GoldenButton();
	gb1.button=button;
	let gb2=gb1.clone();
	//check wheter parent button
	//in both is cloned
	gb2.button.height=18;
	console.log(gb1.button.height);
};Client();
