import {BucketFactory,ClothFactory,BaseProductFactory} from './factoryPattern.js';

function clientCode(factory:BaseProductFactory):void{
	// create 5 new products and display description for each product
	for(let _=0;_<5;_++){ factory.createProduct();}
	factory.displayFactoryDescription();	
	console.log(factory.getProduct(1).description());
}
clientCode(new BucketFactory());
clientCode(new ClothFactory());

