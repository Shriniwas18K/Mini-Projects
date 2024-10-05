// defining interfaces and classes of objects

interface BaseProduct{
	// Contains common properties and methods to all possible
	// ConcreteProducts which inherit this BaseProduct
	// it is never instantiated , instead only its implementing
	// ConcreteProduct classes are instantiated
	productName:string;
	description():string;
}

// ConcreteProducts implement BaseProduct specializing use cases

export class Bucket implements BaseProduct{
	// ConcreteProduct1 class
	public productName:string="bucket";
	public CapacityInLitres:number=314.34;

	public description():string{
		return `This ${this.productName} has capacity
			 ${this.CapacityInLitres} litres`;
	}
}

export class Cloth implements BaseProduct{
	// ConcreteProduct2 class
	public productName:string="cloth";
	public productColor:string="white";

	public description():string{
		return `This ${this.productName} has ${this.productColor} color`;
	}
}

/**
 * Client does not directly create objects of products instead it uses
 * the object of factory of that product. Each product has its own
 * factory. Client assumes all Child factory classes implement public methods of only
 * BaseProductFactory class and writes code using only those methods 
 * allowing more flexiblity to add more child factories easily .
 * Child Factories are ConcreteProductFactory classes.This way the client accesses only
 * BaseProductFactory methods which hides functionality of inner child Factory 
 * classes allowing client code to not break if any new ConcreteProducts are 
 * are added later, i.e. when new ConcreteProducts will be made then simply
 * we add new class implementing BaseProductFactory, and instantiating them
 * allowing existing client code to not break.
 * 
 * Client can access the methods of object as well as its implementations of methods
 * of BaseProductFactory class methods in object's Child Factory class.
 * 
 * Child factory class by default gets all methods of BaseProductFactory class
 * as they implement it.
 * 
 * NOTE : as factory objects own thier created products hence when factory objects
 *        are deleted then thier products will also be deleted as they are stored in
 *        that factory object and thier reference is passed everywhere
 *
 * NOTE : here child factory classes cannot implement any other public methods other than
 *	  the ones in BaseProductFactory class, else client code will break. Yet they
 *        can still implement thier own private methods and data member as they cannot 
 *        be accessed anywhere outside the class.Client can access only public methods 
 *        of any class's object because its rule of OOP.
 */


export abstract class BaseProductFactory{
	// Creator

	// BaseProductFactory
	
	// AbstractCreatorFactory class

	public abstract count:number;
	public abstract productName:string;

	public abstract createProduct():BaseProduct;
	public abstract getProduct(index:number):BaseProduct;

	public displayFactoryDescription():void{
		console.log(`we have ${this.count} ${this.productName}`);
	};

}

export class BucketFactory extends BaseProductFactory{

	// ConcreteCreator1 or ConcreteProduct1Factory

	private buckets:Bucket[]=[];
	public  productName:string="buckets";
	public  count:number=0;

	public createProduct():Bucket{
		console.log(`Bucket instantiated`);
		this.count++;
		this.buckets.push(new Bucket());
		return this.buckets[this.count];
	}

	public getProduct(index:number):Bucket{
		if(this.buckets.length<index)return undefined;
		return this.buckets[index];
	}
}

export class ClothFactory extends BaseProductFactory{

	// ConcreteCreator2 or ConcreteProduct2Factory

	private clothes:Cloth[]=[];
	public  productName:string="clothes";
	public  count:number=0;	

	public createProduct():Cloth{
		console.log(`Cloth instantiated`);
		this.count++;
		this.clothes.push(new Cloth());
		return this.clothes[this.count];
	}

	public getProduct(index:number):Cloth{
		if(this.clothes.length<index)return undefined;
		return this.clothes[index];
	}
}

