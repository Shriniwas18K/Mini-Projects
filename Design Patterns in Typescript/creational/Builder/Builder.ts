// this design pattern helps creation of complex 
// high customised objects at runtime.

// complex objects are composed of various attributes
// whose value is given at runtime allowing customisation

// example : we have diffirent complex objects which can be
// made by same methods but with varying attributes in them.
// These methods are accumulatyed in builder interface which
// are implemented for each complex object by buiulder classes.
// Each builder class implements builder
// interface which defines methods and members common to all 
// builder classes and these classes implement them in 
// their own way as well as each builder class can have its 
// own private members and methods not accessible outside

// Client does not directly do construction steps because he 
// may need creation of many varities of complex object
// hence he uses director who defines the ordering of construction
// methods for each variety of complex object , and will create
// them as many times client calls to create those varities.

// Builder classes should follow same Builder interface but the
// products they create may belong to diffirent interfaces.

export class House{
	public walls:number=0;
	public doors:number=0;
	public style:string="";
	public windows:number=0;
};

export class Flat extends House{
	public balcony:boolean=false;
	public bhk:number=0;
};

export class Bungalow extends House{
	public garage:boolean=false;
	public garden:boolean=false;
};

export interface Builder{
	reset():void;
	buildWalls(count:number):void;
	buildDoors(count:number):void;
	buildWindows(count:number):void;
	setStyle(style:string):void;
	getBuilding():Flat|Bungalow;
};

export class FlatBuilder implements Builder{
	private building:Flat=new Flat();
	public buildWalls(count:number):void{
		//suppose for flat a wall 
		//is shared by two flats	
		this.building.walls+=(count/2);
	};
	public buildDoors(count:number):void{
		//suppose flat has only main door
		this.building.doors=1;
	};
	public buildWindows(count:number):void{
		// suppose count arguement is 
		// number of glasses and each
		// window required three glasses
		this.building.windows+=(count/3);
	};
	public setStyle(style:string):void{
		this.building.style=style;
	};	
	public buildBalcony():void{
		this.building.balcony=true;
	};
	public getBuilding():Flat{
		return this.building;
	};
	public reset():void{
		this.building=new Flat();
	};
	public setbhk(count:number):void{
		this.building.bhk=count;
	};
}

export class BungalowBuilder implements Builder{
	private building:Bungalow=new Bungalow();
	public buildWalls(count:number):void{
		//suppose each bungalow wall requires
		//2 walls due to thickness
		this.building.walls+=(count*2);
	};
	public buildDoors(count:number):void{
		this.building.doors+=count;
	};
	public buildWindows(count:number):void{
		this.building.windows+=count;
	};
	public buildGarden():void{
		this.building.garden=true;	
	};
	public buildGarage():void{
		this.building.garage=true;
	};
	public reset():void{
		this.building=new Bungalow();
	};
	public getBuilding():Bungalow{
		return this.building;
	};
	public setStyle(style:string):void{
		this.building.style=style;	
	};
}

export class director{
	private flatBuilder:FlatBuilder=new FlatBuilder();
	private bungalowBuilder:BungalowBuilder=new BungalowBuilder();
	public get2bhkEthnicFlat():Flat{
		this.flatBuilder.reset();
		this.flatBuilder.setStyle('Ethnic');
		this.flatBuilder.setbhk(2);
		this.flatBuilder.buildBalcony();
		this.flatBuilder.buildWalls(3);
		this.flatBuilder.buildDoors(1);
		return this.flatBuilder.getBuilding();
	};
}
