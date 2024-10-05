// when client interacts with some third party services or
// libraries then usually client interfaces are not compatiable 
// with interfaces required by the apis of those libraries

// In that case client needs an adapter class that implements
// client interfaces but converts inside data to library interfaces
// and call the corresponding libraries apis and methods who use
// those interfaces.Here libaries are called adaptee.Client doesnot
// directly communicate with libraries instead it communiactes with
// adapter only. In short adapter is kind of translator

// here in all above discussion interface was intended to be considered as formats
// client objects and classes formats and schemas dont match with third-party service
// objects and classes and formats they want in api request or as method parameters
// so we use adapter who uses same client formats and converts them to service formats
// and calls corresponding service apis and methods using them.

interface clientNamesInterface{
	firstName:string;
	lastName:string;
}

class ThirdPartyService{
	// format of third party service works with hashed names
	public hashedNames:string[]=['#123$4','#$4'];
	public getNames():string[]{
		return this.hashedNames;
	}
}

class Adapter{
	private thirdPartyServiceInstance:ThirdPartyService;
	constructor(instance:ThirdPartyService){
		this.thirdPartyServiceInstance=instance;
	}
	public getNames():clientNamesInterface[]{
		const res=this.thirdPartyServiceInstance.getNames();
		return res.map(r=>{
			r=r.replace('#','S');
			r=r.replace('$',' ');
			r=r.replace('1','h');
			r=r.replace('2','r');
			r=r.replace('3','i');
			r=r.replace('4','K');
			let arr:string[]=r.split(' ');
			return {firstName:arr[0],lastName:arr[1]};
		})
		
	}
}

function clientCode(){
	const serviceInstance = new ThirdPartyService();
	console.log('cannot work with directly ThirdPartyServiceAPIs');
	console.log(serviceInstance.getNames());
	const adapter=new Adapter(serviceInstance);
	console.log(adapter.getNames());
}
clientCode();
