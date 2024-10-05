// when client interacts with some third party services or
// libraries then usually client interfaces are not compatiable 
// with interfaces required by the apis of those libraries
var ThirdPartyService = /** @class */ (function () {
    function ThirdPartyService() {
        // format of third party service works with hashed names
        this.hashedNames = ['#123$4', '#$4'];
    }
    ThirdPartyService.prototype.getNames = function () {
        return this.hashedNames;
    };
    return ThirdPartyService;
}());
var Adapter = /** @class */ (function () {
    function Adapter(instance) {
        this.thirdPartyServiceInstance = instance;
    }
    Adapter.prototype.getNames = function () {
        var res = this.thirdPartyServiceInstance.getNames();
        return res.map(function (r) {
            r = r.replace('#', 'S');
            r = r.replace('$', ' ');
            r = r.replace('1', 'h');
            r = r.replace('2', 'r');
            r = r.replace('3', 'i');
            r = r.replace('4', 'K');
            var arr = r.split(' ');
            return { firstName: arr[0], lastName: arr[1] };
        });
    };
    return Adapter;
}());
function clientCode() {
    var serviceInstance = new ThirdPartyService();
    console.log('cannot work with directly ThirdPartyServiceAPIs');
    console.log(serviceInstance.getNames());
    var adapter = new Adapter(serviceInstance);
    console.log(adapter.getNames());
}
clientCode();
