"use strict";
// defining interfaces and classes of objects
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        if (typeof b !== "function" && b !== null)
            throw new TypeError("Class extends value " + String(b) + " is not a constructor or null");
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.ClothFactory = exports.BucketFactory = exports.BaseProductFactory = exports.Cloth = exports.Bucket = void 0;
// ConcreteProducts implement BaseProduct specializing use cases
var Bucket = /** @class */ (function () {
    function Bucket() {
        // ConcreteProduct1 class
        this.productName = "bucket";
        this.CapacityInLitres = 314.34;
    }
    Bucket.prototype.description = function () {
        return "This ".concat(this.productName, " has capacity\n\t\t\t ").concat(this.CapacityInLitres, " litres");
    };
    return Bucket;
}());
exports.Bucket = Bucket;
var Cloth = /** @class */ (function () {
    function Cloth() {
        // ConcreteProduct2 class
        this.productName = "cloth";
        this.productColor = "white";
    }
    Cloth.prototype.description = function () {
        return "This ".concat(this.productName, " has ").concat(this.productColor, " color");
    };
    return Cloth;
}());
exports.Cloth = Cloth;
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
var BaseProductFactory = /** @class */ (function () {
    function BaseProductFactory() {
    }
    BaseProductFactory.prototype.displayFactoryDescription = function () {
        console.log("we have ".concat(this.count, " ").concat(this.productName));
    };
    ;
    return BaseProductFactory;
}());
exports.BaseProductFactory = BaseProductFactory;
var BucketFactory = /** @class */ (function (_super) {
    __extends(BucketFactory, _super);
    function BucketFactory() {
        // ConcreteCreator1 or ConcreteProduct1Factory
        var _this = _super !== null && _super.apply(this, arguments) || this;
        _this.buckets = [];
        _this.productName = "buckets";
        _this.count = 0;
        return _this;
    }
    BucketFactory.prototype.createProduct = function () {
        console.log("Bucket instantiated");
        this.count++;
        this.buckets.push(new Bucket());
        return this.buckets[this.count];
    };
    BucketFactory.prototype.getProduct = function (index) {
        if (this.buckets.length < index)
            return undefined;
        return this.buckets[index];
    };
    return BucketFactory;
}(BaseProductFactory));
exports.BucketFactory = BucketFactory;
var ClothFactory = /** @class */ (function (_super) {
    __extends(ClothFactory, _super);
    function ClothFactory() {
        // ConcreteCreator2 or ConcreteProduct2Factory
        var _this = _super !== null && _super.apply(this, arguments) || this;
        _this.clothes = [];
        _this.productName = "clothes";
        _this.count = 0;
        return _this;
    }
    ClothFactory.prototype.createProduct = function () {
        console.log("Cloth instantiated");
        this.count++;
        this.clothes.push(new Cloth());
        return this.clothes[this.count];
    };
    ClothFactory.prototype.getProduct = function (index) {
        if (this.clothes.length < index)
            return undefined;
        return this.clothes[index];
    };
    return ClothFactory;
}(BaseProductFactory));
exports.ClothFactory = ClothFactory;
