"use strict";
// this design pattern helps creation of complex 
// high customised objects at runtime.
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
exports.director = exports.BungalowBuilder = exports.FlatBuilder = exports.Bungalow = exports.Flat = exports.House = void 0;
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
var House = /** @class */ (function () {
    function House() {
        this.walls = 0;
        this.doors = 0;
        this.style = "";
        this.windows = 0;
    }
    return House;
}());
exports.House = House;
;
var Flat = /** @class */ (function (_super) {
    __extends(Flat, _super);
    function Flat() {
        var _this = _super !== null && _super.apply(this, arguments) || this;
        _this.balcony = false;
        _this.bhk = 0;
        return _this;
    }
    return Flat;
}(House));
exports.Flat = Flat;
;
var Bungalow = /** @class */ (function (_super) {
    __extends(Bungalow, _super);
    function Bungalow() {
        var _this = _super !== null && _super.apply(this, arguments) || this;
        _this.garage = false;
        _this.garden = false;
        return _this;
    }
    return Bungalow;
}(House));
exports.Bungalow = Bungalow;
;
;
var FlatBuilder = /** @class */ (function () {
    function FlatBuilder() {
        this.building = new Flat();
    }
    FlatBuilder.prototype.buildWalls = function (count) {
        //suppose for flat a wall 
        //is shared by two flats	
        this.building.walls += (count / 2);
    };
    ;
    FlatBuilder.prototype.buildDoors = function (count) {
        //suppose flat has only main door
        this.building.doors = 1;
    };
    ;
    FlatBuilder.prototype.buildWindows = function (count) {
        // suppose count arguement is 
        // number of glasses and each
        // window required three glasses
        this.building.windows += (count / 3);
    };
    ;
    FlatBuilder.prototype.setStyle = function (style) {
        this.building.style = style;
    };
    ;
    FlatBuilder.prototype.buildBalcony = function () {
        this.building.balcony = true;
    };
    ;
    FlatBuilder.prototype.getBuilding = function () {
        return this.building;
    };
    ;
    FlatBuilder.prototype.reset = function () {
        this.building = new Flat();
    };
    ;
    FlatBuilder.prototype.setbhk = function (count) {
        this.building.bhk = count;
    };
    ;
    return FlatBuilder;
}());
exports.FlatBuilder = FlatBuilder;
var BungalowBuilder = /** @class */ (function () {
    function BungalowBuilder() {
        this.building = new Bungalow();
    }
    BungalowBuilder.prototype.buildWalls = function (count) {
        //suppose each bungalow wall requires
        //2 walls due to thickness
        this.building.walls += (count * 2);
    };
    ;
    BungalowBuilder.prototype.buildDoors = function (count) {
        this.building.doors += count;
    };
    ;
    BungalowBuilder.prototype.buildWindows = function (count) {
        this.building.windows += count;
    };
    ;
    BungalowBuilder.prototype.buildGarden = function () {
        this.building.garden = true;
    };
    ;
    BungalowBuilder.prototype.buildGarage = function () {
        this.building.garage = true;
    };
    ;
    BungalowBuilder.prototype.reset = function () {
        this.building = new Bungalow();
    };
    ;
    BungalowBuilder.prototype.getBuilding = function () {
        return this.building;
    };
    ;
    BungalowBuilder.prototype.setStyle = function (style) {
        this.building.style = style;
    };
    ;
    return BungalowBuilder;
}());
exports.BungalowBuilder = BungalowBuilder;
var director = /** @class */ (function () {
    function director() {
        this.flatBuilder = new FlatBuilder();
        this.bungalowBuilder = new BungalowBuilder();
    }
    director.prototype.get2bhkEthnicFlat = function () {
        this.flatBuilder.reset();
        this.flatBuilder.setStyle('Ethnic');
        this.flatBuilder.setbhk(2);
        this.flatBuilder.buildBalcony();
        this.flatBuilder.buildWalls(3);
        this.flatBuilder.buildDoors(1);
        return this.flatBuilder.getBuilding();
    };
    ;
    return director;
}());
exports.director = director;
