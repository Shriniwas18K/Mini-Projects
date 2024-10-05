// this design pattern is used to make reusable 
// ui component libraries 
// make objects of classes independent of each others
// codes by involving deep copying of objects
// i.e. if parent class object is modified then still
// child class object can remain with copy of 
// orignal parent object.
var __assign = (this && this.__assign) || function () {
    __assign = Object.assign || function(t) {
        for (var s, i = 1, n = arguments.length; i < n; i++) {
            s = arguments[i];
            for (var p in s) if (Object.prototype.hasOwnProperty.call(s, p))
                t[p] = s[p];
        }
        return t;
    };
    return __assign.apply(this, arguments);
};
// prototypes is simply fancy word of object which
// needs to be cloned. For example say we have Golden 
// button to be cloned, then we simply add it to 
// prototype registry and from prototype registry 
// we can anyone can fetch clone of the prototype.
// typescript and javascript support prototyping
// built in languages, so instead of interface
// Prototype,we can directly implement class Prototype
// the below code can be used to create prototypes
// of any class objects and their clones will point to
// new cloned parent objects.
// javascripts has everything as objects and primitives
// each object has a property called __proto__ which
// points to its prototype.Also javascript follows
// prototype inheritance means all properties and methods
// of prototype object of an object will be inherited
// to that object. By default every object has its 
// prototype as global() object in browser and 
// [Object:null Prototype] in node.But we can manually
// set the prototype by
// 1] assigning another object as value of prototype
// 2] setting prototype using Object.create() method
var obj = {
    attr1: "val1",
    attr2: "val2"
};
// typescript compiler shows error for this because
// this __proto__ is part of javascript implicitely
// but still .js file executes correctly
console.log(obj.__proto__);
obj.__proto__ = { 'msg': "its prototype of obj" };
console.log(obj.__proto__);
// see how all prototype chains end at null
console.log(obj.__proto__.__proto__);
console.log(obj.__proto__.__proto__.__proto__);
// console.log(obj.__proto__.__proto__.__proto__.__proto__);
// cannot read properties of null will be atlast because 
// null has no properties and methods.Its building block
// of javascript language.
//when we copy object pointing to some parent object
//then the clone will store reference to same parent object
//i.e. if we modify parent object through clone then 
//the parent object pointed by orignal object will be modified
//i.e. parent object remains same.For this we need to
//explicitely clone parent object inside the object pointing
//or storing it as its value.
var parentobj = { "msg": "parent" };
var child1obj = { "msg": "child1", clone: function () { return __assign({}, this); } };
var child2obj = child1obj.clone();
console.log(child2obj);
child2obj.msg = "child2";
console.log(child1obj); //not modified "msg":"child1" 
//hence child2 is clone of child1
// if clone:function(){return this} then 
// child2 is just reference to child1 object
// and child1 would have been modified above
// this is nature of dynamically typed languages
// they maintain references count
// see class based object oriented prototyping
// we need to implement copy constructor which
// is not possible in typescript directly also if
// we are having basically multiple constructors
// for same class which is not possible directly
// in typescript, because it considers constructor
// as static method of class, i.e. it does not 
// belong to instance and not called with 
// obj.funcCall(func) and there are no ways to 
// workaround this in typescript , so we need 
// some situation based tricky function to implement
// this cloning.
