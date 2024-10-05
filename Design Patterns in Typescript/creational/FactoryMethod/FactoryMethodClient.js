"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var factoryPattern_js_1 = require("./factoryPattern.js");
function clientCode(factory) {
    // create 5 new products and display description for each product
    for (var _ = 0; _ < 5; _++) {
        factory.createProduct();
    }
    factory.displayFactoryDescription();
    console.log(factory.getProduct(1).description());
}
clientCode(new factoryPattern_js_1.BucketFactory());
clientCode(new factoryPattern_js_1.ClothFactory());
