"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var Prototype_js_1 = require("./Prototype.js");
function Client() {
    var button = new Prototype_js_1.Button();
    button.height = 10;
    button.width = 10;
    var gb1 = new Prototype_js_1.GoldenButton();
    gb1.button = button;
    var gb2 = gb1.clone();
    //check wheter parent button
    //in both is cloned
    gb2.button.height = 18;
    console.log(gb1.button.height);
}
;
Client();
