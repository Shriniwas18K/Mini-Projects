"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var Builder_js_1 = require("./Builder.js");
function client() {
    var Director = new Builder_js_1.director();
    var twobhk = Director.get2bhkEthnicFlat();
    console.log(twobhk.style);
}
;
client();
