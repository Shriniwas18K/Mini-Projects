// single class with its single instance instantiated only once
// handling all responsibilities voilating Single Responsibility
// principle and whenever instantiated then returns
// reference to that single instance only
// redux works with this design pattern
// run this script using 
// tsc Singleton.ts --target esnext
class redux {
    static #instance;
    slices;
    constructor() {
        this.slices = [];
    }
    ;
    static getInstance() {
        if (!redux.#instance) {
            redux.#instance = new redux();
        }
        return redux.#instance;
    }
    // other instance functions and business logic
    pushSlices(slice) {
        this.slices.push(slice);
    }
    getSlices() {
        return this.slices;
    }
}
function clientCode() {
    const c1 = redux.getInstance();
    const c2 = redux.getInstance();
    console.log(c1 === c2);
    c1.pushSlices({ "msg": "react component" });
    console.log(c2.getSlices());
}
clientCode();
