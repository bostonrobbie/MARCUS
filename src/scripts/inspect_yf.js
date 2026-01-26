const yf = require('yahoo-finance2');
console.log('Require:', yf);
console.log('Default:', yf.default);
try {
    const instance = new yf.default();
    console.log('Instantiated default');
} catch (e) { console.log('Cannot instantiate default'); }
