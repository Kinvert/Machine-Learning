const REWARDS = {
    MOVE: -0.01, // -0.1
    WIN: 1.0,    // 10.0
    LOSE: -1.0,  // 10.0
    DRAW: 0.1     // 5
};

let LEARNING_RATE1 = 0.025;
let DISCOUNT_FACTOR1 = 0.95;
let EXPLORATION_RATE1 = 0.995;
let DRAW_REWARD1 = 0.1;

let LEARNING_RATE2 = 0.005 //0.025;
let DISCOUNT_FACTOR2 = 0.95;
let EXPLORATION_RATE2 = 0.995;
let DRAW_REWARD2 = 0.5;

const TRAIN_ITERATIONS = 10;

let DEBUG = false;
let FLOATTYPE = 'float32';

const ACTIONS = [0, 1, 2, 3, 4, 5, 6, 7, 8];
