
const REWARDS = {
    MOVE: -0.1,
    WIN: 10.0,
    LOSE: -10.0,
    DRAW: 0.5
};

let LEARNING_RATE1 = 0.25;
let DISCOUNT_FACTOR1 = 0.9;
let EXPLORATION_RATE1 = 0.1;

let LEARNING_RATE2 = 0.25;
let DISCOUNT_FACTOR2 = 0.9;
let EXPLORATION_RATE2 = 0.1;

let DEBUG = false;

const ACTIONS = [0, 1, 2, 3, 4, 5, 6, 7, 8];

class TicTacToeGame {
    constructor() {
        this.board = '---------';
        this.reset();
    }

    reset() {
        this.board = '---------';
        this.currentPlayer = 1;  // Agent1 starts
        this.gameOver = false;
    }

    getState() {
        return this.board;
    }

    printBoard() {
        if (DEBUG) {console.log(`                        printBoard()`);}
        if (DEBUG) {console.log(`                        ${this.board.slice(0,3)}`);}
        if (DEBUG) {console.log(`                        ${this.board.slice(3,6)}`);}
        if (DEBUG) {console.log(`                        ${this.board.slice(6,9)}`);}
    }

    executeAction(index, playingAgent, waitingAgent) {
        if (DEBUG) {console.log(`---             executeAction()`);}
        if (this.board[index] !== '-' || this.gameOver) return REWARDS.MOVE;

        // Whichever agent is acting records their state+action in their moveHistory:
        playingAgent.moveHistory.push({
            state: this.board,
            action: index
        });

        this.board =
            this.board.slice(0, index) +
            (this.currentPlayer === 1 ? 'X' : 'O') +
            this.board.slice(index + 1);
        if (DEBUG) {
            console.log(`                    board after action = `);
        }
        this.printBoard();
        
        const winner = this.checkWinner();

        if (winner === 1) {
            this.gameOver = true;
            playingAgent.updateHistoricalQValues(REWARDS.WIN);
            waitingAgent.updateHistoricalQValues(REWARDS.LOSE);
            return REWARDS.LOSE;
        } else if (winner === -1) {
            this.gameOver = true;
            playingAgent.updateHistoricalQValues(REWARDS.WIN);
            waitingAgent.updateHistoricalQValues(REWARDS.LOSE);
            return REWARDS.WIN;
        } else if (winner === 0) {
            this.gameOver = true;
            playingAgent.updateHistoricalQValues(REWARDS.DRAW);
            waitingAgent.updateHistoricalQValues(REWARDS.DRAW);
            return REWARDS.DRAW;
        }

        this.currentPlayer *= -1;
        
        if (DEBUG) {console.log(`                    this.board = ${this.board}`);}
        return REWARDS.MOVE;
    }

    checkWinner() {
        const winningCombos = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],
            [0, 3, 6], [1, 4, 7], [2, 5, 8],
            [0, 4, 8], [2, 4, 6]
        ];

        for (const combo of winningCombos) {
            const sum = combo.map(i => {
                if (this.board[i] === 'X') return 1;
                if (this.board[i] === 'O') return -1;
                return 0;
            }).reduce((a, b) => a + b, 0);
            if (sum === 3) return 1;   // First player (X)
            if (sum === -3) return -1; // Second player (O)
        }

        if (this.board.split('').every(cell => cell !== '-')) return 0; // Draw
        return null; // Game continues
    }
}

class QLearningAgent {
    constructor(learningRate, discountFactor, explorationRate, number) {
        this.qTable = new Map();
        this.moveHistory = [];
        this.learningRate = learningRate;
        this.discountFactor = discountFactor;
        this.explorationRate = explorationRate;
        this.num = number;
    }

    printQValues(qValues) {
        if (DEBUG) {console.log(`                                printQValues`);}
        if (DEBUG) {console.log(`                                "${qValues[0]}"  "${qValues[1]}"  "${qValues[2]}"`);}
        if (DEBUG) {console.log(`                                "${qValues[3]}"  "${qValues[4]}"  "${qValues[5]}"`);}
        if (DEBUG) {console.log(`                                "${qValues[6]}"  "${qValues[7]}"  "${qValues[8]}"`);}
    }

    clampQValue(value) {
        const clamped = Math.max(Math.min(value, REWARDS.WIN), REWARDS.LOSE);
        return parseFloat(clamped.toFixed(3));
    }

    calculateStateQValues(state) {
        const qValues = {};
        ACTIONS.forEach(action => {
            qValues[action] = this.evaluateMove(state, action);
        });
        return qValues;
    }

    evaluateMove(state, action) {
        if (state[action] === '-') {
            return 0;
        } else {
            return -10.0;
        }
    }

    printState(state) {
        if (DEBUG) {
            console.log(`                    printState()`);
            console.log(`                    ${state.slice(0,3)}`);
            console.log(`                    ${state.slice(3,6)}`);
            console.log(`                    ${state.slice(6,9)}`);
        }
    }

    getQValues(state) {
        if (DEBUG) {console.log(`                            getQValues for ${state}`);}
        if (!this.qTable.has(state)) {
            if (DEBUG) {console.log(`                                !!!UNKNOWN STATE!!! ${state}`);}
            const qValues = this.calculateStateQValues(state);
            this.qTable.set(state, qValues);
        }
        const thisValue = this.qTable.get(state);
        this.printQValues(thisValue);
        return thisValue;
    }

    chooseAction(state) {
        if (DEBUG) {console.log(`            chooseAction agent${this.num}`);}
        // Exploration
        if (Math.random() < this.explorationRate) {
            if (DEBUG) {console.log('                explore');}
            const availableMoves = state
                .split('')
                .map((v, i) => (v === '-' ? i : -1))
                .filter(i => i !== -1);
            return availableMoves[Math.floor(Math.random() * availableMoves.length)];
        }
        
        // Exploitation
        if (DEBUG) {console.log('                exploit');}
        if (DEBUG) {console.log('                about to get Q values');}
        const qValues = this.getQValues(state);
        const availableMoves = [...state]
            .map((v, i) => (v === '-' ? i : -1))
            .filter(i => i !== -1);
        
        let bestMove = availableMoves[0];
        let bestValue = -10.0;

        availableMoves.forEach(move => {
            const value = qValues[move] || 0;
            if (value > bestValue) {
                bestValue = value;
                bestMove = move;
            }
        });
        if (DEBUG) {console.log(`                bestMove = ${bestMove}`);}
        return bestMove;
    }

    learn(state, action, reward, nextState) {
        if (DEBUG) {console.log(`            LEARN agent${this.num} for state ${state}`);}
        this.printState(state);
        const currentQValues = this.getQValues(state);
        const currentQ = currentQValues[action] || 0;

        if (DEBUG) {console.log(`                about to get max q value`);}
        const maxNextQ = this.getMaxQValue(nextState);
        if (DEBUG) {console.log(`                maxNextQ = ${maxNextQ}`);}
        
        const newQ = currentQ + this.learningRate * (
            reward + this.discountFactor * maxNextQ - currentQ
        );

        if (DEBUG) {console.log(`                about to set q values`);}
        this.setQValue(state, action, newQ);
        
        for (let a = 0; a < ACTIONS.length; a++) {
            if (state[a] !== '-') {
                this.setQValue(state, a, -10.0);
            }
        }
    }

    getMaxQValue(state) {
        if (state != null) {
            if (DEBUG) {console.log(`                    getMaxQValue`);}
            const availableMoves = state
                .split('')
                .map((v, i) => (v === '-' ? i : -1))
                .filter(i => i !== -1);
            const qValues = this.getQValues(state);
            const possibleValues = availableMoves.map(move => (qValues[move] ?? 0));
            if (DEBUG) {console.log(`                        possibleValues = ${possibleValues}`);}
            const maxValue = possibleValues.length > 0 ? Math.max(...possibleValues) : 0;
            if (DEBUG) {console.log(`                        maxValue = ${maxValue}`);}
            return maxValue;
        } else {
            return 0;
        }
    }

    setQValue(state, action, value, print=false) {
        if (!this.qTable.has(state)) {
            this.qTable.set(state, this.calculateStateQValues(state));
        }
        const qValues = this.qTable.get(state);

        if (DEBUG && print) {
            console.log(`                    setQValue for state${state} action${action} value${value}`);
            console.log(`                        OLD qValues =`);
            console.log(`                            "${qValues[0]}"  "${qValues[1]}"  "${qValues[2]}"`);
            console.log(`                            "${qValues[3]}"  "${qValues[4]}"  "${qValues[5]}"`);
            console.log(`                            "${qValues[6]}"  "${qValues[7]}"  "${qValues[8]}"`);
        }

        qValues[action] = this.clampQValue(value);
        if (isNaN(qValues[action])) {
            qValues[action] = 0;
        }

        if (DEBUG && print) {
            console.log(`                        NEW qValues =`);
            console.log(`                            "${qValues[0]}"  "${qValues[1]}"  "${qValues[2]}"`);
            console.log(`                            "${qValues[3]}"  "${qValues[4]}"  "${qValues[5]}"`);
            console.log(`                            "${qValues[6]}"  "${qValues[7]}"  "${qValues[8]}"`);
        }
    }

    transformBoard(board, mapArray) {
        const arr = board.split('');
        const newArr = Array(9).fill('-');
        for (let i = 0; i < 9; i++) {
            newArr[i] = arr[ mapArray[i] ];
        }
        return newArr.join('');
    }

    // Trains on full game
    updateHistoricalQValues(finalReward) {
        if (DEBUG) {console.log(`----updateHistoricalQValues------------------------ agent${this.num}`);}
        for (let i = this.moveHistory.length - 1; i >= 0; i--) {
            const discount = Math.pow(this.discountFactor, (this.moveHistory.length - 1 - i));
            const discountedReward = finalReward * discount;
            const move = this.moveHistory[i];
            if (DEBUG) {
                console.log(`    Move #${i} from agent${this.num} => state: ${move.state}, action: ${move.action}`);
            }
            //for (rotation in allRotations) { // todo
            //    translatedState = this.transformBoard(move.state, rotation);
            //}

            // Next state: either the next move in the array, or null if last
            if (i < this.moveHistory.length - 1) {
                const nextState = this.moveHistory[i + 1].state;
                this.learn(move.state, move.action, discountedReward, nextState);
            } else {
                this.learn(move.state, move.action, discountedReward, null);
            }
        }
        this.moveHistory = [];
        if (DEBUG) {console.log(`----end updateHistoricalQValues------------------------ agent${this.num}`);}
    }
}

class GameRenderer {
    constructor(boardEl, statsEl, debugEl) {
        this.boardEl = boardEl;
        this.statsEl = statsEl;
        this.debugEl = debugEl;
    }

    render(game) {
        this.boardEl.innerHTML = '';
        [...game.board].forEach((value, index) => {
            const cell = document.createElement('div');
            cell.className = 'cell';

            cell.textContent = value === 'X' ? 'X' : value === 'O' ? 'O' : '';

            if (value === '-') {
                cell.addEventListener('click', () => makeMove(index));
            } else {
                cell.classList.add('taken');
            }

            this.boardEl.appendChild(cell);
        });
    }

    updateStats(episodes, wins, draws, losses) {
        this.statsEl.textContent = `Episodes: ${episodes} | Wins: ${wins} | Draws: ${draws} | Losses: ${losses}`;
    }
}

async function trainAgent() {
    console.log('Training');
    const trainButton = document.getElementById('train-button');
    const trainGamesInput = document.getElementById('train-games');
    const numGames = parseInt(trainGamesInput.value);

    if (isNaN(numGames) || numGames < 1) {
        alert('Please enter a valid number of games');
        return;
    }

    trainButton.disabled = true;
    statusEl.textContent = 'Training...';

    episodes = 0;
    wins = 0;
    draws = 0;
    losses = 0;

    for (let i = 0; i < numGames; i++) {
        if (i % 100 === 0) {
            DEBUG = true;
        } else {
            DEBUG = false;
        }
        if (DEBUG) {
            console.log(`==========================BEGIN GAME ${i}==========================`);
        }
        await selfPlay();
        
        if (i % 10 === 0) {
            renderer.updateStats(episodes, wins, draws, losses);
        }
        if (DEBUG) {
            console.log(`==========================END GAME ${i}==========================`);
        }
    }
    game.reset();
    trainButton.disabled = false;
    statusEl.textContent = 'Training Complete';
    renderer.updateStats(episodes, wins, draws, losses);
}

async function selfPlay() {
    console.log('    selfPlay');
    game.reset();
    let currentPlayer = 1; // agent1 (X) starts

    while (!game.gameOver) {
        let moveIndex;
        if (currentPlayer === 1) {
            // Agent 1 (X) goes first
            if (DEBUG) {console.log('        Agent1 Begin Turn (agent1)');}
            if (DEBUG) {console.log('        Agent1 about to choose action');}
            moveIndex = agent1.chooseAction(game.board);
            if (DEBUG) {console.log('        Player 1 chose action ' + moveIndex);}
            if (DEBUG) {console.log('        about to execute action');}
            const reward = game.executeAction(moveIndex, agent1, agent2);
            if (DEBUG) {console.log('        done executing action back in selfPlay()');}
            if (DEBUG) {console.log(`        reward = ${reward}`);}

            if (game.gameOver) {
                episodes++;
            }
        } else {
            // Agent 2 (O)
            if (DEBUG) {console.log('        AGENT2 Begin Turn (agent2)');}
            if (DEBUG) {console.log('        AGENT2 about to choose action');}
            moveIndex = agent2.chooseAction(game.board);
            if (DEBUG) {console.log('        Player 2 chose action ' + moveIndex);}
            if (DEBUG) {console.log('        about to execute action');}
            const reward = game.executeAction(moveIndex, agent2, agent1);
            if (DEBUG) {console.log('        done executing action back in selfPlay()');}
            if (DEBUG) {console.log(`        reward = ${reward}`);}

            if (game.gameOver) {
                episodes++;
                if (reward === REWARDS.WIN) {
                    wins++;
                }
                if (reward === REWARDS.DRAW) {
                    draws++;
                }
                if (reward === REWARDS.LOSE) {
                    losses++;
                }
            }
        }
        game.printBoard();
        currentPlayer *= -1;
    }
}

// Global Variables
const boardEl = document.getElementById('board');
const statsEl = document.getElementById('stats');
const debugEl = document.getElementById('debug');
const statusEl = document.getElementById('status');

const rotation0Map = [0,1,2,3,4,5,6,7,8];
const rotation90Map = [6,3,0,7,4,1,8,5,2];
const rotation180Map = [8,7,6,5,4,3,2,1,0];
const rotation270Map = [2,5,8,1,4,7,0,3,6];
const mirrorXMap = [6,7,8,3,4,5,0,1,2];
const mirrorYMap = [2,1,0,5,4,3,8,7,6];

const allRotations = [rotation0Map, rotation90Map, rotation180Map, rotation270Map];

const game = new TicTacToeGame();

const agent1 = new QLearningAgent(LEARNING_RATE1, DISCOUNT_FACTOR1, EXPLORATION_RATE1, 1);
const agent2 = new QLearningAgent(LEARNING_RATE2, DISCOUNT_FACTOR2, EXPLORATION_RATE2, 2);

const renderer = new GameRenderer(boardEl, statsEl, debugEl);

let episodes = 0;
let wins = 0;
let draws = 0;
let losses = 0;

function makeMove(index) {
    DEBUG = true;
    console.log("");
    console.log("===========================makeMove=============================");

    const reward = game.executeAction(index, agent2, agent1);
    renderer.render(game);

    if (game.gameOver) {
        episodes++;
        if (reward === REWARDS.WIN) {
            wins++;
        } else if (reward === REWARDS.DRAW) {
            draws++;
        } else if (reward === REWARDS.LOSE) {
            losses++;
        }
        renderer.updateStats(episodes, wins, draws, losses);
        game.reset();
        renderer.render(game);
    } else {
        // Now agent2 moves
        console.log("    AI Turn");
        game.printBoard();
        const aiMoveIndex = agent2.chooseAction(game.board);
        const aiReward = game.executeAction(aiMoveIndex, agent2, agent1);
        renderer.render(game);

        if (game.gameOver) {
            episodes++;
            if (aiReward === REWARDS.WIN) {
                wins++;
            } else if (aiReward === REWARDS.DRAW) {
                draws++;
            } else if (aiReward === REWARDS.LOSE) {
                losses++;
            }
            renderer.updateStats(episodes, wins, draws, losses);
            game.reset();
            renderer.render(game);
        }
    }
    DEBUG = false;
}

document.getElementById('epsilon-slider1').addEventListener('input', (e) => {
    EXPLORATION_RATE1 = parseFloat(e.target.value);
    agent1.explorationRate = EXPLORATION_RATE1;
    document.getElementById('epsilon-value1').textContent = EXPLORATION_RATE1.toFixed(2);
});
document.getElementById('gamma-slider1').addEventListener('input', (e) => {
    DISCOUNT_FACTOR1 = parseFloat(e.target.value);
    agent1.discountFactor = DISCOUNT_FACTOR1;
    document.getElementById('gamma-value1').textContent = DISCOUNT_FACTOR1.toFixed(2);
});
document.getElementById('alpha-slider1').addEventListener('input', (e) => {
    LEARNING_RATE1 = parseFloat(e.target.value);
    agent1.learningRate = LEARNING_RATE1;
    document.getElementById('alpha-value1').textContent = LEARNING_RATE1.toFixed(2);
});

document.getElementById('epsilon-slider2').addEventListener('input', (e) => {
    EXPLORATION_RATE2 = parseFloat(e.target.value);
    agent2.explorationRate = EXPLORATION_RATE2;
    document.getElementById('epsilon-value2').textContent = EXPLORATION_RATE2.toFixed(2);
});
document.getElementById('gamma-slider2').addEventListener('input', (e) => {
    DISCOUNT_FACTOR2 = parseFloat(e.target.value);
    agent2.discountFactor = DISCOUNT_FACTOR2;
    document.getElementById('gamma-value2').textContent = DISCOUNT_FACTOR2.toFixed(2);
});
document.getElementById('alpha-slider2').addEventListener('input', (e) => {
    LEARNING_RATE2 = parseFloat(e.target.value);
    agent2.learningRate = LEARNING_RATE2;
    document.getElementById('alpha-value2').textContent = LEARNING_RATE2.toFixed(2);
});

document.getElementById('train-button').addEventListener('click', trainAgent);

renderer.render(game);