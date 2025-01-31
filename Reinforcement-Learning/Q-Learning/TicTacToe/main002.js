// Game Constants
const GRID_SIZE = 3;
let GAME_SPEED = 250;

// Rewards
const REWARDS = {
    MOVE: -0.1,
    WIN: 10.0,
    LOSE: -10.0,
    DRAW: 0.5
};

// Q-Learning Parameters
let LEARNING_RATE = 0.25;
let DISCOUNT_FACTOR = 0.9;
let EXPLORATION_RATE = 0.1;

let DEBUG = false;

const ACTIONS = [0, 1, 2, 3, 4, 5, 6, 7, 8]; // Representing all board indices

class TicTacToeGame {
    constructor() {
        this.board = Array(9).fill(0);
        this.moveHistory = [];
        this.reset();
    }

    reset() {
        this.board.fill(0);
        this.currentPlayer = 1;  // Human starts
        this.gameOver = false;
        this.moveHistory = [];
    }

    generateGridState() {
        return this.board.join(',');
    }

    getState() {
        //return this.generateGridState();
        return [...this.board];
    }

    executeAction(index, agent) {
        if (this.board[index] !== 0 || this.gameOver) return REWARDS.MOVE;

        this.board[index] = this.currentPlayer === 1 ? 1 : -1;
        if (this.currentPlayer == -1) {
            this.moveHistory.push({
                state: [...this.board],
                action: index
            });
        }
        const winner = this.checkWinner();

        if (winner === 1) {
            this.gameOver = true;
            this.updateHistoricalQValues(REWARDS.LOSE, agent);
            return REWARDS.LOSE;
        } else if (winner === -1) {
            this.gameOver = true;
            this.updateHistoricalQValues(REWARDS.WIN, agent);
            return REWARDS.WIN;
        } else if (winner === 0) {
            this.gameOver = true;
            this.updateHistoricalQValues(REWARDS.DRAW, agent);
            return REWARDS.DRAW;
        }

        this.currentPlayer *= -1;
        if (DEBUG) {console.log('THESE Q VALUES = ' + JSON.stringify(agent.getQValues([1, 0, 0, 0, 0, 0, 0, 0, 0])));}
        return REWARDS.MOVE;
    }

    updateHistoricalQValues(finalReward, agent) {
        if (DEBUG) {console.log('----update values------------------------');}
        for (let i = 0; i < this.moveHistory.length; i++) {
            const move = this.moveHistory[i];
            if (DEBUG) {console.log(`        Turn ${i}`);}
            if (DEBUG) {console.log(`            ${move["state"].slice(0,3)}`);}
            if (DEBUG) {console.log(`            ${move["state"].slice(3,6)}`);}
            if (DEBUG) {console.log(`            ${move["state"].slice(6,9)}`);}
        }
        for (let i = this.moveHistory.length - 1; i >= 0; i--) {
            const move = this.moveHistory[i];
            if (DEBUG) {console.log(`    ${JSON.stringify(move)}`);}
            if (DEBUG) {console.log(`        ${move["state"].slice(0,3)}`);}
            if (DEBUG) {console.log(`        ${move["state"].slice(3,6)}`);}
            if (DEBUG) {console.log(`        ${move["state"].slice(6,9)}`);}
            const discountedReward = finalReward * Math.pow(DISCOUNT_FACTOR, this.moveHistory.length - 1 - i);
            if (DEBUG) {console.log(`    discountedReward = ${discountedReward}`);}
            if (DEBUG) {console.log(`    about to learn`);}
            agent.learn(move.state, move.action, discountedReward, this.board);

        }
        if (DEBUG) {console.log('----end update values------------------------');}
    }

    checkWinner() {
        const winningCombos = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],
            [0, 3, 6], [1, 4, 7], [2, 5, 8],
            [0, 4, 8], [2, 4, 6]
        ];

        for (const combo of winningCombos) {
            const sum = combo.map(i => this.board[i]).reduce((a, b) => a + b, 0);
            if (sum === 3) return 1;   // Human wins
            if (sum === -3) return -1; // AI wins
        }

        if (this.board.every(cell => cell !== 0)) return 0; // Draw
        return null; // Game continues
    }
}

class QLearningAgent {
    constructor() {
        this.qTable = new Map();
    }

    clampQValue(value) {
        return Math.max(Math.min(value, REWARDS.WIN), REWARDS.LOSE);
    }

    calculateStateQValues(state) {
        const qValues = {};
        ACTIONS.forEach(action => {
            qValues[action] = this.evaluateMove(state, action);
        });
        return qValues;
    }

    evaluateMove(state, action) {
        // Basic evaluation logic
        const emptySpaces = state.filter(x => x === 0).length;
        return emptySpaces > 0 ? 0 : -1;
    }

    getQValues(state) {
        if (DEBUG) {console.log(`                            getQValues`);}
        const stateKey = state.join(',');
        if (!this.qTable.has(stateKey)) {
            if (DEBUG) {console.log(`                                !!!UNKNOWN STATE!!!`);}
            const qValues = this.calculateStateQValues(state);
            this.qTable.set(stateKey, qValues);
        }
        let thisValue = this.qTable.get(stateKey);
        if (DEBUG) {console.log(`                                "${thisValue[0]}"  "${thisValue[1]}"  "${thisValue[2]}"`);}
        if (DEBUG) {console.log(`                                "${thisValue[3]}"  "${thisValue[4]}"  "${thisValue[5]}"`);}
        if (DEBUG) {console.log(`                                "${thisValue[6]}"  "${thisValue[7]}"  "${thisValue[8]}"`);}
        return thisValue;
    }

    chooseAction(state) {
        if (DEBUG) {console.log('            chooseAction');}
        // Exploration
        if (Math.random() < EXPLORATION_RATE) {
            if (DEBUG) {console.log('                explore');}
            const availableMoves = state.map((v, i) => v === 0 ? i : -1).filter(i => i !== -1);
            return availableMoves[Math.floor(Math.random() * availableMoves.length)];
        }
        
        // Exploitation
        if (DEBUG) {console.log('                exploit');}
        if (DEBUG) {console.log('                about to get Q values');}
        const qValues = this.getQValues(state);
        const availableMoves = state.map((v, i) => v === 0 ? i : -1).filter(i => i !== -1);
        
        let bestMove = availableMoves[0];
        let bestValue = -1;

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
        if (DEBUG) {console.log('            LEARN');}
        const currentQValues = this.getQValues(state);
        //console.log(`                currentQValues =`);
        //console.log(`                    "${currentQValues[0]}"  "${currentQValues[1]}"  "${currentQValues[2]}"`);
        //console.log(`                    "${currentQValues[3]}"  "${currentQValues[4]}"  "${currentQValues[5]}"`);
        //console.log(`                    "${currentQValues[6]}"  "${currentQValues[7]}"  "${currentQValues[8]}"`);
        const currentQ = currentQValues[action] || 0;

        if (DEBUG) {console.log(`                about to get max q value`);}
        const maxNextQ = this.getMaxQValue(nextState);
        if (DEBUG) {console.log(`                maxNextQ = ${maxNextQ}`);}
        const effectiveLR = reward === REWARDS.WIN || reward === REWARDS.LOSE ? 
            LEARNING_RATE * 2 : LEARNING_RATE;
        
        const newQ = currentQ + LEARNING_RATE * (
            reward + DISCOUNT_FACTOR * maxNextQ - currentQ
        );
        if (DEBUG) {console.log(`                about to set q values`);}
        this.setQValue(state, action, newQ);
        //console.log(JSON.stringify(state));
        for (action in ACTIONS) {
            //console.log(action);
            if ( state[action] != 0 ) {
                //console.log(state[action]);
                this.setQValue(state, action, -10);
            }
        }
    }

    getMaxQValue(state) {
        if (DEBUG) {console.log(`                    getMaxQValue`);}
        const availableMoves = state.map((v, i) => v === 0 ? i : -1).filter(i => i !== -1);
        const qValues = this.getQValues(state);
        const maxValue = Math.max(...availableMoves.map(move => qValues[move] || 0));
        if (DEBUG) {console.log(`                        maxValue = ${maxValue}`);}
        return maxValue;
    }

    setQValue(state, action, value) {
        if (DEBUG) {console.log('                    setQValue');}
        const qValues = this.qTable.get(state) || Array(9).fill(0);
        if (DEBUG) {console.log(`                        OLD qValues =`);}
        if (DEBUG) {console.log(`                            "${qValues[0]}"  "${qValues[1]}"  "${qValues[2]}"`);}
        if (DEBUG) {console.log(`                            "${qValues[3]}"  "${qValues[4]}"  "${qValues[5]}"`);}
        if (DEBUG) {console.log(`                            "${qValues[6]}"  "${qValues[7]}"  "${qValues[8]}"`);}
        qValues[action] = this.clampQValue(value);
        if (isNaN(qValues[action])) {
            qValues[action] = 0;
        }
        if (DEBUG) {console.log(`                        NEW qValues =`);}
        if (DEBUG) {console.log(`                            "${qValues[0]}"  "${qValues[1]}"  "${qValues[2]}"`);}
        if (DEBUG) {console.log(`                            "${qValues[3]}"  "${qValues[4]}"  "${qValues[5]}"`);}
        if (DEBUG) {console.log(`                            "${qValues[6]}"  "${qValues[7]}"  "${qValues[8]}"`);}
        if (DEBUG) {console.log('============ = ' + JSON.stringify(this.getQValues([1, 0, 0, 0, 0, 0, 0, 0, 0])));}
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
        game.board.forEach((value, index) => {
            const cell = document.createElement('div');
            cell.className = 'cell';
            cell.textContent = value === 1 ? 'X' : value === -1 ? 'O' : '';
            if (value === 0) {
                cell.addEventListener('click', () => makeMove(index));
            } else {
                cell.classList.add('taken');
            }
            this.boardEl.appendChild(cell);
        });
    }

    updateStats(episodes, wins, draws) {
        this.statsEl.textContent = `Episodes: ${episodes} | Wins: ${wins} | Draws: ${draws}`;
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

    // Reset game stats
    episodes = 0;
    wins = 0;
    draws = 0;

    // Self-play training
    for (let i = 0; i < numGames; i++) {
        if (i % 100 == 0) { DEBUG = true; } else { DEBUG = false; }
        if (DEBUG) {console.log(`==========================BEGIN GAME ${i}==========================`);}
        await selfPlay();
        
        if (i % 10 === 0) {
            renderer.updateStats(episodes, wins, draws);
        }
        if (DEBUG) {console.log(`==========================END GAME ${i}==========================`);}
    }

    trainButton.disabled = false;
    statusEl.textContent = 'Training Complete';
    renderer.updateStats(episodes, wins, draws);
}

async function selfPlay() {
    console.log('    selfPlay');
    const game = new TicTacToeGame();
    let currentPlayer = 1;

    while (!game.gameOver) {
        let moveIndex;
        if (currentPlayer === 1) {
            // AI player 1
            if (DEBUG) {console.log('        Player 1 Begin Turn');}
            if (DEBUG) {console.log('        about to choose action');}
            moveIndex = agent.chooseAction(game.board);
            if (DEBUG) {console.log('        Player 1 chose action ' + moveIndex);}
        } else {
            // AI player 2 (opponent)
            if (DEBUG) {console.log('        Player 2 Begin Turn - THIS IS THE AGENT');}
            if (DEBUG) {console.log('        about to choose action');}
            moveIndex = agent.chooseAction(game.board);
            if (DEBUG) {console.log('        Player 2 chose action ' + moveIndex);}
        }

        const reward = game.executeAction(moveIndex, agent);
        if (DEBUG) {console.log(`        reward = ${reward}`);}
        const currentState = game.getState();
        if (DEBUG) {console.log(`        ${currentState.slice(0,3)}`);}
        if (DEBUG) {console.log(`        ${currentState.slice(3,6)}`);}
        if (DEBUG) {console.log(`        ${currentState.slice(6,9)}`);}
        const nextState = game.getState();

        //if (DEBUG) {console.log('        about to learn');}
        //agent.learn(currentState, moveIndex, reward, nextState);

        if (game.gameOver) {
            episodes++;
            if (reward === REWARDS.WIN) wins++;
            if (reward === REWARDS.DRAW) draws++;
            if (DEBUG) {console.log('the first qValues = ' + JSON.stringify(agent.getQValues([1, 0, 0, 0, 0, 0, 0, 0, 0])));}
        }

        currentPlayer *= -1;
    }
}

// Global Variables
const boardEl = document.getElementById('board');
const statsEl = document.getElementById('stats');
const debugEl = document.getElementById('debug');
const statusEl = document.getElementById('status');

const game = new TicTacToeGame();
const agent = new QLearningAgent();
const renderer = new GameRenderer(boardEl, statsEl, debugEl);

let episodes = 0;
let wins = 0;
let draws = 0;

function makeMove(index) {
    console.log('    makeMove');
    const reward = game.executeAction(index, agent);
    const currentState = game.getState().split(',').map(Number);
    const action = index;
    const nextState = game.getState().split(',').map(Number);

    console.log('        about to learn');
    if (DEBUG) {console.log('THESE Q VALUES = ' + JSON.stringify(agent.getQValues([1, 0, 0, 0, 0, 0, 0, 0, 0])));}
    agent.learn(currentState, action, reward, nextState);
    if (DEBUG) {console.log('THESE Q VALUES = ' + JSON.stringify(agent.getQValues([1, 0, 0, 0, 0, 0, 0, 0, 0])));}
    renderer.render(game);

    if (game.gameOver) {
        episodes++;
        if (reward === REWARDS.WIN) wins++;
        if (reward === REWARDS.DRAW) draws++;
        
        renderer.updateStats(episodes, wins, draws);
        game.reset();
        renderer.render(game);
    } else {
        // AI's turn
        console.log('        AI Turn');
        const aiMoveIndex = agent.chooseAction(game.board);
        const aiReward = game.executeAction(aiMoveIndex, agent);
        const aiCurrentState = game.getState().split(',').map(Number);
        const aiNextState = game.getState().split(',').map(Number);

        agent.learn(aiCurrentState, aiMoveIndex, aiReward, aiNextState);
        renderer.render(game);

        if (game.gameOver) {
            episodes++;
            if (aiReward === REWARDS.WIN) wins++;
            if (aiReward === REWARDS.DRAW) draws++;
            
            renderer.updateStats(episodes, wins, draws);
            game.reset();
            renderer.render(game);
        }
    }
}

document.getElementById('speed-slider').addEventListener('input', (e) => {
    GAME_SPEED = parseFloat(e.target.value);
    document.getElementById('speed-value').textContent = GAME_SPEED;
});

document.getElementById('epsilon-slider').addEventListener('input', (e) => {
    EXPLORATION_RATE = parseFloat(e.target.value);
    document.getElementById('epsilon-value').textContent = EXPLORATION_RATE.toFixed(2);
});

document.getElementById('gamma-slider').addEventListener('input', (e) => {
    DISCOUNT_FACTOR = parseFloat(e.target.value);
    document.getElementById('gamma-value').textContent = DISCOUNT_FACTOR.toFixed(2);
});

document.getElementById('alpha-slider').addEventListener('input', (e) => {
    LEARNING_RATE = parseFloat(e.target.value);
    document.getElementById('alpha-value').textContent = LEARNING_RATE.toFixed(2);
});

document.getElementById('train-button').addEventListener('click', trainAgent);

renderer.render(game);