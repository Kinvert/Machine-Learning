
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
        this.moveHistory = [];
        this.reset();
    }

    reset() {
        this.board = '---------';
        this.currentPlayer = 1;  // Human starts
        this.gameOver = false;
        this.moveHistory = [];
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

    executeAction(index, agent) {
        if (DEBUG) {console.log(`---             executeAction()`);}
        if (this.board[index] !== '-' || this.gameOver) return REWARDS.MOVE;

        if (this.currentPlayer == -1) {
            this.moveHistory.push({
                state: this.board,
                action: index
            });
        }

        this.board = 
            this.board.slice(0, index) + 
            (this.currentPlayer === 1 ? 'X' : 'O') + 
            this.board.slice(index + 1);
        if (DEBUG) {console.log(`                    board after action = `);}
        game.printBoard();
        
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
        
        if (DEBUG) {console.log(`                    this.board = ${this.board}`);}
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
            const discountedReward = finalReward * Math.pow(DISCOUNT_FACTOR2, this.moveHistory.length - 1 - i);
            if (DEBUG) {console.log(`    discountedReward = ${discountedReward}`);}
            if (DEBUG) {console.log(`    about to learn`);}
            if (i < this.moveHistory.length - 1) {
                agent.learn(move.state, move.action, discountedReward, this.moveHistory[i+1].state);
            } else {
                agent.learn(move.state, move.action, discountedReward, null);
            }
            if (DEBUG) {console.log(`    DONE learning this state action`);}
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
            const sum = combo.map(i => {
                if (this.board[i] === 'X') return 1;
                if (this.board[i] === 'O') return -1;
                return 0;
            }).reduce((a, b) => a + b, 0);
            if (sum === 3) return 1;   // Human wins
            if (sum === -3) return -1; // AI wins
        }

        if (this.board.split('').every(cell => cell !== '-')) return 0; // Draw
        return null; // Game continues
    }
}

class QLearningAgent {
    constructor() {
        this.qTable = new Map();
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
        if ( state[action] == '-' ) {return 0;} else {return -1;}
    }

    printState(state) {
        if (DEBUG) {console.log(`                    printState()`);}
        if (DEBUG) {console.log(`                    ${state.slice(0,3)}`);}
        if (DEBUG) {console.log(`                    ${state.slice(3,6)}`);}
        if (DEBUG) {console.log(`                    ${state.slice(6,9)}`);}
    }

    getQValues(state) {
        if (DEBUG) {console.log(`                            getQValues for ${state}`);}
        if (!this.qTable.has(state)) {
            if (DEBUG) {console.log(`                                !!!UNKNOWN STATE!!! ${state}`);}
            const qValues = this.calculateStateQValues(state);
            this.qTable.set(state, qValues);
        }
        let thisValue = this.qTable.get(state);
        this.printQValues(thisValue);
        return thisValue;
    }

    chooseAction(state) {
        if (DEBUG) {console.log('            chooseAction');}
        // Exploration
        if (Math.random() < EXPLORATION_RATE2) {
            if (DEBUG) {console.log('                explore');}
            const availableMoves = state.split('').map((v, i) => v === '-' ? i : -1).filter(i => i !== -1);
            return availableMoves[Math.floor(Math.random() * availableMoves.length)];
        }
        
        // Exploitation
        if (DEBUG) {console.log('                exploit');}
        if (DEBUG) {console.log('                about to get Q values');}
        game.printBoard();
        const qValues = this.getQValues(state);
        const availableMoves = [...state]
            .map((v, i) => v === '-' ? i : -1)
            .filter(i => i !== -1);
        
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
        if (DEBUG) {console.log(`            LEARN for state ${state}`);}
        this.printState(state);
        const currentQValues = this.getQValues(state);
        const currentQ = currentQValues[action] || 0;

        if (DEBUG) {console.log(`                about to get max q value`);}
        const maxNextQ = this.getMaxQValue(nextState);
        if (DEBUG) {console.log(`                maxNextQ = ${maxNextQ}`);}
        const effectiveLR = reward === REWARDS.WIN || reward === REWARDS.LOSE ? 
            LEARNING_RATE2 * 2 : LEARNING_RATE2;
        
        const newQ = currentQ + LEARNING_RATE2 * (
            reward + DISCOUNT_FACTOR2 * maxNextQ - currentQ
        );
        if (DEBUG) {console.log(`                about to set q values`);}
        this.setQValue(state, action, newQ);
        for (action in ACTIONS) {
            if ( state[action] != '-' ) {
                this.setQValue(state, action, -10.0);
            }
        }
    }

    getMaxQValue(state) {
        if (state != null) {
            if (DEBUG) {console.log(`                    getMaxQValue`);}
            const availableMoves = state.split('').map((v, i) => v === '-' ? i : -1).filter(i => i !== -1);
            const qValues = this.getQValues(state);
            const possibleValues = availableMoves.map(move => qValues[move] ?? 0);
            if (DEBUG) {console.log(`                        possibleValues = ${possibleValues}`);}
            const maxValue = possibleValues.length > 0 ? Math.max(...possibleValues) : 0;
            if (DEBUG) {console.log(`                        maxValue = ${maxValue}`);}
            return maxValue;
        } else {
            return 0;
        }
    }

    setQValue(state, action, value, print=false) {
        if (DEBUG && print) {console.log(`                    setQValue for state${state} action${action} value${value}`);}
        const qValues = this.qTable.get(state) || Array(9).fill(0);
        if (DEBUG && print) {console.log(`                        OLD qValues =`);}
        if (DEBUG && print) {console.log(`                            "${qValues[0]}"  "${qValues[1]}"  "${qValues[2]}"`);}
        if (DEBUG && print) {console.log(`                            "${qValues[3]}"  "${qValues[4]}"  "${qValues[5]}"`);}
        if (DEBUG && print) {console.log(`                            "${qValues[6]}"  "${qValues[7]}"  "${qValues[8]}"`);}
        qValues[action] = this.clampQValue(value);
        if (isNaN(qValues[action])) {
            qValues[action] = 0;
        }
        if (DEBUG && print) {console.log(`                        NEW qValues =`);}
        if (DEBUG && print) {console.log(`                            "${qValues[0]}"  "${qValues[1]}"  "${qValues[2]}"`);}
        if (DEBUG && print) {console.log(`                            "${qValues[3]}"  "${qValues[4]}"  "${qValues[5]}"`);}
        if (DEBUG && print) {console.log(`                            "${qValues[6]}"  "${qValues[7]}"  "${qValues[8]}"`);}
    }
}

class GameRenderer {
    constructor(boardEl, statsEl, debugEl, gameBinariesEl) {
        this.boardEl = boardEl;
        this.statsEl = statsEl;
        this.debugEl = debugEl;
        this.gameBinariesEl = gameBinariesEl;
    }

    render(game) {
        this.boardEl.innerHTML = '';
        [...game.board].forEach((value, index) => { // Spread the string into an array of characters
            const cell = document.createElement('div');
            cell.className = 'cell';

            // Set the text content based on the value
            cell.textContent = value === 'X' ? 'X' : value === 'O' ? 'O' : '';

            // If the cell is empty ('-'), make it clickable
            if (value === '-') {
                cell.addEventListener('click', () => makeMove(index));
            } else {
                cell.classList.add('taken'); // Mark non-empty cells as "taken"
            }

            this.boardEl.appendChild(cell); // Add the cell to the board
        });
    }

    updateStats(episodes, wins, draws, losses) {
        this.statsEl.textContent = `Episodes: ${episodes} | Wins: ${wins} | Draws: ${draws} | Losses: ${losses}`;
    }
}

async function trainAgent() {
    console.log('Training');
    gameBinariesEl.textContent = '';
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
        if (i % 100 == 0) { DEBUG = true; } else { DEBUG = false; }
        if (DEBUG) {console.log(`==========================BEGIN GAME ${i}==========================`);}
        await selfPlay();
        
        if (i % 10 === 0) {
            renderer.updateStats(episodes, wins, draws, losses);
        }
        if (DEBUG) {console.log(`==========================END GAME ${i}==========================`);}
    }
    game.reset();
    trainButton.disabled = false;
    statusEl.textContent = 'Training Complete';
    renderer.updateStats(episodes, wins, draws, losses);
}

async function selfPlay() {
    console.log('    selfPlay');
    game.reset();
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

        if (DEBUG) {console.log('        about to execute action');}
        const reward = game.executeAction(moveIndex, agent);
        if (DEBUG) {console.log('        done executing action back in selfPlay()');}
        if (DEBUG) {console.log(`        reward = ${reward}`);}
        const currentState = game.getState();
        game.printBoard();
        const nextState = game.getState();

        if (game.gameOver) {
            episodes++;
            if (reward === REWARDS.WIN) { wins++; gameBinariesEl.textContent += '1'; }
            if (reward === REWARDS.DRAW) { draws++; gameBinariesEl.textContent += '0'; }
            if (reward === REWARDS.LOSE) { losses++; gameBinariesEl.textContent += 'X'; }
        }
        currentPlayer *= -1;
    }
}

// Global Variables
const boardEl = document.getElementById('board');
const statsEl = document.getElementById('stats');
const gameBinariesEl = document.getElementById('gameBinaries');
const debugEl = document.getElementById('debug');
const statusEl = document.getElementById('status');

const game = new TicTacToeGame();
const agent = new QLearningAgent();
const renderer = new GameRenderer(boardEl, statsEl, debugEl, gameBinariesEl);

let episodes = 0;
let wins = 0;
let draws = 0;
let losses = 0;

function makeMove(index) {
    DEBUG = true;
    console.log("");
    console.log("===========================makeMove=============================");

    // Human move
    const reward = game.executeAction(index, agent);
    renderer.render(game);

    // If the game just ended (win, lose, or draw)
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
        // Otherwise, AI's turn
        console.log("    AI Turn");
        game.printBoard();
        const aiMoveIndex = agent.chooseAction(game.board);
        const aiReward = game.executeAction(aiMoveIndex, agent);
        renderer.render(game);

        // Check if the AI’s move ended the game
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
    document.getElementById('epsilon-value1').textContent = EXPLORATION_RATE1.toFixed(2);
});
document.getElementById('gamma-slider1').addEventListener('input', (e) => {
    DISCOUNT_FACTOR1 = parseFloat(e.target.value);
    document.getElementById('gamma-value1').textContent = DISCOUNT_FACTOR1.toFixed(2);
});
document.getElementById('alpha-slider1').addEventListener('input', (e) => {
    LEARNING_RATE1 = parseFloat(e.target.value);
    document.getElementById('alpha-value1').textContent = LEARNING_RATE1.toFixed(2);
});
document.getElementById('train-button').addEventListener('click', trainAgent);
document.getElementById('epsilon-slider2').addEventListener('input', (e) => {
    EXPLORATION_RATE2 = parseFloat(e.target.value);
    document.getElementById('epsilon-value2').textContent = EXPLORATION_RATE2.toFixed(2);
});
document.getElementById('gamma-slider2').addEventListener('input', (e) => {
    DISCOUNT_FACTOR2 = parseFloat(e.target.value);
    document.getElementById('gamma-value2').textContent = DISCOUNT_FACTOR2.toFixed(2);
});
document.getElementById('alpha-slider2').addEventListener('input', (e) => {
    LEARNING_RATE2 = parseFloat(e.target.value);
    document.getElementById('alpha-value2').textContent = LEARNING_RATE2.toFixed(2);
});

renderer.render(game);