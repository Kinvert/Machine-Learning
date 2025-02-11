const REWARDS = {
    MOVE: -0.1,
    WIN: 10.0,
    LOSE: -10.0,
    DRAW: 5
};

let LEARNING_RATE1 = 0.25;
let DISCOUNT_FACTOR1 = 0.9;
let EXPLORATION_RATE1 = 0.1;
let DRAW_REWARD1 = 5.0;

let LEARNING_RATE2 = 0.25;
let DISCOUNT_FACTOR2 = 0.9;
let EXPLORATION_RATE2 = 0.1;
let DRAW_REWARD2 = 5.0;

let DEBUG = false;

const ACTIONS = [0, 1, 2, 3, 4, 5, 6, 7, 8];

class TicTacToeGame {
    constructor() {
        this.reset();
    }

    reset() {
        this.board = '---------';
        this.recentAgent2Board = '---------';
        this.currentPlayer = 1; // Agent1 is 1, Agent2 is -1
        this.gameOver = false;
        this.winningCombo = null;
    }

    getState() {
        return this.board;
    }

    printBoard() {
        if (DEBUG) {console.log(`                        ${this.board.slice(0,3)}`);}
        if (DEBUG) {console.log(`                        ${this.board.slice(3,6)}`);}
        if (DEBUG) {console.log(`                        ${this.board.slice(6,9)}`);}
    }

    executeAction(index, playingAgent, waitingAgent) {
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

        this.printBoard();
        
        const winner = this.checkWinner();

        if (winner === 1) {
            this.gameOver = true;
            playingAgent.updateHistoricalQValues(REWARDS.WIN); // Agent1
            waitingAgent.updateHistoricalQValues(REWARDS.LOSE); // Agent2
            return REWARDS.LOSE; // Agent 2 loses
        } else if (winner === -1) {
            this.gameOver = true;
            playingAgent.updateHistoricalQValues(REWARDS.WIN); // Agent2
            waitingAgent.updateHistoricalQValues(REWARDS.LOSE); // Agent1
            return REWARDS.WIN; // Agent 2 wins
        } else if (winner === 0) {
            this.gameOver = true;
            playingAgent.updateHistoricalQValues(REWARDS.DRAW);
            waitingAgent.updateHistoricalQValues(REWARDS.DRAW);
            return REWARDS.DRAW; // Agent 2 draw
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
            if (sum === 3) {
                this.winningCombo = combo;
                return 1;
            }
            if (sum === -3) {
                this.winningCombo = combo;
                return -1;
            }
        }

        if (this.board.split('').every(cell => cell !== '-')) return 0;
        return null;
    }
}

class QLearningAgent {
    constructor(learningRate, discountFactor, explorationRate, drawReward, number) {
        this.qTable = new Map();
        this.moveHistory = [];
        this.learningRate = learningRate;
        this.discountFactor = discountFactor;
        this.explorationRate = explorationRate;
        this.drawReward = drawReward;
        this.num = number;
    }

    printQValues(qValues, print=false) {
        if (DEBUG && print) {
            console.log(`                                printQValues`);
            console.log(`                                "${qValues[0]}"  "${qValues[1]}"  "${qValues[2]}"`);
            console.log(`                                "${qValues[3]}"  "${qValues[4]}"  "${qValues[5]}"`);
            console.log(`                                "${qValues[6]}"  "${qValues[7]}"  "${qValues[8]}"`);
        }
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

    printState(state, print=false) {
        if (DEBUG && print) {
            console.log(`                    printState()`);
            console.log(`                    ${state.slice(0,3)}`);
            console.log(`                    ${state.slice(3,6)}`);
            console.log(`                    ${state.slice(6,9)}`);
        }
    }

    getQValues(state, print=false) {
        if (DEBUG && print) {console.log(`                            getQValues for ${state}`);}
        if (!this.qTable.has(state)) {
            if (DEBUG && print) {console.log(`                                !!!UNKNOWN STATE!!! ${state}`);}
            const qValues = this.calculateStateQValues(state);
            this.qTable.set(state, qValues);
        }
        const thisValue = this.qTable.get(state);
        this.printQValues(thisValue);
        return thisValue;
    }

    chooseAction(state) {
        // Exploration
        if (Math.random() < this.explorationRate) {
            if (DEBUG) {console.log('                EXPLORE');}
            const availableMoves = state
                .split('')
                .map((v, i) => (v === '-' ? i : -1))
                .filter(i => i !== -1);
            return availableMoves[Math.floor(Math.random() * availableMoves.length)];
        }
        
        // Exploitation
        if (DEBUG) {console.log('                EXPLOIT');}
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
        return bestMove;
    }

    learn(state, action, reward, nextState, print=false) {
        if (DEBUG && print) {console.log(`            LEARN agent${this.num} for state ${state}`);}
        this.printState(state);
        const currentQValues = this.getQValues(state);
        const currentQ = currentQValues[action] || 0;

        const maxNextQ = this.getMaxQValue(nextState);
        
        const newQ = currentQ + this.learningRate * (
            reward + this.discountFactor * maxNextQ - currentQ
        );

        this.setQValue(state, action, newQ);
        
        for (let a = 0; a < ACTIONS.length; a++) {
            if (state[a] !== '-') {
                this.setQValue(state, a, -10.0);
            }
        }
    }

    getMaxQValue(state) {
        if (state != null) {
            const availableMoves = state
                .split('')
                .map((v, i) => (v === '-' ? i : -1))
                .filter(i => i !== -1);
            const qValues = this.getQValues(state);
            const possibleValues = availableMoves.map(move => (qValues[move] ?? 0));
            const maxValue = possibleValues.length > 0 ? Math.max(...possibleValues) : 0;
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

    transformAction(actionIndex, mapArray) {
        for (let i = 0; i < 9; i++) {
            if (mapArray[i] === actionIndex) { return i; }
        }
        return actionIndex;
    }

    // Train on the whole game once it is over
    updateHistoricalQValues(finalReward) {
        if (finalReward == 5) {finalReward = this.drawReward;} // If Draw, use this Agent's draw reward set by user
        for (let i = this.moveHistory.length - 1; i >= 0; i--) {
            const discount = Math.pow(this.discountFactor, (this.moveHistory.length - 1 - i));
            const discountedReward = finalReward * discount;
            const move = this.moveHistory[i];
            const nextState = (i < this.moveHistory.length - 1)
                ? this.moveHistory[i + 1].state
                : null;
            if (DEBUG) {
                console.log(`    Move #${i} from agent${this.num} => state: ${move.state}, action: ${move.action}`);
            }
            if (trainOnRotations) {
                for (const rotationMap of allRotations) {
                    const translatedState     = this.transformBoard(move.state, rotationMap);
                    const translatedAction    = this.transformAction(move.action, rotationMap);
                    const translatedNextState = nextState
                        ? this.transformBoard(nextState, rotationMap)
                        : null;

                    this.learn(translatedState, translatedAction, discountedReward, translatedNextState);
                }
            } else {
                this.learn(move.state, move.action, discountedReward, nextState);
            }
            
        }
        this.moveHistory = [];
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
            cell.id = `cell${index}`;

            cell.textContent = value === 'X' ? 'X' : value === 'O' ? 'O' : '';

            if (value === '-') {
                cell.addEventListener('click', () => makeMove(index));
            } else {
                cell.classList.add('taken');
            }

            this.boardEl.appendChild(cell);
        });

        const oldLine = document.getElementById('winLine');
        if (oldLine) {
            oldLine.remove();
        }

        if (game.winningCombo) {
            this.drawWinLine(game.winningCombo);
        }
    }

    drawWinLine(combo) {
        const line = document.createElement('div');
        line.id = 'winLine';
        line.style.position = 'absolute';
        line.style.backgroundColor = 'blue';
        line.style.height = '6px';
        line.style.width = '0px';
        line.style.transformOrigin = '0 50%';

        const cellSize = 100;
        const cellGap = 5;
        
        // Get actual grid position within the centered board
        const firstCell = this.boardEl.firstElementChild;
        const gridRect = firstCell.getBoundingClientRect();
        const boardRect = this.boardEl.getBoundingClientRect();
        
        const gridOffsetX = gridRect.left - boardRect.left;
        const gridOffsetY = gridRect.top - boardRect.top;

        const start = combo[0];
        const end = combo[2];

        const startRow = Math.floor(start / 3);
        const startCol = start % 3;
        const endRow   = Math.floor(end / 3);
        const endCol   = end % 3;

        const x1 = startCol * (cellSize + cellGap) + cellSize / 2 + gridOffsetX;
        const y1 = startRow * (cellSize + cellGap) + cellSize / 2 + gridOffsetY;
        const x2 = endCol   * (cellSize + cellGap) + cellSize / 2 + gridOffsetX;
        const y2 = endRow   * (cellSize + cellGap) + cellSize / 2 + gridOffsetY;

        const dx = x2 - x1;
        const dy = y2 - y1;
        const length = Math.sqrt(dx * dx + dy * dy);
        const angle = Math.atan2(dy, dx) * (180 / Math.PI);

        // Correct positioning inside the board
        line.style.left = `${x1}px`;
        line.style.top  = `${y1}px`;
        line.style.width = `${length}px`;
        line.style.transform = `rotate(${angle}deg)`;

        this.boardEl.appendChild(line);
    }

    updateStats(episodes, wins1, draws1, losses1) {
        this.statsEl.textContent = `Agent2 Episodes: ${episodes} | Wins: ${wins1} | Draws: ${draws1} | Losses: ${losses1}`;
    }
}

// Training function invoked by "Train" button
async function trainAgent(numGames) {
    console.log('Training');
    const trainButton1 = document.getElementById('train-100');
    const trainButton2 = document.getElementById('train-10k');
    const trainButton3 = document.getElementById('train-1m');
    const trainGamesInput = document.getElementById('train-games');

    trainButton1.disabled = true;
    trainButton2.disabled = true;
    trainButton3.disabled = true;
    statusEl.textContent = 'Training...';
    game.reset();
    renderer.render(game);
    reapplyColorMode(game.board, game.recentAgent2Board);
    await new Promise(resolve => setTimeout(resolve, 0));

    episodes = 0;
    wins1 = 0;
    draws1 = 0;
    losses1 = 0;

    let updatesEvery = 1000;
    let numLoops = numGames / updatesEvery;
    if (numGames == 100) { updatesEvery = 100; numLoops = 1; }
    for (let j = 0; j < numLoops; j++) {
        for (let i = 0; i < updatesEvery; i++) {
            await selfPlay();
        }
        game.reset();
        
        renderer.updateStats(episodes, wins1, draws1, losses1);
        reapplyColorMode("---------", "---------");
        await new Promise(resolve => setTimeout(resolve, 0));
    }
    trainButton1.disabled = false;
    trainButton2.disabled = false;
    trainButton3.disabled = false;
    statusEl.textContent = 'Training Complete';
}

async function selfPlay() {
    game.reset();
    let currentPlayer = 1; // agent1 (X) starts

    while (!game.gameOver) {
        let moveIndex;
        if (currentPlayer === 1) {
            if (DEBUG) {console.log('        AGENT1 "X" Begin Turn (agent1)');}
            moveIndex = agent1.chooseAction(game.board);
            const reward = game.executeAction(moveIndex, agent1, agent2);

            if (game.gameOver) {
                episodes++;
            }
            if (reward === REWARDS.WIN) {
                wins1++;
            }
            if (reward === REWARDS.DRAW) {
                draws1++;
            }
            if (reward === REWARDS.LOSE) {
                losses1++;
            }
        } else {
            if (DEBUG) {console.log('        AGENT2 "O" Begin Turn (agent2)');}
            moveIndex = agent2.chooseAction(game.board);
            const reward = game.executeAction(moveIndex, agent2, agent1);

            if (game.gameOver) {
                episodes++;
            }
            if (reward === REWARDS.WIN) {
                wins1++;
            }
            if (reward === REWARDS.DRAW) {
                draws1++;
            }
            if (reward === REWARDS.LOSE) {
                losses1++;
            }
        }
        game.printBoard();
        currentPlayer *= -1;
    }
}

// Global Variables
const agent1 = new QLearningAgent(LEARNING_RATE1, DISCOUNT_FACTOR1, EXPLORATION_RATE1, DRAW_REWARD1, 1);
const agent2 = new QLearningAgent(LEARNING_RATE2, DISCOUNT_FACTOR2, EXPLORATION_RATE2, DRAW_REWARD2, 2);

const boardEl = document.getElementById('board');
const statsEl = document.getElementById('stats');
const debugEl = document.getElementById('debug');
const statusEl = document.getElementById('status');

let colorMode = 'turn1';
let trainOnRotations = false;

const rotation0Map = [0,1,2,3,4,5,6,7,8];
const rotation90Map = [6,3,0,7,4,1,8,5,2];
const rotation180Map = [8,7,6,5,4,3,2,1,0];
const rotation270Map = [2,5,8,1,4,7,0,3,6];
const mirrorXMap = [6,7,8,3,4,5,0,1,2];
const mirrorYMap = [2,1,0,5,4,3,8,7,6];
const allRotations = [rotation0Map, rotation90Map, rotation180Map, rotation270Map];

const game = new TicTacToeGame();
const renderer = new GameRenderer(boardEl, statsEl, debugEl);

let episodes = 0;
let wins1 = 0;
let draws1 = 0;
let losses1 = 0;

function updateCellColors(agent, boardState) {
    const cells = document.getElementsByClassName('cell');
    const qValues = agent.getQValues(boardState);
    
    for (let i = 0; i < 9; i++) {
        const cell = cells[i];
        const thisCell = document.getElementById(`cell${i}`);
        const qValue = qValues[i] || 0;
        
        if (qValue < 0) {
            // Negative values - red gradient
            const redIntensity = Math.abs(qValue) / 10.0;
            thisCell.style.backgroundColor = `rgb(${Math.floor(Math.min(255, (255 * redIntensity)))}, 0, 0)`;
        } else if (qValue > 0) {
            // Positive values - green gradient
            const greenIntensity = qValue / 10.0;
            thisCell.style.backgroundColor = `rgb(0, ${Math.floor(Math.min(255, (255 * greenIntensity)))}, 0)`;
        } else {
            if (thisCell) {
                thisCell.style.backgroundColor = '#2a2a2a';
            }
        }
    }
}

async function reapplyColorMode(boardState1, boardState2) {
    if (colorMode === 'none') {
        const cells = document.getElementsByClassName('cell');
        for (let i = 0; i < cells.length; i++) {
            cells[i].style.backgroundColor = '#2a2a2a';
        }
    } else if (colorMode === 'turn1') {
        const qValues = agent1.getQValues(boardState1);
        updateCellColors(agent1, boardState1);
    } else if (colorMode === 'turn2') {
        const qValues = agent2.getQValues(boardState2);
        updateCellColors(agent2, boardState2);
    }
}

function makeMove(index) {
    console.log("");
    console.log("===========================makeMove=============================");

    // In a user-vs-agent2 scenario, we treat the user as "X" and agent2 as "O"
    const reward = game.executeAction(index, agent1, agent2);
    game.recentAgent2Board = game.board;
    renderer.render(game);

    const tempBoard = game.board;

    reapplyColorMode(game.board, game.recentAgent2Board);

    if (game.gameOver) {
        episodes++;
        if (reward === REWARDS.WIN) {
            wins1++;
        } else if (reward === REWARDS.DRAW) {
            draws1++;
        } else if (reward === REWARDS.LOSE) {
            losses1++;
        }
        renderer.updateStats(episodes, wins1, draws1, losses1);
        setTimeout(() => {
            game.reset();
            renderer.render(game);
            reapplyColorMode(game.board, game.recentAgent2Board);
        }, 500);
    } else {
        // Now agent2 moves
        console.log("    Agent2's Turn");
        game.printBoard();
        reapplyColorMode(game.board, game.recentAgent2Board);
        setTimeout(() => {
            const aiMoveIndex = agent2.chooseAction(game.board);
            const aiReward = game.executeAction(aiMoveIndex, agent2, agent1);
            renderer.render(game);

            reapplyColorMode(game.board, game.recentAgent2Board);

            // Check if the game ended after Agent2's move
            if (game.gameOver) {
                episodes++;
                if (aiReward === REWARDS.WIN) {
                    wins1++;
                } else if (aiReward === REWARDS.DRAW) {
                    draws1++;
                } else if (aiReward === REWARDS.LOSE) {
                    losses1++;
                }
                renderer.updateStats(episodes, wins1, draws1, losses1);

                // Delay resetting the board by 1/2 second
                setTimeout(() => {
                    game.reset();
                    renderer.render(game);
                    reapplyColorMode(game.board, game.recentAgent2Board);
                }, 500);
            }

            DEBUG = false;
        }, 250);
    }
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
document.getElementById('draw-slider1').addEventListener('input', (e) => {
    DRAW_REWARD1 = parseFloat(e.target.value);
    agent1.drawReward = DRAW_REWARD1;
    document.getElementById('draw-value1').textContent = DRAW_REWARD1.toFixed(2);
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
document.getElementById('draw-slider2').addEventListener('input', (e) => {
    DRAW_REWARD2 = parseFloat(e.target.value);
    agent2.drawReward = DRAW_REWARD2;
    document.getElementById('draw-value2').textContent = DRAW_REWARD2.toFixed(2);
});

document.getElementById('train-100').addEventListener('click', () => {
  trainAgent(100);
});
document.getElementById('train-10k').addEventListener('click', () => {
    trainAgent(10000);
});
document.getElementById('train-1m').addEventListener('click', () => {
    trainAgent(1000000);
});

document.getElementsByName('colorMode').forEach(radio => {
    radio.addEventListener('change', (e) => {
        colorMode = e.target.value;  // "none", "turn1", or "turn2"
        console.log("colorMode is now:", colorMode);
        reapplyColorMode(game.board, game.recentAgent2Board);
    });
});

document.getElementById('rotationToggle').addEventListener('change', (e) => {
  trainOnRotations = e.target.checked;
  console.log('Train on Rotations is now:', trainOnRotations);
});

renderer.render(game);
reapplyColorMode(game.board, game.recentAgent2Board);