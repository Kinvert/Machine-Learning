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

let LEARNING_RATE2 = 0.025;
let DISCOUNT_FACTOR2 = 0.95;
let EXPLORATION_RATE2 = 0.995;
let DRAW_REWARD2 = 0.1;

const TRAIN_ITERATIONS = 10;

let DEBUG = false;
let FLOATTYPE = 'float32';

const ACTIONS = [0, 1, 2, 3, 4, 5, 6, 7, 8];

// Function to select the appropriate backend (GPU if available, otherwise CPU)
async function setBackend() {
    // Check if WebGL backend (GPU) is available
    const webglBackend = tf.engine().findBackend('webgl');
    if (webglBackend) {
        await tf.setBackend('webgl');
        console.log('Using GPU acceleration with WebGL backend.');
    } else {
        await tf.setBackend('cpu');
        console.log('GPU acceleration not available; using CPU backend.');
    }
}

// Set the backend before any TensorFlow operations
setBackend().then(() => {
    // Function to encode game state for TicTacToe.
    // 'X' -> +1, 'O' -> -1, '-' -> 0
    function encodeState(stateStr) {
        const arr = stateStr.split('').map(ch => {
            if (ch === 'X') return 1.0;
            if (ch === 'O') return -1.0;
            return 0.0;
        });
        return tf.tensor2d([arr], [1, 9], FLOATTYPE);
    }

    // Example usage of encodeState:
    const sampleState = 'XOX------'; // Example TicTacToe board state.
    const stateTensor = encodeState(sampleState);
    stateTensor.print(); // Print the tensor to verify

    // Additional code for model setup and training would go here...
});

// From Numbers to Letters for rendering
function decodeState(state) {
    return state.map(val => {
        if (val === 1) return 'X';
        if (val === -1) return 'O';
        return '-';
    }).join('');
}

// From Letters to Numbers - Only used for backward compatibility
function encodeState(stateStr) {
    if (Array.isArray(stateStr)) {
        return tf.tensor2d([stateStr], [1, 9], FLOATTYPE);
    }
    // 'X' -> +1, 'O' -> -1, '-' -> 0
    const arr = stateStr.split('').map(ch => {
        if (ch === 'X') return 1.0;
        if (ch === 'O') return -1.0;
        return 0.0;
    });
    return tf.tensor2d([arr], [1, 9], FLOATTYPE);
}

function calculateEpsilonDecay(currentEpisode, totalEpisodes) {
    const startEpsilon = 0.995;
    const endEpsilon = 0.2;
    const decay = Math.exp(Math.log(endEpsilon/startEpsilon) / (totalEpisodes * 0.5));
    const eps = Math.max(endEpsilon, startEpsilon * Math.pow(decay, currentEpisode));
    return eps;
}

class TicTacToeGame {
    constructor() {
        this.reset();
    }

    reset() {
        this.board = [0, 0, 0, 0, 0, 0, 0, 0, 0];
        this.recentAgent2Board = [0, 0, 0, 0, 0, 0, 0, 0, 0];
        this.currentPlayer = 1; // Agent1 is 1, Agent2 is -1
        this.gameOver = false;
        this.winningCombo = null;
    }

    getState() {
        return [...this.board];
    }

    getDecodedBoard() {
        return decodeState(this.board);
    }

    printBoard() {
        if (DEBUG) {
            const decodedBoard = decodeState(this.board);
            console.log(`                        ${decodedBoard.slice(0,3)}`);
            console.log(`                        ${decodedBoard.slice(3,6)}`);
            console.log(`                        ${decodedBoard.slice(6,9)}`);
        }
    }

    executeAction(index, playingAgent, waitingAgent, forcedAction=null) {
        if (forcedAction) index = forcedAction;
        if (this.board[index] !== 0 || this.gameOver) return REWARDS.MOVE;

        playingAgent.moveHistory.push({
            state: [...this.board], // Train them on the state they SAW, not the state they created
            action: index
        });

        this.board[index] = this.currentPlayer;

        this.printBoard();
        
        const winner = this.checkWinner();

        if (winner === 1) {
            this.gameOver = true;
            if (DEBUG) console.log('AGENT 1 WINS');
            playingAgent.updateHistoricalQValues(REWARDS.WIN);
            waitingAgent.updateHistoricalQValues(REWARDS.LOSE);
            return REWARDS.LOSE; // Agent 2 loses
        } else if (winner === -1) {
            this.gameOver = true;
            if (DEBUG) console.log('AGENT 2 WINS');
            playingAgent.updateHistoricalQValues(REWARDS.WIN);
            waitingAgent.updateHistoricalQValues(REWARDS.LOSE);
            return REWARDS.WIN; // Agent 2 wins
        } else if (winner === 0) {
            this.gameOver = true;
            if (DEBUG) console.log('DRAW');
            playingAgent.updateHistoricalQValues(REWARDS.DRAW);
            waitingAgent.updateHistoricalQValues(REWARDS.DRAW);
            return REWARDS.DRAW; // Agent 2 draws
        }

        this.currentPlayer *= -1;
        //if (DEBUG) {console.log(`                    this.board = ${decodeState(this.board)}`);}
        if (playingAgent.num == 2) {
            return REWARDS.MOVE; // Agent 2 moved
        } else {
            return 0.00001; // Basically nothing, they survived agent 1's move
        }
    }

    checkWinner() {
        const winningCombos = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],
            [0, 3, 6], [1, 4, 7], [2, 5, 8],
            [0, 4, 8], [2, 4, 6]
        ];

        for (const combo of winningCombos) {
            const sum = combo.map(i => this.board[i]).reduce((a, b) => a + b, 0);
            if (sum === 3) {
                this.winningCombo = combo;
                return 1;
            }
            if (sum === -3) {
                this.winningCombo = combo;
                return -1;
            }
        }

        if (this.board.every(cell => cell !== 0)) return 0;
        return null;
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
        const decodedBoard = decodeState(game.board);
        [...decodedBoard].forEach((value, index) => {
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

    updateStats(episodes, wins2, draws2, losses2) {
        this.statsEl.textContent = `Agent2 Episodes: ${episodes} | Wins: ${wins2} | Draws: ${draws2} | Losses: ${losses2}`;
    }
}

class DQNAgent {
    constructor(learningRate, discountFactor, explorationRate, drawReward, number) {
        // Main Network
        this.model = tf.sequential();
        this.model.add(tf.layers.dense({
            units: 64,
            activation: 'tanh', // tanh relu
            inputShape: [9],
            kernelInitializer: 'glorotNormal', // glorotNormal zeros
            dtype: FLOATTYPE
        }));
        this.model.add(tf.layers.batchNormalization());
        this.model.add(tf.layers.dense({
            units: 64,
            activation: 'tanh',
            kernelInitializer: 'glorotNormal',
            dtype: FLOATTYPE
        }));
        this.model.add(tf.layers.batchNormalization());
        this.model.add(tf.layers.dense({
            units: 64,
            activation: 'tanh',
            kernelInitializer: 'glorotNormal',
            dtype: FLOATTYPE
        }));
        this.model.add(tf.layers.batchNormalization());
        this.model.add(tf.layers.dense({
            units: 9,
            activation: 'tanh',
            kernelInitializer: 'glorotNormal',
            dtype: FLOATTYPE
        }));
        
        // Target Network
        this.targetModel = tf.sequential();
        this.targetModel.add(tf.layers.dense({
            units: 64,
            activation: 'tanh',
            inputShape: [9],
            kernelInitializer: 'glorotNormal',
            dtype: FLOATTYPE
        }));
        this.targetModel.add(tf.layers.batchNormalization());
        this.targetModel.add(tf.layers.dense({
            units: 64,
            activation: 'tanh',
            kernelInitializer: 'glorotNormal',
            dtype: FLOATTYPE
        }));
        this.targetModel.add(tf.layers.batchNormalization());
        this.targetModel.add(tf.layers.dense({
            units: 64,
            activation: 'tanh',
            kernelInitializer: 'glorotNormal',
            dtype: FLOATTYPE
        }));
        this.targetModel.add(tf.layers.batchNormalization());
        this.targetModel.add(tf.layers.dense({
            units: 9,
            activation: 'tanh',
            kernelInitializer: 'glorotNormal',
            dtype: FLOATTYPE
        }));
        
        this.updateTargetModel(); // Initialize target model with main model weights
        this.updateCounter = 0;
        this.targetUpdateFrequency = 500; // Update target network every 500 training steps
        
        // Use RMSprop optimizer with gradient clipping
        //const optimizer = tf.train.rmsprop(learningRate, 0.95, 0.01, 1e-7);
        const optimizer = tf.train.adam(learningRate);
        this.model.compile({
            optimizer: optimizer,
            loss: 'meanSquaredError'
        });

        this.maxBufferSize = 10000;
        this.batchSize = 2048;
        
        this.learningRate = learningRate;
        this.discountFactor = discountFactor;
        this.explorationRate = explorationRate;
        this.drawReward = drawReward;
        this.num = number;
        this.replayBuffer = [];
        this.moveHistory = [];
        this._fitInProgress = false;
    }
    
    async updateTargetModel() {
        const weights = this.model.getWeights();
        this.targetModel.setWeights(weights);
    }

    async predictQValues(state) {
        const inputTensor = encodeState(state);
        if (DEBUG) console.log(`inputTensor = ${inputTensor}`);
        const output = this.model.predict(inputTensor).dataSync();
        inputTensor.dispose();
        return output;
    }

    // Epsilon-greedy policy
    async chooseAction(state) {
        if (Math.random() < this.explorationRate) {
            // Explore
            const availableMoves = Array.isArray(state) ? 
                state.map((v, i) => (v === 0 ? i : -1)).filter(i => i !== -1) :
                state.split('').map((v, i) => (v === '-' ? i : -1)).filter(i => i !== -1);
            return availableMoves[Math.floor(Math.random() * availableMoves.length)];
        }
        // Exploit
        const qValues = await this.predictQValues(state);
        const availableMoves = Array.isArray(state) ? 
            state.map((v, i) => (v === 0 ? i : -1)).filter(i => i !== -1) :
            state.split('').map((v, i) => (v === '-' ? i : -1)).filter(i => i !== -1);

        let bestMove = availableMoves[0];
        let bestValue = -Infinity;
        for (const move of availableMoves) {
            if (qValues[move] > bestValue) {
                bestValue = qValues[move];
                bestMove = move;
            }
        }
        return bestMove;
    }

    transformBoard(board, mapArray) {
        if (Array.isArray(board)) {
            const newArr = Array(9).fill(0);
            for (let i = 0; i < 9; i++) {
                newArr[i] = board[mapArray[i]];
            }
            return newArr;
        } else {
            const arr = board.split('');
            const newArr = Array(9).fill('-');
            for (let i = 0; i < 9; i++) {
                newArr[i] = arr[mapArray[i]];
            }
            return newArr.join('');
        }
    }

    transformAction(actionIndex, mapArray) {
        for (let i = 0; i < 9; i++) {
            if (mapArray[i] === actionIndex) { return i; }
        }
        return actionIndex;
    }

    async updateHistoricalQValues(finalReward) {
        //if (DEBUG) console.log(`Storing Values for Agent ${this.num}`);
        if (finalReward === REWARDS.DRAW) finalReward = this.drawReward;
        
        const numMoves = this.moveHistory.length;
        for (let i = numMoves - 1; i >= 0; i--) {
            const move = this.moveHistory[i];
            if (!move || !move.state) continue;
            
            // Single discount factor for the whole sequence
            const discount = Math.pow(this.discountFactor, numMoves - 1 - i);
            const discountedReward = finalReward * discount;
            const nextState = (i < numMoves - 1) ? this.moveHistory[i + 1].state : null;
            
            this.storeExperience(
                move.state,
                move.action,
                discountedReward,
                nextState,
                1.0
            );
        }
        
        this.moveHistory = [];
        
        const trainIterations = Math.min(TRAIN_ITERATIONS, Math.floor(this.replayBuffer.length / this.batchSize));
        if (DEBUG) console.log(`Agent${this.num} trainIterations = ${trainIterations}`);
        for (let i = 0; i < trainIterations; i++) {
            await this.trainOnBatch();
        }
    }

    // Store experience prioritization
    storeExperience(state, action, reward, nextState, discountFactor) {
        const experience = {
            state,
            action,
            reward,
            nextState,
            discountFactor,
            priority: Math.abs(reward) // Simple priority based on reward magnitude
        };
        
        this.replayBuffer.push(experience);
        if (this.replayBuffer.length > this.maxBufferSize) {
            this.replayBuffer.shift();
        }
    }

    // Prioritized sampling for batch training
    async trainOnBatch() {
        if (this._fitInProgress) {
            if (DEBUG) console.log(`Agent${this.num} _fitInProgress`);
            return;
        } else {
            if (DEBUG) console.log(`Agent${this.num} fitting`);
        }
        if (this.replayBuffer.length < this.batchSize) return;

        this._fitInProgress = true;
        try {
            const sortedBuffer = [...this.replayBuffer].sort((a, b) => b.priority - a.priority);
            if (DEBUG) console.log('sortedBuffer = ', JSON.stringify(sortedBuffer));
            const batch = [];
            const batchSize = Math.min(this.batchSize, this.replayBuffer.length);
            if (DEBUG) console.log('batchSize = ', batchSize);
            
            for (let i = 0; i < batchSize; i++) {
                // Weighted random sampling
                const idx = Math.floor(Math.pow(Math.random(), 2) * sortedBuffer.length);
                batch.push(sortedBuffer[idx]);
            }
            if (DEBUG) console.log('batch = ', JSON.stringify(batch));

            let rotatedBatch = [];
            if (trainOnRotations) {
                for (const experience of batch) {
                    for (const rotationMap of allRotations) {
                        rotatedBatch.push({
                            state: this.transformBoard(experience.state, rotationMap),
                            action: this.transformAction(experience.action, rotationMap),
                            reward: experience.reward,
                            nextState: experience.nextState ? this.transformBoard(experience.nextState, rotationMap) : null,
                            discountFactor: experience.discountFactor,
                            priority: experience.priority
                        });
                    }
                }
            } else {
                rotatedBatch = batch;
            }
            if (DEBUG) console.log('rotatedBatch = ', JSON.stringify(rotatedBatch));

            // Use separate tensors for states and next states
            const states = tf.tidy(() => tf.concat(
                rotatedBatch.map(item => encodeState(item.state))
            ));
            if (DEBUG) console.log('states = ', JSON.stringify(states));
            
            const nextStates = tf.tidy(() => {
                const validNextStates = rotatedBatch
                    .filter(item => item.nextState !== null)
                    .map(item => encodeState(item.nextState));
                return validNextStates.length > 0 ? tf.concat(validNextStates) : null;
            });
            
            const targetQs = tf.tidy(() => {
                // Split batch into terminal and non-terminal states
                const terminalStates = rotatedBatch.filter(exp => !exp.nextState);
                const nonTerminalStates = rotatedBatch.filter(exp => exp.nextState);
                
                // Process terminal states (they only need rewards)
                const terminalQs = this.model.predict(
                    tf.tensor2d(terminalStates.map(exp => exp.state), [terminalStates.length, 9], FLOATTYPE)
                ).arraySync();

                // Update terminal states with just rewards
                terminalStates.forEach((exp, i) => {
                    terminalQs[i][exp.action] = exp.reward;
                });

                // Process non-terminal states
                let nonTerminalQs = [];
                if (nonTerminalStates.length > 0) {
                    const currentPredictions = this.model.predict(
                        tf.tensor2d(nonTerminalStates.map(exp => exp.state), [nonTerminalStates.length, 9], FLOATTYPE)
                    );

                    const nextStatePredictions = this.targetModel.predict(
                        tf.tensor2d(nonTerminalStates.map(exp => exp.nextState), [nonTerminalStates.length, 9], FLOATTYPE)
                    );

                    const maxNextQs = nextStatePredictions.max(1).arraySync();
                    nonTerminalQs = currentPredictions.arraySync();

                    // Update non-terminal states with discounted rewards
                    nonTerminalStates.forEach((exp, i) => {
                        nonTerminalQs[i][exp.action] = exp.reward + this.discountFactor * maxNextQs[i];
                    });

                    nextStatePredictions.dispose();
                    currentPredictions.dispose();
                }

                // Combine results in original order
                const allQs = new Array(rotatedBatch.length);
                let terminalIdx = 0;
                let nonTerminalIdx = 0;

                rotatedBatch.forEach((exp, i) => {
                    if (exp.nextState) {
                        allQs[i] = nonTerminalQs[nonTerminalIdx++];
                    } else {
                        allQs[i] = terminalQs[terminalIdx++];
                    }
                });

                return tf.tensor2d(allQs, [rotatedBatch.length, 9]);
            });

            await this.model.fit(states, targetQs, {
                epochs: 2,
                verbose: 0,
                batchSize: 512
            });

            // Update target network periodically
            this.updateCounter++;
            if (this.updateCounter % this.targetUpdateFrequency === 0) {
                await this.updateTargetModel();
            }

            states.dispose();
            if (nextStates !== null) nextStates.dispose();
            targetQs.dispose();
        } finally {
            this._fitInProgress = false;
        }
    }
}


const agent1 = new DQNAgent(LEARNING_RATE1, DISCOUNT_FACTOR1, EXPLORATION_RATE1, DRAW_REWARD1, 1);
const agent2 = new DQNAgent(LEARNING_RATE2, DISCOUNT_FACTOR2, EXPLORATION_RATE2, DRAW_REWARD2, 2);

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
let wins2 = 0;
let draws2 = 0;
let losses2 = 0;

function updateCellColors(agent, boardState) {
    const cells = document.getElementsByClassName('cell');
    agent.predictQValues(boardState).then(qValues => {
        console.log(`Agent${agent.num} qValues Predicted = ${qValues}`)
        for (let i = 0; i < 9; i++) {
            const cell = document.getElementById(`cell${i}`);
            const val = qValues[i];
            if (Array.isArray(boardState) ? boardState[i] !== 0 : boardState[i] !== '-') {
                cell.style.backgroundColor = '#2a2a2a';
                continue;
            }
            if (val < 0) {
                const redIntensity = Math.min(255, 255 * (Math.abs(val) / REWARDS.WIN));
                cell.style.backgroundColor = `rgb(${Math.floor(redIntensity)}, 0, 0)`;
            } else if (val > 0) {
                const greenIntensity = Math.min(255, 255 * (val / REWARDS.WIN));
                cell.style.backgroundColor = `rgb(0, ${Math.floor(greenIntensity)}, 0)`;
            } else {
                cell.style.backgroundColor = '#2a2a2a';
            }
        }
    });
}

async function reapplyColorMode(boardState1, boardState2) {
    if (DEBUG) console.log(`BoardState1 = ${boardState1}`);
    if (DEBUG) console.log(`BoardState2 = ${boardState2}`);
    if (colorMode === 'none') {
        const cells = document.getElementsByClassName('cell');
        for (let i = 0; i < cells.length; i++) {
            cells[i].style.backgroundColor = '#2a2a2a';
        }
    } else if (colorMode === 'turn1') {
        updateCellColors(agent1, boardState1);
    } else if (colorMode === 'turn2') {
        updateCellColors(agent2, boardState2);
    }
}

function makeMove(index) {
    console.log("");
    console.log("===========================makeMove=============================");
    const reward = game.executeAction(index, agent1, agent2);
    game.recentAgent2Board = [...game.board];
    renderer.render(game);

    reapplyColorMode(game.board, game.recentAgent2Board);

    if (game.gameOver) {
        episodes++;
        if (reward === REWARDS.WIN) {
            wins2++;
        } else if (reward === REWARDS.DRAW) {
            draws2++;
        } else if (reward === REWARDS.LOSE) {
            losses2++;
        }
        renderer.updateStats(episodes, wins2, draws2, losses2);
        setTimeout(() => {
            game.reset();
            renderer.render(game);
            reapplyColorMode(game.board, game.recentAgent2Board);
        }, 500);
    } else {
        console.log("    Agent2's Turn");
        game.printBoard();
        reapplyColorMode(game.board, game.recentAgent2Board);
        setTimeout(async () => {
            const aiMoveIndex = await agent2.chooseAction(game.board);
            const aiReward = game.executeAction(aiMoveIndex, agent2, agent1);
            renderer.render(game);
            reapplyColorMode(game.board, game.recentAgent2Board);

            if (game.gameOver) {
                episodes++;
                if (aiReward === REWARDS.WIN) {
                    wins2++;
                } else if (aiReward === REWARDS.DRAW) {
                    draws2++;
                } else if (aiReward === REWARDS.LOSE) {
                    losses2++;
                }
                renderer.updateStats(episodes, wins2, draws2, losses2);
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
    agent1.model.compile({ optimizer: tf.train.adam(LEARNING_RATE1), loss: 'meanSquaredError' });
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
    agent2.model.compile({ optimizer: tf.train.adam(LEARNING_RATE2), loss: 'meanSquaredError' });
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
    wins2 = 0;
    draws2 = 0;
    losses2 = 0;

    let updatesEvery = 1000;
    let numLoops = numGames / updatesEvery;
    if (numGames <= 100) {
        updatesEvery = numGames;
        numLoops = 1;
    }
    for (let j = 0; j < numLoops; j++) {
        for (let i = 0; i < updatesEvery; i++) {
            agent1.explorationRate = calculateEpsilonDecay(episodes, numGames);
            agent1.explorationRate = 1.0;
            agent2.explorationRate = calculateEpsilonDecay(episodes, numGames);
            await selfPlay();
        }
        game.reset();
        renderer.updateStats(episodes, wins2, draws2, losses2);
        console.log(episodes);
        reapplyColorMode([0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0]);
        await new Promise(resolve => setTimeout(resolve, 0));
    }
    trainButton1.disabled = false;
    trainButton2.disabled = false;
    trainButton3.disabled = false;
    statusEl.textContent = 'Training Complete';
}

async function selfPlay() {
    game.reset();
    let currentPlayer = 1;

    while (!game.gameOver) {
        let moveIndex;
        if (currentPlayer === 1) {
            if (DEBUG) {console.log('        AGENT1 "X" Begin Turn (agent1)');}
            moveIndex = await agent1.chooseAction(game.board);
            const reward = game.executeAction(moveIndex, agent1, agent2);
            if (game.gameOver) episodes++;
            if (reward === REWARDS.WIN) wins2++; // Agent 2 wins
            if (reward === REWARDS.DRAW) draws2++; // Agent 2 draws
            if (reward === REWARDS.LOSE) losses2++; // Agent 2 loses
        } else {
            if (DEBUG) {console.log('        AGENT2 "O" Begin Turn (agent2)');}
            moveIndex = await agent2.chooseAction(game.board);
            const reward = game.executeAction(moveIndex, agent2, agent1);
            if (game.gameOver) episodes++;
            if (reward === REWARDS.WIN) wins2++; // Agent 2 wins
            if (reward === REWARDS.DRAW) draws2++; // Agent 2 draws
            if (reward === REWARDS.LOSE) losses2++; // Agent 2 loses
        }
        //game.printBoard();
        currentPlayer *= -1;
    }
}

renderer.render(game);
reapplyColorMode(game.board, game.recentAgent2Board);