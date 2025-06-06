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

    executeAction(index, playingAgent, waitingAgent, forcedAction=null) {
        if (forcedAction) index = forcedAction;
        if (this.board[index] !== 0 || this.gameOver) return REWARDS.MOVE;

        playingAgent.moveHistory.push({
            state: [...this.board],
            action: index
        });

        this.board[index] = this.currentPlayer;

        printBoard(this.board);
        
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
            return 0.00001;
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

function buildLayers(model) {
    model.add(tf.layers.dense({
        units: 128,
        activation: 'tanh',
        inputShape: [9],
        kernelInitializer: 'glorotNormal',
        dtype: FLOATTYPE
    }));
    for (let i = 0; i < 2; i++) {
        model.add(tf.layers.batchNormalization());
        model.add(tf.layers.dense({
            units: 128,
            activation: 'tanh',
            kernelInitializer: 'glorotNormal',
            dtype: FLOATTYPE
        }));
    }
    model.add(tf.layers.batchNormalization());
    model.add(tf.layers.dense({
        units: 9,
        activation: 'tanh',
        kernelInitializer: 'glorotNormal',
        dtype: FLOATTYPE
    }));
}

class DQNAgent {
    constructor(learningRate, discountFactor, explorationRate, drawReward, number) {

        this.model = tf.sequential();
        buildLayers(this.model)

        this.targetModel = tf.sequential();
        buildLayers(this.targetModel)

        this.updateTargetModel();
        this.updateCounter = 0;
        this.targetUpdateFrequency = 500;

        //const optimizer = tf.train.rmsprop(learningRate, 0.95, 0.01, 1e-7);
        const optimizer = tf.train.adam(learningRate);
        this.model.compile({
            optimizer: optimizer,
            loss: 'meanSquaredError'
        });

        this.maxBufferSize = 2000;
        this.batchSize = 32; //2048;

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
        //for (let i = 0; i < trainIterations; i++) {
        //    await this.trainOnBatch();
        //}
    }

    storeExperience(state, action, reward, nextState, discountFactor) {
        const experience = {
            state,
            action,
            reward,
            nextState,
            discountFactor,
            priority: Math.abs(reward)
        };
        
        this.replayBuffer.push(experience);
        if (this.replayBuffer.length > this.maxBufferSize) {
            this.replayBuffer.shift();
        }
    }

    async trainOnBatch2(runs=15, epochs=10) {
        if (this._fitInProgress) {
            if (DEBUG) console.log(`Agent${this.num} _fitInProgress`);
            return;
        } else {
            if (DEBUG) console.log(`Agent${this.num} fitting`);
        }

        this._fitInProgress = true;
        for (let i = 1; i < runs; i++) {

            const indices = Array.from({ length: this.batchSize }, () =>
                Math.floor(Math.random() * this.replayBuffer.length));

            const chosenExps = indices.map(idx => this.replayBuffer[idx]);

            const states = tf.tidy(() => {
                const encodedStates = chosenExps.map(item => encodeState(item.state));
                return tf.concat(encodedStates);
            });

            const targetQs = tf.tidy(() => {
                const predictions = this.targetModel.predict(states);
                const qValues = predictions.arraySync();

                chosenExps.forEach((exp, i) => {
                    qValues[i][exp.action] = exp.reward;
                });

                return tf.tensor2d(qValues);
            });

            await this.model.fit(states, targetQs, {
                epochs: epochs,
                verbose: 0,
                batchSize: Math.min(this.batchSize, this.replayBuffer.length)
            });

            if (i%2 == 0) {
                this.updateTargetModel();
            }

            targetQs.dispose();
            states.dispose();
        }
    }

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
        printBoard(game.board);
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

// Sliders go here, currently in old.js
/*
document.getElementById('train-100').addEventListener('click', () => {
    trainAgent(100);
});
document.getElementById('train-10k').addEventListener('click', () => {
    trainAgent(10000);
});
document.getElementById('train-1m').addEventListener('click', () => {
    trainAgent(1000000);
});
*/

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
    console.time('train');
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
    let trainEvery = 100
    if (numGames <= 100) {
        updatesEvery = numGames;
        numLoops = 1;
    }


    /*
    const games = [
        [4, 1, 0, 2, 8],
        [3, 0, 4, 2, 5],
        [6, 0, 7, 2, 8],
        [0, 1, 3, 2, 6],
        [1, 0, 4, 2, 7],
        [1, 0, 8, 3, 7, 6],
        [1, 3, 8, 0, 2, 6],
        [0, 2, 1, 4, 3, 6],
        [0, 4, 1, 2, 7, 6],
        [1, 8, 4, 7, 0, 6]
    ];

    // Play Forced Games
    for (const moves of games) {
        game.reset();
        for (let i = 0; i < moves.length; i++) {
            game.executeAction(moves[i],
                            i % 2 === 0 ? agent1 : agent2,
                            i % 2 === 0 ? agent2 : agent1);
        }
    }
    agent2.trainOnBatch2();
    */


    for (let j = 0; j < numLoops; j++) {
        for (let i = 0; i < updatesEvery / trainEvery; i++) {
            for (let k = 0; k < trainEvery; k++) {
                agent1.explorationRate = calculateEpsilonDecay(episodes, numGames);
                //agent1.explorationRate = 1.0;
                agent2.explorationRate = calculateEpsilonDecay(episodes, numGames);
                await selfPlay();
            }
            //agent1.trainOnBatch2();
            agent2.trainOnBatch2();
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
    console.timeEnd('train');
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
        //printBoard(game.board);
        currentPlayer *= -1;
    }
}

renderer.render(game);
reapplyColorMode(game.board, game.recentAgent2Board);
