const NUM_ACTIONS = 3; // Number of buttons
let alpha = 0.05;  // Learning rate
let gamma = 0.9;   // Discount factor
let epsilon = 0.25; // Exploration rate
let batchSize = 40; // Replay buffer batch size

let trainingSessions = 0;
let model;
let targetModel;
let replayBuffer = [];
const MAX_REPLAY_BUFFER_SIZE = 1000;
let isInitialized = false;
let isTrainingBatch = false; // Prevents concurrent fit() calls

// Generate all possible states with one button filled
function generateInitialStates() {
    return Array.from({length: NUM_ACTIONS}, (_, i) => {
        const state = Array(NUM_ACTIONS).fill(0);
        state[i] = 1;
        return state;
    });
}

// Generate all possible states with two buttons filled
function generateIntermediateStates() {
    const states = [];
    for (let i = 0; i < NUM_ACTIONS; i++) {
        for (let j = i + 1; j < NUM_ACTIONS; j++) {
            const state = Array(NUM_ACTIONS).fill(0);
            state[i] = 1;
            state[j] = 1;
            states.push(state);
        }
    }
    return states;
}

const INITIAL_STATES = generateInitialStates();
const INTERMEDIATE_STATES = generateIntermediateStates();

// Create model
function createModel() {
    const model = tf.sequential();
    model.add(tf.layers.dense({
        units: 24,
        activation: 'relu',
        inputShape: [NUM_ACTIONS]
    }));
    model.add(tf.layers.dense({
        units: NUM_ACTIONS,
        activation: 'linear'
    }));
    model.compile({
        optimizer: tf.train.adam(alpha),
        loss: 'meanSquaredError'
    });
    return model;
}

// Initialize models
async function initializeModels() {
    if (!isInitialized) {
        model = createModel();
        targetModel = createModel();
        await updateTargetModel();
        isInitialized = true;
    }
}

// Update target model
async function updateTargetModel() {
    const weights = model.getWeights();
    targetModel.setWeights(weights);
}

function addToReplayBuffer(state, action, reward, nextState) {
    replayBuffer.push({ state, action, reward, nextState });
    if (replayBuffer.length > MAX_REPLAY_BUFFER_SIZE) {
        replayBuffer.shift();
    }
}

async function trainOnBatch() {
    // Skip if not enough data
    if (replayBuffer.length < batchSize) return;

    // Prevent overlapping fits
    if (isTrainingBatch) return;
    isTrainingBatch = true;

    const indices = Array.from({length: batchSize}, () =>
        Math.floor(Math.random() * replayBuffer.length)
    );
    const samples = indices.map(i => replayBuffer[i]);

    // Split the samples
    const validSamples = samples.filter(s => s.nextState !== null);
    const terminalSamples = samples.filter(s => s.nextState === null);

    try {
        // 1) Train on non-null nextState
        if (validSamples.length > 0) {
            // Gather states and updated Q-values inside a tidy
            const { states, updatedQ } = tf.tidy(() => {
                const states = tf.stack(validSamples.map(s => tf.tensor1d(s.state)));
                const nextStates = tf.stack(validSamples.map(s => tf.tensor1d(s.nextState)));
                const currentQ = model.predict(states);
                const targetQ = targetModel.predict(nextStates);

                const currentQValues = currentQ.arraySync();
                const targetQValues = targetQ.arraySync();

                validSamples.forEach((sample, i) => {
                    currentQValues[i][sample.action] =
                        sample.reward + gamma * Math.max(...targetQValues[i]);
                });

                const updatedQ = tf.tensor2d(currentQValues);
                return { states, updatedQ };
            });

            // Fit outside tidying, so we can await it
            await model.fit(states, updatedQ, { epochs: 1, verbose: 0 });

            states.dispose();
            updatedQ.dispose();
        }

        // 2) Train on null nextState
        if (terminalSamples.length > 0) {
            const { statesTerm, updatedQTerm } = tf.tidy(() => {
                const statesTerm = tf.stack(terminalSamples.map(s => tf.tensor1d(s.state)));
                const currentQTerm = model.predict(statesTerm);
                const currentQArray = currentQTerm.arraySync();

                terminalSamples.forEach((sample, i) => {
                    // For terminal states, the nextState is null, so Q = reward
                    currentQArray[i][sample.action] = sample.reward;
                });

                const updatedQTerm = tf.tensor2d(currentQArray);
                return { statesTerm, updatedQTerm };
            });

            // Again, fit is outside the tidy
            await model.fit(statesTerm, updatedQTerm, { epochs: 1, verbose: 0 });

            statesTerm.dispose();
            updatedQTerm.dispose();
        }
    } catch (err) {
        console.error('Error in trainOnBatch:', err);
    } finally {
        isTrainingBatch = false;
    }
}

async function getAction(state, explore = true) {
    if (!isInitialized) await initializeModels();
    
    if (explore && Math.random() < epsilon) {
        return Math.floor(Math.random() * NUM_ACTIONS); // Explore
    }
    
    const stateTensor = tf.tensor2d([state], [1, NUM_ACTIONS]);
    const prediction = await model.predict(stateTensor).array();
    stateTensor.dispose();
    return prediction[0].indexOf(Math.max(...prediction[0])); // Exploit
}

async function visualizeQSpace() {
    if (!isInitialized) await initializeModels();
    
    const qSpace = document.getElementById('q-space');
    qSpace.innerHTML = '';
    qSpace.style.gridTemplateColumns = `repeat(${NUM_ACTIONS}, 50px)`;

    const states = [...INITIAL_STATES, ...INTERMEDIATE_STATES];
    
    for (const state of states) {
        const stateTensor = tf.tensor2d([state], [1, NUM_ACTIONS]);
        const qValues = await model.predict(stateTensor).array();
        stateTensor.dispose();
        
        qValues[0].forEach(value => {
            const cell = document.createElement('div');
            cell.className = 'q-cell';
            const normalizedValue = (value + 1) / 2;
            const red = Math.floor((1 - normalizedValue) * 255);
            const green = Math.floor(normalizedValue * 255);
            cell.style.backgroundColor = `rgb(${red}, ${green}, 0)`;
            cell.textContent = value.toFixed(2);
            qSpace.appendChild(cell);
        });
    }
}

function initializeButtons() {
    const buttonGrid = document.getElementById('buttons');
    buttonGrid.style.gridTemplateColumns = `repeat(${NUM_ACTIONS}, 100px)`;
    for (let i = 0; i < NUM_ACTIONS; i++) {
        const button = document.createElement('button');
        button.className = 'game-button';
        button.id = `button-${i}`;
        button.textContent = i;
        buttonGrid.appendChild(button);
    }
}

async function train(numGames = batchSize) {
    if (!isInitialized) await initializeModels();
    
    let totalGames = 0;
    let totalWins = 0;
    trainingSessions++;

    for (let game = 1; game <= numGames; game++) {
        totalGames++;
        
        const initialStateIndex = Math.floor(Math.random() * INITIAL_STATES.length);
        let currentState = [...INITIAL_STATES[initialStateIndex]];
        let moves = [];
        let won = true;

        for (let step = 0; step < NUM_ACTIONS - 1; step++) {
            const action = await getAction(currentState);
            moves.push(action);
            
            if (currentState[action] === 1) {
                addToReplayBuffer(currentState, action, -1, null);
                won = false;
                break;
            }

            const nextState = [...currentState];
            nextState[action] = 1;

            if (step === NUM_ACTIONS - 2) {
                addToReplayBuffer(currentState, action, won ? 1 : -1, null);
            } else {
                addToReplayBuffer(currentState, action, 0, nextState);
                currentState = nextState;
            }
        }

        if (won) totalWins++;
        
        await trainOnBatch();
        if (game % 10 === 0) await updateTargetModel();
    }

    const winRate = (totalWins / totalGames) * 100;
    document.getElementById('stats').innerHTML =
        `Session: ${trainingSessions} Game: ${numGames}: Win rate: ${winRate.toFixed(1)}%;<br>` +
        document.getElementById('stats').innerHTML;
    await visualizeQSpace();
}

function resetButtons() {
    const buttons = document.querySelectorAll('.game-button');
    buttons.forEach(button => {
        button.className = 'game-button';
    });
}

async function handleHumanPlay() {
    if (!isInitialized) await initializeModels();
    
    const buttons = document.querySelectorAll('.game-button');
    buttons.forEach((button, index) => {
        button.onclick = async () => {
            resetButtons();
            document.getElementById('game-status').textContent = 'AGENT PLAYING';
            button.classList.add('human-choice');
            let humanPlayState = Array(NUM_ACTIONS).fill(0);
            humanPlayState[index] = 1;
            await delay(500);

            // Computer choices
            const firstAction = await getAction(humanPlayState, false);
            const firstButton = document.getElementById(`button-${firstAction}`);
            let agentLoses = humanPlayState[parseInt(firstAction)] === 1;
            firstButton.classList.add(agentLoses ? 'wrong-choice' : 'computer-choice');
            if (agentLoses) {
                document.getElementById('game-status').textContent = 'AGENT LOSES';
            }
            await delay(500);

            const nextState = [...humanPlayState];
            nextState[firstAction] = 1;
            const secondAction = await getAction(nextState, false);
            const secondButton = document.getElementById(`button-${secondAction}`);
            agentLoses = agentLoses || nextState[parseInt(secondAction)] === 1;
            secondButton.classList.add(agentLoses ? 'wrong-choice' : 'computer-choice');

            if (agentLoses) {
                document.getElementById('game-status').textContent = 'AGENT LOSES';
            } else {
                document.getElementById('game-status').textContent = 'AGENT WINS';
            }
            await delay(1000);
            
            document.getElementById('game-status').textContent = 'Awaiting player input';
            resetButtons();
        };
    });
}

function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function ready(fn) {
    await delay(200);
    if (document.readyState !== 'loading') {
        console.log('[READY] DOM content loaded');
        await initializeModels();

        const header = document.querySelector('h2');

        const trainButton = document.getElementById('train-button');
        const numGamesInput = document.getElementById('num-games');
        trainButton.addEventListener('click', async () => {
            const numGames = parseInt(numGamesInput.value);
            header.textContent = 'Training...';
            await train(numGames);
            header.textContent = 'PLAY HERE';
        });

        const batchSlider = document.getElementById('batch-slider');
        const alphaSlider = document.getElementById('alpha-slider');
        const gammaSlider = document.getElementById('gamma-slider');
        const epsilonSlider = document.getElementById('epsilon-slider');

        const batchValue = document.getElementById('batch-value');
        const alphaValue = document.getElementById('alpha-value');
        const gammaValue = document.getElementById('gamma-value');
        const epsilonValue = document.getElementById('epsilon-value');

        batchSlider.addEventListener('input', () => {
            batchValue.textContent = batchSlider.value;
            batchSize = parseFloat(batchSlider.value);
        });

        alphaSlider.addEventListener('input', () => {
            alphaValue.textContent = alphaSlider.value;
            alpha = parseFloat(alphaSlider.value);
            model.compile({
                optimizer: tf.train.adam(alpha),
                loss: 'meanSquaredError'
            });
        });

        gammaSlider.addEventListener('input', () => {
            gammaValue.textContent = gammaSlider.value;
            gamma = parseFloat(gammaSlider.value);
        });

        epsilonSlider.addEventListener('input', () => {
            epsilonValue.textContent = epsilonSlider.value;
            epsilon = parseFloat(epsilonSlider.value);
        });
    } else {
        document.addEventListener('DOMContentLoaded', fn);
    }
}

ready();
initializeButtons();
visualizeQSpace();
handleHumanPlay();