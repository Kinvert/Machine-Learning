// agent.js: Implementation of a DQN agent. 
// The agent's hyperparameters and usage are managed here, 
// while main.js handles the game logic and UI.

const agent = (function() {
    // Hyperparameters for the DQN
    let NUM_ACTIONS = 3;
    let alpha = 0.05;   // Learning rate
    let gamma = 0.9;    // Discount factor
    let epsilon = 0.25; // Exploration rate
    let batchSize = 40; // Replay buffer batch size
    let maxReplaySize = 1000;

    // Model references
    let model;        // Online network
    let targetModel;  // Target network to stabilize training
    let isInitialized = false;
    let isTrainingBatch = false;
    let replayBuffer = [];
    let trainingSessions = 0;

    // States for the game
    let initialStates = [];
    let intermediateStates = [];

    function getNumActions() {
        return NUM_ACTIONS;
    }

    // Create the neural network
    function createModel() {
        const m = tf.sequential();
        m.add(tf.layers.dense({
            units: 24,
            activation: 'relu',
            inputShape: [NUM_ACTIONS]
        }));
        m.add(tf.layers.dense({
            units: NUM_ACTIONS,
            activation: 'linear'
        }));
        m.compile({
            optimizer: tf.train.adam(alpha),
            loss: 'meanSquaredError'
        });
        return m;
    }

    // Generate states with exactly one button pressed
    function generateInitialStates() {
        return Array.from({ length: NUM_ACTIONS }, (_, i) => {
            const state = Array(NUM_ACTIONS).fill(0);
            state[i] = 1;
            return state;
        });
    }

    // Generate states with exactly two buttons pressed
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

    // Initialize the models and states
    async function initialize() {
        if (!isInitialized) {
            model = createModel();
            targetModel = createModel();
            await updateTargetModel();

            initialStates = generateInitialStates();
            intermediateStates = generateIntermediateStates();

            isInitialized = true;
        }
    }

    // Update the target network
    async function updateTargetModel() {
        const weights = model.getWeights();
        targetModel.setWeights(weights);
    }

    // Store a transition in the replay buffer
    function addToReplayBuffer(state, action, reward, nextState) {
        replayBuffer.push({ state, action, reward, nextState });
        if (replayBuffer.length > maxReplaySize) {
            replayBuffer.shift();
        }
    }

    // Sample a batch from the replay buffer and train the model
    async function trainOnBatch() {
        // If we have fewer experiences than batchSize, skip
        if (replayBuffer.length < batchSize) return;

        if (isTrainingBatch) return;
        isTrainingBatch = true;

        const indices = Array.from({ length: batchSize }, () =>
            Math.floor(Math.random() * replayBuffer.length)
        );
        const samples = indices.map(i => replayBuffer[i]);
    
        // Split into terminal and non-terminal
        const validSamples = samples.filter(s => s.nextState !== null);
        const terminalSamples = samples.filter(s => s.nextState === null);

        try {
            // 1) Train on non-terminal transitions
            if (validSamples.length > 0) {
                const { stateTensor, targetTensor } = tf.tidy(() => {
                    // Stack states
                    const stateTensor = tf.stack(validSamples.map(s => tf.tensor1d(s.state)));
                    // Stack nextStates
                    const nextTensor = tf.stack(validSamples.map(s => tf.tensor1d(s.nextState)));
                    const currentQValues = model.predict(stateTensor).arraySync();
                    const nextQValues = targetModel.predict(nextTensor).arraySync();

                    validSamples.forEach((sample, i) => {
                        const maxNextQ = Math.max(...nextQValues[i]);
                        currentQValues[i][sample.action] =
                            sample.reward + gamma * maxNextQ;
                    });
                    return {
                        stateTensor,
                        targetTensor: tf.tensor2d(currentQValues)
                    };
                });
                await model.fit(stateTensor, targetTensor, { epochs: 1, verbose: 0 });

                stateTensor.dispose();
                targetTensor.dispose();
            }

            // 2) Train on terminal transitions
            if (terminalSamples.length > 0) {
                const { stateT2, targetT2 } = tf.tidy(() => {
                    const stateT2 = tf.stack(terminalSamples.map(s => tf.tensor1d(s.state)));
                    const currentQs = model.predict(stateT2).arraySync();

                    terminalSamples.forEach((sample, i) => {
                        currentQs[i][sample.action] = sample.reward;
                    });
                    return {
                        stateT2,
                        targetT2: tf.tensor2d(currentQs)
                    };
                });
                await model.fit(stateT2, targetT2, { epochs: 1, verbose: 0 });

                stateT2.dispose();
                targetT2.dispose();
            }
        } catch (err) {
            console.error('trainOnBatch error:', err);
        } finally {
            isTrainingBatch = false;
        }
    }

    // Epsilon-greedy action selection
    async function getAction(state, explore = true) {
        if (!isInitialized) await initialize();

        if (explore && Math.random() < epsilon) {
            return Math.floor(Math.random() * NUM_ACTIONS);
        }
        // Predict Q-values
        const stateTensor = tf.tensor2d([state], [1, NUM_ACTIONS]);
        const prediction = await model.predict(stateTensor).array();
        stateTensor.dispose();
        return prediction[0].indexOf(Math.max(...prediction[0]));
    }

    // Train the agent for a specified number of games
    async function train(numGames = 50) {
        if (!isInitialized) await initialize();

        let totalWins = 0;
        let totalGames = 0;
        trainingSessions++;

        for (let i = 0; i < numGames; i++) {
            totalGames++;
            // Random initial state
            const idx = Math.floor(Math.random() * initialStates.length);
            let currentState = [...initialStates[idx]];
            let won = true;

            // Number of moves = NUM_ACTIONS - 1
            for (let step = 0; step < NUM_ACTIONS - 1; step++) {
                const action = await getAction(currentState);
                if (currentState[action] === 1) {
                    // Invalid move
                    addToReplayBuffer(currentState, action, -1, null);
                    won = false;
                    break;
                }
                // Make the move
                const nextState = [...currentState];
                nextState[action] = 1;

                if (step === NUM_ACTIONS - 2) {
                    addToReplayBuffer(currentState, action, won ? 1 : -1, null);
                } else {
                    addToReplayBuffer(currentState, action, 0, nextState);
                    currentState = nextState;
                }
            }

            if (won) {
                totalWins++;
            }

            await trainOnBatch();
            if ((i + 1) % 10 === 0) {
                await updateTargetModel();
            }
        }

        const stats = document.getElementById('stats');
        const winRate = (totalWins / totalGames) * 100;
        stats.innerHTML = `
            Session: ${trainingSessions}, Games: ${numGames},
            Win rate: ${winRate.toFixed(1)}%;<br>
        ` + stats.innerHTML;
    }

    // For Q-space visualization, predict Q-values for a given state
    async function predictStateQValues(state) {
        if (!isInitialized) await initialize();
        const stateTensor = tf.tensor2d([state], [1, NUM_ACTIONS]);
        const qVals = await model.predict(stateTensor).array();
        stateTensor.dispose();
        return qVals[0];
    }

    // Return all states (both initial and intermediate)
    function getAllStates() {
        return [...initialStates, ...intermediateStates];
    }

    // These are handy if you need to tweak hyperparameters on the fly
    function setAlpha(newAlpha) {
        alpha = newAlpha;
        model.compile({
            optimizer: tf.train.adam(alpha),
            loss: 'meanSquaredError'
        });
    }
    function setGamma(newGamma) {
        gamma = newGamma;
    }
    function setEpsilon(newEpsilon) {
        epsilon = newEpsilon;
    }
    function setBatchSize(newSize) {
        batchSize = newSize;
    }

    // Return public methods
    return {
        initialize,
        getNumActions,
        train,
        getAction,
        getAllStates,
        predictStateQValues,
        setAlpha,
        setGamma,
        setEpsilon,
        setBatchSize
    };
})();

// Expose as global so main.js can use "agent"
window.agent = agent;