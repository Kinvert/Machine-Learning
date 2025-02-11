// agent.js: A Q-learning agent that depends on the game logic in main.js

const agent = (function() {
    // Preset hyperparameters
    let alpha = 0.25;   // Learning rate
    let gamma = 0.9;    // Discount factor
    let epsilon = 0.25; // Exploration rate

    let trainingSessions = 0;

    // Q-Table and possible states
    let qTable = {};
    let INITIAL_STATES = [];
    let INTERMEDIATE_STATES = [];

    // Generate all possible states with one button filled
    function generateInitialStates() {
        const NUM_ACTIONS = 3;
        return Array.from({length: NUM_ACTIONS}, (_, i) => {
            const state = Array(NUM_ACTIONS).fill('0');
            state[i] = '1';
            return state.join('');
        });
    }

    // Generate all possible states with two buttons filled
    function generateIntermediateStates() {
        const NUM_ACTIONS = 3;
        const states = [];
        for (let i = 0; i < NUM_ACTIONS; i++) {
            for (let j = i + 1; j < NUM_ACTIONS; j++) {
                const state = Array(NUM_ACTIONS).fill('0');
                state[i] = '1';
                state[j] = '1';
                states.push(state.join(''));
            }
        }
        return states;
    }

    // Initialize the Q-table
    function initAgent() {
        INITIAL_STATES = generateInitialStates();
        INTERMEDIATE_STATES = generateIntermediateStates();

        const allStates = [...INITIAL_STATES, ...INTERMEDIATE_STATES];
        allStates.forEach(state => {
            qTable[state] = new Array(3).fill(0);
        });
    }

    // For the UI to retrieve the Q-table
    function getQTable() {
        return qTable;
    }

    // For the UI to retrieve all state strings
    function getAllStates() {
        return [...INITIAL_STATES, ...INTERMEDIATE_STATES];
    }

    // Agent chooses an action via epsilon-greedy
    function getAction(state, explore = true) {
        if (explore && Math.random() < epsilon) {
            return Math.floor(Math.random() * 3); // Explore
        }
        return qTable[state].indexOf(Math.max(...qTable[state])); // Exploit
    }

    // Standard Q-table update: Q(s,a) = (1 - α) * oldQ + α * (reward + γ * maxQ(s'))
    function updateQTable(state, action, reward, nextState) {
        const oldQ = qTable[state][action];
        const nextMaxQ = nextState ? Math.max(...qTable[nextState]) : 0;
        qTable[state][action] = (1 - alpha) * oldQ + alpha * (reward + gamma * nextMaxQ);
    }

    // The training logic: run "numGames" episodes and update the Q-table
    function train(numGames) {
        const NUM_ACTIONS = 3;
        let totalGames = 0;
        let totalWins = 0;
        trainingSessions++;

        for (let game = 1; game <= numGames; game++) {
            totalGames++;

            // Random initial state
            const initialStateIndex = Math.floor(
                Math.random() * INITIAL_STATES.length
            );
            let currentState = INITIAL_STATES[initialStateIndex];
            let won = true;

            // Make (NUM_ACTIONS - 1) moves
            for (let step = 0; step < NUM_ACTIONS - 1; step++) {
                const action = getAction(currentState);

                // If that action is already "1," it's invalid
                if (currentState[action] === '1') {
                    updateQTable(currentState, action, -1, null);
                    won = false;
                    break;
                }

                // Otherwise, apply the move
                const stateArray = currentState.split('');
                stateArray[action] = '1';
                const nextState = stateArray.join('');

                if (step === NUM_ACTIONS - 2) {
                    // Final move
                    updateQTable(currentState, action, won ? 1 : -1, null);
                } else {
                    // Intermediate move
                    updateQTable(currentState, action, 0, nextState);
                    currentState = nextState;
                }
            }
            if (won) {
                totalWins++;
            }
        }

        // Show training stats in the same spot
        const winRate = (totalWins / totalGames) * 100;
        const statsElement = document.getElementById('stats');
        statsElement.innerHTML = `
            Session: ${trainingSessions} Games: ${numGames}:
            Win rate: ${winRate.toFixed(1)}%;<br>
            ` + statsElement.innerHTML;
    }

    // Setters for alpha, gamma, epsilon if the sliders need to update them
    function setAlpha(newAlpha) {
        alpha = newAlpha;
    }
    function setGamma(newGamma) {
        gamma = newGamma;
    }
    function setEpsilon(newEpsilon) {
        epsilon = newEpsilon;
    }

    // Expose a public API
    return {
        initAgent,
        getAction,
        train,
        getQTable,
        getAllStates,
        updateQTable,
        setAlpha,
        setGamma,
        setEpsilon
    };
})();

// Expose “agent” as a global so main.js can reference it.
window.agent = agent;