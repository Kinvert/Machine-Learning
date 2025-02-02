const NUM_ACTIONS = 3; // Number of buttons
let alpha = 0.25;  // Learning rate
let gamma = 0.9;  // Discount factor
let epsilon = 0.25; // Exploration rate

let trainingSessions = 0;

// Generate all possible states with one button filled
function generateInitialStates() {
    return Array.from({length: NUM_ACTIONS}, (_, i) => {
        const state = Array(NUM_ACTIONS).fill('0');
        state[i] = '1';
        return state.join('');
    });
}

// Generate all possible states with two buttons filled
function generateIntermediateStates() {
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

const INITIAL_STATES = generateInitialStates();
const INTERMEDIATE_STATES = generateIntermediateStates();

// Initialize Q-table
let qTable = {};
[...INITIAL_STATES, ...INTERMEDIATE_STATES].forEach(state => {
    qTable[state] = new Array(NUM_ACTIONS).fill(0);
});

let humanPlayState = Array(NUM_ACTIONS).fill('0').join('');
let humanButtons = document.querySelectorAll('.game-button');

function getAction(state, explore = true) {
    if (explore && Math.random() < epsilon) {
        return Math.floor(Math.random() * NUM_ACTIONS); // Explore
    }
    return qTable[state].indexOf(Math.max(...qTable[state])); // Exploit
}

function updateQTable(state, action, reward, nextState) {
    // Q(s,a) = (1-α)Q(s,a) + α(R + γ*max(Q(s')))
    const oldQ = qTable[state][action];
    const nextMaxQ = nextState ? Math.max(...qTable[nextState]) : 0;
    qTable[state][action] = (1 - alpha) * oldQ + alpha * (reward + gamma * nextMaxQ);
}

function visualizeQSpace() {
    const qSpace = document.getElementById('q-space');
    qSpace.innerHTML = '';
    qSpace.style.gridTemplateColumns = `repeat(${NUM_ACTIONS}, 50px)`;

    [...INITIAL_STATES, ...INTERMEDIATE_STATES].forEach(state => {
        qTable[state].forEach(value => {
            const cell = document.createElement('div');
            cell.className = 'q-cell';
            const normalizedValue = (value + 1) / 2;
            const red = Math.floor((1 - normalizedValue) * 255);
            const green = Math.floor(normalizedValue * 255);
            cell.style.backgroundColor = `rgb(${red}, ${green}, 0)`;
            cell.textContent = value.toFixed(2);
            qSpace.appendChild(cell);
        });
    });
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

function train(numGames = NUM_GAMES) {
    let totalGames = 0;
    let totalWins = 0;
    trainingSessions++;

    for (let game = 1; game <= numGames; game++) {
        totalGames++;
        
        // Random initial state
        const initialStateIndex = Math.floor(Math.random() * INITIAL_STATES.length);
        let currentState = INITIAL_STATES[initialStateIndex];
        let moves = [];
        let won = true;

        // Make NUM_ACTIONS-1 moves
        for (let step = 0; step < NUM_ACTIONS - 1; step++) {
            const action = getAction(currentState);
            moves.push(action);
            
            if (currentState[action] === '1') {
                // Invalid move
                updateQTable(currentState, action, -1, null);
                won = false;
                break;
            }

            // Make move
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
    const winRate = (totalWins / totalGames) * 100;
    document.getElementById('stats').innerHTML = `Session: ${trainingSessions} Game: ${numGames}: Win rate: ${winRate.toFixed(1)}%;<br>` + document.getElementById('stats').innerHTML;
    visualizeQSpace();
}

function resetButtons() {
    const buttons = document.querySelectorAll('.game-button');
    buttons.forEach(button => {
        button.className = 'game-button';
    });
}

function handleHumanPlay() {
    const buttons = document.querySelectorAll('.game-button');
    buttons.forEach((button, index) => {
        button.onclick = async () => {
            resetButtons();
            document.getElementById('game-status').textContent = 'AGENT PLAYING'
            button.classList.add('human-choice');
            humanPlayState = Array(NUM_ACTIONS).fill('0');
            humanPlayState[index] = '1';
            await delay(500);

            // Computer choices
            const firstAction = getAction(humanPlayState.join(''), false);
            const firstButton = document.getElementById(`button-${firstAction}`);
            let agentLoses = humanPlayState.join('')[parseInt(firstAction)] == '1';
            firstButton.classList.add(agentLoses ? 'wrong-choice' : 'computer-choice');
            if (agentLoses) {
                document.getElementById('game-status').textContent = 'AGENT LOSES'
            }
            await delay(500);

            const nextState = [...humanPlayState];
            nextState[firstAction] = '1';
            const secondAction = getAction(nextState.join(''), false);
            const secondButton = document.getElementById(`button-${secondAction}`);
            agentLoses = agentLoses || nextState.join('')[parseInt(secondAction)] == '1';
            secondButton.classList.add(agentLoses ? 'wrong-choice' : 'computer-choice');

            if (agentLoses) {
                document.getElementById('game-status').textContent = 'AGENT LOSES';
            } else{
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

function setupTrainButton() {
    const trainButton = document.getElementById('train-button');
    const numGamesInput = document.getElementById('num-games');

    trainButton.addEventListener('click', () => {
        console.log('train clicked');
        const numGames = parseInt(numGamesInput.value, 10) || 100;
        train(numGames);
        visualizeQSpace();
    });
}

async function ready(fn) {
    await delay(100);
    if (document.readyState !== 'loading') {
        // Train Button
        const trainButton = document.getElementById('train-button');
        const numGamesInput = document.getElementById('num-games');
        trainButton.addEventListener('click', () => {
            const numGames = parseInt(numGamesInput.value);
            train(numGames);
        });

        // Sliders
        const alphaSlider = document.getElementById('alpha-slider');
        const gammaSlider = document.getElementById('gamma-slider');
        const epsilonSlider = document.getElementById('epsilon-slider');

        const alphaValue = document.getElementById('alpha-value');
        const gammaValue = document.getElementById('gamma-value');
        const epsilonValue = document.getElementById('epsilon-value');

        alphaSlider.addEventListener('input', () => {
            alphaValue.textContent = alphaSlider.value;
            alpha = parseFloat(alphaSlider.value); // Update alpha globally
        });

        gammaSlider.addEventListener('input', () => {
            gammaValue.textContent = gammaSlider.value;
            gamma = parseFloat(gammaSlider.value); // Update gamma globally
        });

        epsilonSlider.addEventListener('input', () => {
            epsilonValue.textContent = epsilonSlider.value;
            epsilon = parseFloat(epsilonSlider.value); // Update epsilon globally
        });
    } else {
        document.addEventListener('DOMContentLoaded', fn);
    }
}

ready();

initializeButtons();
visualizeQSpace();

handleHumanPlay();