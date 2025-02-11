// main.js: The game logic, UI, and event handling. 
// It relies on agent.js for the DQN agent functionality.

function initializeButtons() {
    const NUM_ACTIONS = 3; 
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

// Provide some visual representation of Q-values learned by the agent
async function visualizeQSpace() {
    const qSpace = document.getElementById('q-space');
    qSpace.innerHTML = '';
    qSpace.style.gridTemplateColumns = `repeat(${agent.getNumActions()}, 50px)`;

    const allStates = agent.getAllStates();
    for (const state of allStates) {
        // Request Q-values from the DQN model
        const qValues = await agent.predictStateQValues(state);
        qValues.forEach(value => {
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

// Reset highlights from all buttons
function resetButtons() {
    const buttons = document.querySelectorAll('.game-button');
    buttons.forEach(button => {
        button.className = 'game-button';
    });
}

// Let a human player click a button and see the agent respond
async function handleHumanPlay() {
    const buttons = document.querySelectorAll('.game-button');
    buttons.forEach((button, index) => {
        button.onclick = async () => {
            resetButtons();
            document.getElementById('game-status').textContent = 'AGENT PLAYING';
            button.classList.add('human-choice');

            // We store the click in an array representing the state
            const state = Array(agent.getNumActions()).fill(0);
            state[index] = 1;

            await delay(500);

            // Agent picks first action
            const action1 = await agent.getAction(state, false);
            const btn1 = document.getElementById(`button-${action1}`);
            let agentLoses = state[action1] === 1;
            btn1.classList.add(agentLoses ? 'wrong-choice' : 'computer-choice');
            if (agentLoses) {
                document.getElementById('game-status').textContent = 'AGENT LOSES';
            }
            await delay(500);

            // Agent picks second action
            const nextState = [...state];
            nextState[action1] = 1;
            const action2 = await agent.getAction(nextState, false);
            const btn2 = document.getElementById(`button-${action2}`);
            agentLoses = agentLoses || nextState[action2] === 1;
            btn2.classList.add(agentLoses ? 'wrong-choice' : 'computer-choice');

            document.getElementById('game-status').textContent = agentLoses
                ? 'AGENT LOSES'
                : 'AGENT WINS';

            await delay(1000);
            document.getElementById('game-status').textContent = 'Awaiting player input';
            resetButtons();
        };
    });
}

// Helper delay function
function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// Train the agent
async function train(numGames) {
    const header = document.querySelector('h2');
    header.textContent = 'Training...';
    await agent.train(numGames);
    header.textContent = 'PLAY HERE';
    await visualizeQSpace();
}

// Wait for DOM
async function ready() {
    await delay(200);
    if (document.readyState !== 'loading') {
        console.log('[READY] DOM content loaded');
        await agent.initialize(); // Make sure the model is ready

        const trainButton = document.getElementById('train-button');
        const numGamesInput = document.getElementById('num-games');

        trainButton.addEventListener('click', async () => {
            const numGames = parseInt(numGamesInput.value) || 50;
            await train(numGames);
        });

        // Sliders for alpha, gamma, epsilon, batchSize
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
            agent.setBatchSize(parseFloat(batchSlider.value));
        });

        alphaSlider.addEventListener('input', () => {
            alphaValue.textContent = alphaSlider.value;
            agent.setAlpha(parseFloat(alphaSlider.value));
        });

        gammaSlider.addEventListener('input', () => {
            gammaValue.textContent = gammaSlider.value;
            agent.setGamma(parseFloat(gammaSlider.value));
        });

        epsilonSlider.addEventListener('input', () => {
            epsilonValue.textContent = epsilonSlider.value;
            agent.setEpsilon(parseFloat(epsilonSlider.value));
        });
    } else {
        document.addEventListener('DOMContentLoaded', () => ready());
    }
}

// Startup
ready();
initializeButtons();
visualizeQSpace();
handleHumanPlay();