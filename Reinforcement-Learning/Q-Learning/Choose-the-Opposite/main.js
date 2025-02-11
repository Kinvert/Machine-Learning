// main.js: Game logic, UI, and event handling

// Create the grid of buttons and the Q-space display.
function initializeButtons() {
    const NUM_ACTIONS = 3; // Must match agent.js if you want consistency
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

// Visualize the agent’s Q-table in a grid on the page.
function visualizeQSpace() {
    const NUM_ACTIONS = 3; // Keep consistent with agent.js
    const qSpace = document.getElementById('q-space');
    qSpace.innerHTML = '';
    qSpace.style.gridTemplateColumns = `repeat(${NUM_ACTIONS}, 50px)`;

    // Ask the agent for its Q-table and the set of possible states
    const qTable = agent.getQTable();
    const allStates = agent.getAllStates(); // Combined initial & intermediate

    allStates.forEach(state => {
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

// Delay utility for animations
function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// Clear highlight from all buttons
function resetButtons() {
    const buttons = document.querySelectorAll('.game-button');
    buttons.forEach(button => {
        button.className = 'game-button';
    });
}

// Handles the human play portion
function handleHumanPlay() {
    const NUM_ACTIONS = 3;
    const buttons = document.querySelectorAll('.game-button');
    buttons.forEach((button, index) => {
        button.onclick = async () => {
            resetButtons();
            document.getElementById('game-status').textContent = 'AGENT PLAYING';
            button.classList.add('human-choice');

            let humanPlayState = Array(NUM_ACTIONS).fill('0');
            humanPlayState[index] = '1';
            await delay(500);

            // Computer choices
            const firstAction = agent.getAction(humanPlayState.join(''), false);
            const firstButton = document.getElementById(`button-${firstAction}`);
            let agentLoses = humanPlayState.join('')[parseInt(firstAction)] === '1';
            firstButton.classList.add(agentLoses ? 'wrong-choice' : 'computer-choice');
            if (agentLoses) {
                document.getElementById('game-status').textContent = 'AGENT LOSES';
            }
            await delay(500);

            const nextState = [...humanPlayState];
            nextState[firstAction] = '1';
            const secondAction = agent.getAction(nextState.join(''), false);
            const secondButton = document.getElementById(`button-${secondAction}`);
            agentLoses = agentLoses || nextState.join('')[parseInt(secondAction)] === '1';
            secondButton.classList.add(agentLoses ? 'wrong-choice' : 'computer-choice');

            document.getElementById('game-status').textContent =
                agentLoses ? 'AGENT LOSES' : 'AGENT WINS';

            await delay(1000);
            document.getElementById('game-status').textContent = 'Awaiting player input';
            resetButtons();
        };
    });
}

// Wait for DOM, then wire up the UI
async function ready() {
    await delay(100);
    if (document.readyState !== 'loading') {
        const trainButton = document.getElementById('train-button');
        const numGamesInput = document.getElementById('num-games');

        // Link the agent's train function to the button
        trainButton.addEventListener('click', () => {
            const numGames = parseInt(numGamesInput.value);
            agent.train(numGames);        // The agent trains
            visualizeQSpace();            // Show updated Q-values
        });

        // Slider for alpha
        const alphaSlider = document.getElementById('alpha-slider');
        const alphaValue = document.getElementById('alpha-value');
        alphaSlider.addEventListener('input', () => {
            alphaValue.textContent = alphaSlider.value;
            agent.setAlpha(parseFloat(alphaSlider.value));
        });

        // Slider for gamma
        const gammaSlider = document.getElementById('gamma-slider');
        const gammaValue = document.getElementById('gamma-value');
        gammaSlider.addEventListener('input', () => {
            gammaValue.textContent = gammaSlider.value;
            agent.setGamma(parseFloat(gammaSlider.value));
        });

        // Slider for epsilon
        const epsilonSlider = document.getElementById('epsilon-slider');
        const epsilonValue = document.getElementById('epsilon-value');
        epsilonSlider.addEventListener('input', () => {
            epsilonValue.textContent = epsilonSlider.value;
            agent.setEpsilon(parseFloat(epsilonSlider.value));
        });
    } else {
        document.addEventListener('DOMContentLoaded', () => ready());
    }
}

// Run startup routines
ready();
initializeButtons();
agent.initAgent(); // Initialize Q-table inside the agent
visualizeQSpace();
handleHumanPlay();