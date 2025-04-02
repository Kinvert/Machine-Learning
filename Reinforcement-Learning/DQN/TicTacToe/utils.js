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

function calculateEpsilonDecay(currentEpisode, totalEpisodes) {
    const startEpsilon = 0.995;
    const endEpsilon = 0.2;
    const decay = Math.exp(Math.log(endEpsilon/startEpsilon) / (totalEpisodes * 0.5));
    const eps = Math.max(endEpsilon, startEpsilon * Math.pow(decay, currentEpisode));
    return eps;
}

function printBoard(board) {
    if (DEBUG) {
        const decodedBoard = decodeState(board);
        console.log(`                        ${decodedBoard.slice(0,3)}`);
        console.log(`                        ${decodedBoard.slice(3,6)}`);
        console.log(`                        ${decodedBoard.slice(6,9)}`);
    }
}

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
