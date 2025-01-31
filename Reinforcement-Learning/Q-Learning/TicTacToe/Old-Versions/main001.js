// Initialize variables
const boardEl = document.getElementById('board');
const statusEl = document.getElementById('status');
const board = Array(9).fill(0);

// Q-Learning Parameters
const discountFactor = 0.95;
let explorationRate = 1.0;
const explorationDecay = 0.995;
const explorationMin = 0.15;

// Experience replay buffer
const replayBuffer = [];
const maxReplayBufferSize = 10000;
const miniBatchSize = 1; // 32

let isTraining = false;
let gameTrajectory = [];

// Create board UI
function renderBoard() {
    boardEl.innerHTML = '';
    board.forEach((value, index) => {
        const cell = document.createElement('div');
        cell.className = 'cell';
        cell.textContent = value === 1 ? 'X' : value === -1 ? 'O' : '';
        if (value === 0) {
            cell.addEventListener('click', () => makeMove(index));
        } else {
            cell.classList.add('taken');
        }
        boardEl.appendChild(cell);
    });
}

function renderSelfPlayBoard(boardState) {
    boardEl.innerHTML = '';
    boardState.forEach((value, index) => {
        const cell = document.createElement('div');
        cell.className = 'cell';
        cell.textContent = value === 1 ? 'X' : value === -1 ? 'O' : '';
        cell.classList.add('taken');
        boardEl.appendChild(cell);
    });
}

function calculateReward(state, action, winner) {
    if (winner === null) {
        // Reward potential 2 in a row
        const potentialLines = getPotentialLines(state, action);
        return potentialLines * 0.05;
    }
    
    if (winner === -1) return 1;  // AI wins
    if (winner === 1) return -1;  // Human wins
    return 0.1;  // Draw is slightly positive
}

function getPotentialLines(state, action) {
    const lines = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],
        [0, 3, 6], [1, 4, 7], [2, 5, 8],
        [0, 4, 8], [2, 4, 6]
    ];
    
    let count = 0;
    const player = -1;  // AI player
    
    for (const line of lines) {
        if (!line.includes(action)) continue;
        
        const values = line.map(i => state[i]);
        const playerCount = values.filter(v => v === player).length;
        const emptyCount = values.filter(v => v === 0).length;
        
        if (playerCount === 2 && emptyCount === 1) count++;
    }
    
    return count;
}

function checkWinner() {
    const winningCombos = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],
        [0, 3, 6], [1, 4, 7], [2, 5, 8],
        [0, 4, 8], [2, 4, 6]
    ];
    for (const combo of winningCombos) {
        const sum = combo.map(i => board[i]).reduce((a, b) => a + b, 0);
        if (sum === 3) return 1;
        if (sum === -3) return -1;
    }
    if (board.every(cell => cell !== 0)) return 0;
    return null;
}

function makeMove(index) {
    if (board[index] !== 0) return;
    board[index] = 1;
    gameTrajectory.push({
        state: [...board],
        action: index,
        nextState: null,
        reward: null
    });
    
    renderBoard();
    const winner = checkWinner();
    if (winner !== null) {
        endGame(winner);
    } else {
        aiMove();
    }
}

async function aiMove() {
    let moveIndex;
    let THINKING = 0;
    if (Math.random() < explorationRate) {
        const availableMoves = board.map((v, i) => v === 0 ? i : -1).filter(i => i !== -1);
        moveIndex = availableMoves[Math.floor(Math.random() * availableMoves.length)];
    } else {
        THINKING = 1;
        await tf.tidy(() => {
            const input = tf.tensor2d([board], [1, 9]);
            const qValues = model.predict(input).dataSync();
            console.log(qValues);
            
            // Filter only valid moves
            //const validMoves = board.map((v, i) => v === 0 ? {index: i, value: qValues[i]} : null)
            //                      .filter(x => x !== null);
            
            // Add noise to Q-values for exploration
            //const noise = 0.01;
            //validMoves.forEach(move => {
            //    move.value += (Math.random() - 0.5) * noise;
            //});
            
            // Select best valid move
            //moveIndex = validMoves.reduce((a, b) => a.value > b.value ? a : b).index;
            const moveIndex = qValues.indexOf(Math.max(...qValues));
            console.log('moveIndex = ' + moveIndex);
            if (board[moveIndex] != 0) {
                // invalid move, immediate loss
                endGame(1);
            } else {
                board[moveIndex] = -1;
            }
        });
    }
    
    board[moveIndex] = -1;
    
    // Update previous state's next state
    if (gameTrajectory.length > 0) {
        gameTrajectory[gameTrajectory.length - 1].nextState = [...board];
    }
    
    gameTrajectory.push({
        state: [...board],
        action: moveIndex,
        nextState: null,
        reward: null
    });

    if ( THINKING ) {
        console.log('got here 2');
    }
    
    renderBoard();
    const winner = checkWinner();
    if (winner !== null) {
        endGame(winner);
    }
}

async function endGame(winner) {
    if (winner === 1) {
        statusEl.textContent = 'You Win!';
    } else if (winner === -1) {
        statusEl.textContent = 'AI Wins!';
    } else {
        statusEl.textContent = 'It\'s a Draw!';
    }
    
    document.querySelectorAll('.cell').forEach(cell => cell.classList.add('taken'));
    
    // Calculate rewards and add to replay buffer
    let cumulativeReward = winner === -1 ? 1 : winner === 1 ? -1 : 0.1; // Start from final reward
    for (let i = 0; i < gameTrajectory.length; i++) {
        const experience = gameTrajectory[i];
        experience.reward = calculateReward(experience.state, experience.action, i === gameTrajectory.length - 1 ? winner : null);
        replayBuffer.push(experience);
    }
    
    // Trim replay buffer if needed
    while (replayBuffer.length > maxReplayBufferSize) {
        replayBuffer.shift();
    }
    
    // Train on mini-batches from replay buffer
    await trainModel();
    
    // Self-play for additional training
    for (let j = 0; j < 100; j++) {
        await selfPlay(10);  // Increased number of self-play games
    }
    
    gameTrajectory = [];
    resetBoard();
}

async function trainModel() {
    if (isTraining || replayBuffer.length < miniBatchSize) return;
    
    isTraining = true;
    
    try {
        for (let epoch = 0; epoch < 5; epoch++) {  // Multiple training epochs
            // Sample mini-batch
            const batch = [];
            for (let i = 0; i < miniBatchSize; i++) {
                const index = Math.floor(Math.random() * replayBuffer.length);
                batch.push(replayBuffer[index]);
            }
            
            const states = batch.map(exp => exp.state);
            const actions = batch.map(exp => exp.action);
            const rewards = batch.map(exp => exp.reward);
            const nextStates = batch.map(exp => exp.nextState || Array(9).fill(0));
            
            const currentQ = await model.predict(tf.tensor2d(states)).array();
            
            const nextQ = await model.predict(tf.tensor2d(nextStates)).array();
            
            const targetQ = currentQ.map((q, i) => {
                const newQ = [...q];
                const nextStateMaxQ = Math.max(...nextQ[i]);
                newQ[actions[i]] = rewards[i] + (nextStates[i].some(v => v !== 0) ? discountFactor * nextStateMaxQ : 0);
                return newQ;
            });
            
            await model.fit(tf.tensor2d(states), tf.tensor2d(targetQ), {
                epochs: 1,
                verbose: 0
            });
        }
        
        updateExplorationRate();
        
    } catch (err) {
        console.error("Error during training:", err);
    } finally {
        isTraining = false;
    }
}

function resetBoard() {
    board.fill(0);
    renderBoard();
    statusEl.textContent = 'Your Turn';
}

async function selfPlay(games) {
    let totalWins = 0;
    const selfPlayTrajectories = [];
    
    for (let game = 0; game < games; game++) {
        let tempBoard = Array(9).fill(0);
        let currentPlayer = 1;
        let trajectory = [];
        
        while (true) {
            let move;
            let LOSE = 0;
            if (Math.random() < explorationRate || currentPlayer === 1) {
                const availableMoves = tempBoard.map((v, i) => v === 0 ? i : -1).filter(i => i !== -1);
                move = availableMoves[Math.floor(Math.random() * availableMoves.length)];
            } else {
                const input = tf.tensor2d([tempBoard], [1, 9]);
                const prediction = model.predict(input).dataSync();
                //const availableMoves = tempBoard.map((v, i) => v === 0 ? {index: i, value: prediction[i]} : null)
                //                               .filter(x => x !== null);
                //move = availableMoves.reduce((a, b) => a.value > b.value ? a : b).index;
                move = prediction.indexOf(Math.max(...prediction));
                //console.log('move = ' + move + ' tempBoard[move] = ' + tempBoard[move]);
                if (tempBoard[move] != 0) {
                    LOSE = 1;
                } else {
                    tempBoard[move] = -1;
                }
            }

            //console.log('player' + currentPlayer + ' does move ' + move);
            if (tempBoard[move] != 0) {
                //console.log(tempBoard);
                //console.log('move = ' + move);
                console.log('INVALID MOVE IN SELF PLAY');
            } else {
                tempBoard[move] = currentPlayer;
            }
            
            // Update previous state's next state
            if (trajectory.length > 0) {
                trajectory[trajectory.length - 1].nextState = [...tempBoard];
            }
            
            if (currentPlayer == -1) {
                trajectory.push({
                    state: [...tempBoard],
                    action: move,
                    nextState: null,
                    reward: null
                });
            }
            
            let winner = checkSelfPlayWinner(tempBoard);
            if (LOSE) {
                winner = 1;
            }
            if (winner !== null) {
                if (winner === -1) totalWins++;
                
                // Calculate rewards and add to replay buffer
                //console.log('winner = ' + winner);
                for (let i = 0; i < trajectory.length; i++) {
                    const experience = trajectory[i];

                    //console.log('(winner === -1 ? 1 : -1) EQUALS ' + (winner === -1 ? 1 : -1) );
                    //console.log('Math.pow(0.5, trajectory.length - 1 - i) EQUALS ' + Math.pow(0.5, trajectory.length - 1 - i));
                    const moveReward = (winner === -1 ? 1 : -1) * Math.pow(0.5, trajectory.length - 1 - i);
                    experience.reward = i === trajectory.length - 1 ? -1*winner : moveReward;

                    //console.log(i);
                    //console.log('experience.state = ' + experience.state.slice(0, 3).join("  "));
                    //console.log('experience.state = ' + experience.state.slice(3, 6).join("  "));
                    //console.log('experience.state = ' + experience.state.slice(6, 9).join("  "));
                    //console.log('experience.action = ' + experience.action);
                    //try {
                    //console.log('experience.nextState = ' + experience.nextState.slice(0, 3).join("  "));
                    //console.log('experience.nextState = ' + experience.nextState.slice(3, 6).join("  "));
                    //console.log('experience.nextState = ' + experience.nextState.slice(6, 9).join("  "));
                    //} catch {}
                    //console.log('experience.reward = ' + experience.reward);
                    //console.log('\n');
                    replayBuffer.push(experience);
                }
                
                selfPlayTrajectories.push(trajectory);
                break;
            }
            
            currentPlayer *= -1;
        }
    }
    
    await trainModel();
    
    console.log(`Self-play win rate: ${(totalWins / games) * 100}%`);
}

function updateExplorationRate() {
    explorationRate = Math.max(explorationRate * explorationDecay, explorationMin);
    console.log('explorationRate = ' + explorationRate);
}

function checkSelfPlayWinner(tempBoard) {
    const winningCombos = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],
        [0, 3, 6], [1, 4, 7], [2, 5, 8],
        [0, 4, 8], [2, 4, 6]
    ];
    for (const combo of winningCombos) {
        const sum = combo.map(i => tempBoard[i]).reduce((a, b) => a + b, 0);
        if (sum === 3) return 1;
        if (sum === -3) return -1;
    }
    if (tempBoard.every(cell => cell !== 0)) return 0;
    return null;
}

let qTable = new Map();

renderBoard();