const targetQs = tf.tidy(() => {
    // Split batch into terminal and non-terminal states
    const terminalStates = rotatedBatch.filter(exp => !exp.nextState);
    const nonTerminalStates = rotatedBatch.filter(exp => exp.nextState);
    
    // Process terminal states (they only need rewards)
    const terminalQs = this.model.predict(
        tf.tensor2d(terminalStates.map(exp => exp.state), [terminalStates.length, 9])
    ).arraySync();
    
    // Update terminal states with just rewards
    terminalStates.forEach((exp, i) => {
        terminalQs[i][exp.action] = exp.reward;
    });
    
    // Process non-terminal states
    let nonTerminalQs = [];
    if (nonTerminalStates.length > 0) {
        const currentPredictions = this.model.predict(
            tf.tensor2d(nonTerminalStates.map(exp => exp.state), [nonTerminalStates.length, 9])
        );
        
        const nextStatePredictions = this.targetModel.predict(
            tf.tensor2d(nonTerminalStates.map(exp => exp.nextState), [nonTerminalStates.length, 9])
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
//new
//vs
//old
const targetQs = tf.tidy(() => {
    // Get current predictions for all states
    const currentQs = this.model.predict(states).arraySync();
    
    // Get target network predictions for next states
    let nextQs = [];
    if (nextStates !== null) {
        nextQs = this.targetModel.predict(nextStates).arraySync();
    }
    
    let nextStateIdx = 0;
    for (let i = 0; i < rotatedBatch.length; i++) {
        const { action, reward, nextState } = rotatedBatch[i];
        
        if (nextState) {
            // Use target network for next state Q-values
            const maxNextQ = Math.max(...nextQs[nextStateIdx]);
            currentQs[i][action] = reward + this.discountFactor * maxNextQ;
            nextStateIdx++;
        } else {
            // Terminal state - just use the reward
            currentQs[i][action] = reward;
        }
    }
    return tf.tensor2d(currentQs, [rotatedBatch.length, 9]);
});












// =======================================================================
// SLIDERS
// =======================================================================

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



