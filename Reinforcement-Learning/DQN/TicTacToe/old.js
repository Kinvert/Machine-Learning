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