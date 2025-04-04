const testSuites = {
    decodeState: {
        name: 'DecodeState Tests',
        tests: [
            {
                name: "Convert board with X's, O's and empty spaces",
                run: () => {
                    const input = [1, -1, 0, 1, 0, -1, 0, 0, 1];
                    return {
                        input: input,
                        expected: 'XO-X-O--X',
                        result: decodeState(input)
                    };
                }
            },
            {
                name: "Convert empty board",
                run: () => {
                    const input = [0, 0, 0, 0, 0, 0, 0, 0, 0];
                    return {
                        input: input,
                        expected: '---------',
                        result: decodeState(input)
                    };
                }
            }
        ]
    },

    encodeState: {
        name: 'EncodeState Tests',
        tests: [
            {
                name: "Convert string with X's, O's and dashes",
                run: () => {
                    const input = 'XO-X-O--X';
                    const expected = [1, -1, 0, 1, 0, -1, 0, 0, 1];
                    const result = Array.from(encodeState(input).dataSync());
                    return {
                        input: input,
                        expected: expected,
                        result: result
                    };
                }
            },
            {
                name: "Convert empty board string",
                run: () => {
                    const input = '---------';
                    const expected = [0, 0, 0, 0, 0, 0, 0, 0, 0];
                    const result = Array.from(encodeState(input).dataSync());
                    return {
                        input: input,
                        expected: expected,
                        result: result
                    };
                }
            }
        ]
    },
    
    calculateEpsilonDecay: {
        name: 'CalculateEpsilonDecay Tests',
        tests: [
            {
                name: "Start of training (episode 0)",
                run: () => {
                    const input = { currentEpisode: 0, totalEpisodes: 1000 };
                    return {
                        input: `episode: ${input.currentEpisode}, total: ${input.totalEpisodes}`,
                        expected: 0.995,
                        result: calculateEpsilonDecay(input.currentEpisode, input.totalEpisodes)
                    };
                }
            },
            {
                name: "End of training",
                run: () => {
                    const input = { currentEpisode: 1000, totalEpisodes: 1000 };
                    return {
                        input: `episode: ${input.currentEpisode}, total: ${input.totalEpisodes}`,
                        expected: 0.2,
                        result: calculateEpsilonDecay(input.currentEpisode, input.totalEpisodes)
                    };
                }
            }
        ]
    },

    dqnAgent: {
        name: 'DQN Agent Tests',
        tests: [
            {
                name: "Transform board array using rotation map",
                run: () => {
                    const agent = new DQNAgent(0.001, 0.95, 0.995, -0.5, 1);
                    const board = [1, 2, 3, 4, 5, 6, 7, 8, 9];
                    return {
                        input: `board: [${board}], rotationMap: [${rotation90Map}]`,
                        expected: [7, 4, 1, 8, 5, 2, 9, 6, 3],
                        result: agent.transformBoard(board, rotation90Map)
                    };
                }
            },
            {
                name: "Transform empty board using rotation map",
                run: () => {
                    const agent = new DQNAgent(0.001, 0.95, 0.995, -0.5, 1);
                    const board = [0, 0, 0, 0, 0, 0, 0, 0, 0];
                    const rotationMap = [2, 5, 8, 1, 4, 7, 0, 3, 6]; // 90 degree rotation
                    return {
                        input: `board: [${board}], rotationMap: [${rotationMap}]`,
                        expected: [0, 0, 0, 0, 0, 0, 0, 0, 0],
                        result: agent.transformBoard(board, rotationMap)
                    };
                }
            },
            {
                name: "Play Forced Game - Test Blocking Failure",
                async: true,
                run: async () => {
                    const agent1 = new DQNAgent(0.005, 0.95, 0.0, -0.5, 1);
                    const agent2 = new DQNAgent(0.005, 0.95, 0.0, -0.5, 2);
                    const batchSize = 4;

                    const games = [
                        [4, 1, 0, 2, 8],
                        [3, 0, 4, 2, 5],
                        [6, 0, 7, 2, 8],
                        [0, 1, 3, 2, 6],
                        [1, 0, 4, 2, 7],
                        [1, 0, 8, 3, 7, 6],
                        [1, 3, 8, 0, 2, 6],
                        [0, 2, 1, 4, 3, 6],
                        [0, 4, 1, 2, 7, 6],
                        [1, 8, 4, 7, 0, 6]
                    ];

                    await playForcedGames(agent1, agent2, games);
                    await trainAgent(agent2, batchSize, 15);

                    const losingStates = [ //                              !
                        [1, -1, 0, 0, 1, 0, 0, 0, 0],  // from game: 4 1 0 2 8
                        [0, -1, 0, 1, 1, 0, 0, 0, 0],  // from game: 3 0 4 2 5
                        [-1, 0, 0, 0, 0, 0, 1, 1, 0],  // from game: 6 0 7 2 8
                        [1, -1, 0, 1, 0, 0, 0, 0, 0],  // from game: 0 1 3 2 6
                        [-1, 1, 0, 0, 1, 0, 0, 0, 0]   // from game: 1 0 4 2 7
                    ];

                    const winningStates = [ //                                 !
                        [-1, 1, 0, -1, 0, 0, 0, 1, 1], // from game: 1 0 8 3 7 6
                        [-1, 1, 1, -1, 0, 0, 0, 0, 1], // from game: 1 3 8 0 2 6
                        [1, 1, -1, 1, -1, 0, 0, 0, 0], // from game: 0 2 1 4 3 6
                        [1, 1, -1, 0, -1, 0, 0, 1, 0], // from game: 0 4 1 2 7 6
                        [1, 1, 0, 0, 1, 0, 0, -1, -1]  // from game: 1 8 4 7 0 6
                    ];

                    const avgLosingValue = await evaluateAgent(agent2, losingStates, 2, 'losing');
                    const avgWinningValue = await evaluateAgent(agent2, winningStates, 6, 'winning');

                    console.log('Average Q-value for losing moves:', avgLosingValue);
                    console.log('Average Q-value for winning moves:', avgWinningValue);

                    return {
                        input: `Losing states: ${losingStates.length}, Winning states: ${winningStates.length}`,
                        expected: true,
                        result: avgLosingValue < 0 && avgWinningValue > 0,
                        debug: { avgLosingValue, avgWinningValue }
                    };
                }
            }
        ]
    }
};

async function playForcedGames(agent1, agent2, games) {
    const game = new TicTacToeGame();
    for (const moves of games) {
        game.reset();
        for (let i = 0; i < moves.length; i++) {
            game.executeAction(moves[i],
                            i % 2 === 0 ? agent1 : agent2,
                            i % 2 === 0 ? agent2 : agent1);
        }
    }
}

async function trainAgent(agent, batchSize, iterations) {
    for (let i = 1; i < iterations; i++) {
        const indices = Array.from({ length: batchSize }, () =>
            Math.floor(Math.random() * agent.replayBuffer.length));

        const chosenExps = indices.map(idx => agent.replayBuffer[idx]);
        const states = tf.tidy(() => {
            const encodedStates = chosenExps.map(item => encodeState(item.state));
            return tf.concat(encodedStates);
        });

        const targetQs = tf.tidy(() => {
            const predictions = agent.model.predict(states);
            const qValues = predictions.arraySync();
            chosenExps.forEach((exp, i) => {
                qValues[i][exp.action] = exp.reward;
            });
            return tf.tensor2d(qValues);
        });

        await modelFitOperation(agent, states, targetQs, batchSize);

        agent.updateTargetModel();
        targetQs.dispose();
        states.dispose();
    }
}

async function modelFitOperation(agent, states, targetQs, batchSize) {
    return await agent.model.fit(states, targetQs, {
        epochs: 10,
        verbose: 0,
        batchSize: Math.min(batchSize, agent.replayBuffer.length)
    });
}

async function evaluateAgent(agent, states, moveIndex, stateType) {
    let totalValue = 0;
    const statesTensor = tf.tidy(() => {
        const encodedStates = states.map(state => tf.tensor2d([state], [1, 9]));
        return tf.concat(encodedStates);
    });
    const allQValues = await agent.model.predict(statesTensor).arraySync();
    for (let i = 0; i < states.length; i++) {
        totalValue += allQValues[i][moveIndex];
    }
    statesTensor.dispose();
    return totalValue / states.length;
}

window.onload = async function() {
    const startTime = performance.now();
    const summary = document.getElementById('summary');
    const failures = document.getElementById('failures');
    
    let totalTests = 0;
    let totalPassed = 0;
    let allFailures = [];

    // Run all test suites
    for (const [suiteName, suite] of Object.entries(testSuites)) {
        for (const test of suite.tests) {
            totalTests++;
            
            try {
                // Handle both async and sync tests
                const { input, expected, result, debug } = test.async ?
                    await test.run() :
                    test.run();

                // Different comparison logic based on type
                let passed;
                if (Array.isArray(expected)) {
                    passed = expected.length === result.length &&
                            expected.every((val, idx) => Math.abs(val - result[idx]) < 0.0001);
                } else if (typeof expected === 'number') {
                    passed = Math.abs(result - expected) < 0.0001;
                } else {
                    passed = result === expected;
                }

                if (passed) {
                    totalPassed++;
                } else {
                    allFailures.push({
                        suiteName: suite.name,
                        ...test,
                        input,
                        expected,
                        result,
                        debug
                    });
                }
            } catch (error) {
                console.error(`Test failed with error: ${error}`);
                allFailures.push({
                    suiteName: suite.name,
                    ...test,
                    error: error.message
                });
            }
        }
    }

    // Display summary
    summary.innerHTML = `
        <div class="${totalPassed === totalTests ? 'pass' : 'fail'}">
            ${totalPassed} of ${totalTests} tests passed
        </div>
    `;

    // Display failures if any
    if (allFailures.length > 0) {
        failures.innerHTML = '<h3 class="fail">Failed Tests:</h3>';
        allFailures.forEach(test => {
            failures.innerHTML += `
                <p class="fail">
                    ❌ ${test.suiteName} - ${test.name}<br>
                    Input: ${test.input}<br>
                    Expected: ${test.expected}<br>
                    Got: ${test.result}
                </p>
            `;
        });
    }
    const endTime = performance.now();
    const totalTimeSeconds = ((endTime - startTime) / 1000).toFixed(3);
    console.log(`Total testing time: ${totalTimeSeconds} seconds`);
};
