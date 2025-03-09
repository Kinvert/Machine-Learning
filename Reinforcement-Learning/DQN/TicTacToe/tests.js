// Test Suite Configuration
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
            }/*,
            {
                name: "FAIL ON PURPOSE Convert empty board",
                run: () => {
                    const input = [0, 0, 0, 1, 0, 0, 0, 0, 0];
                    return {
                        input: input,
                        expected: '---------',
                        result: decodeState(input)
                    };
                }
            }*/
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
            }
        ]
    }
};

// Test Runner
window.onload = function() {
    const summary = document.getElementById('summary');
    const failures = document.getElementById('failures');
    
    let totalTests = 0;
    let totalPassed = 0;
    let allFailures = [];

    // Run all test suites
    Object.entries(testSuites).forEach(([suiteName, suite]) => {
        const suiteResults = suite.tests.map(test => {
            const { input, expected, result } = test.run();
            
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
                    
            return {
                suiteName: suite.name,
                ...test,
                input,
                expected,
                result,
                passed
            };
        });

        totalTests += suiteResults.length;
        totalPassed += suiteResults.filter(t => t.passed).length;
        allFailures = allFailures.concat(suiteResults.filter(t => !t.passed));
    });

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
};