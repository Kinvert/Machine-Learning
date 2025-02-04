/*
  agent.js
  A single-file RL agent for the artillery game, with an optional built-in Q-space
  drawing function. Zero-valued Q cells appear as black (or dark grey).

  Environment Requirements (index.html must define):
    1) Global arrays window.targetXs, window.targetYs for discrete target positions
    2) Functions: resetGame(), setAction(angle, velocity), fireCannon(), stepGame()
    3) Functions: getState() => { targetX, targetY, isGameOver, ... }
                  getReward() => numeric reward
  Usage:
    1) Include <script src="agent.js"></script> after your main game script in index.html.
    2) If you want Q-space visuals, add <div id="qSpaceContainer"></div> in your HTML.
    3) The agent auto-runs after a short delay. Watch the console logs or see the Q-visuals
       if #qSpaceContainer is present.
*/

/* global resetGame, setAction, fireCannon, stepGame, getState, getReward */

(function () {
    // Discrete sets of angles and velocities for possible actions
    const angleOptions = [30, 40, 50, 60, 70];
    const velocityOptions = [150, 200, 250, 300];

    // Discrete target positions
    const targetXs = window.targetXs || [300, 600];
    const targetYs = window.targetYs || [250, 400];

    // Q will be stored in a Map with key = "x,y" => 2D array for Q
    // Q[x,y][angleIndex][velocityIndex]
    let QTable = new Map();

    // Hyperparameters
    let EPISODES = 30;
    const MAX_STEPS = 1000;
    const ALPHA = 0.3;    // learning rate
    const GAMMA = 0.9;    // discount factor
    const EPSILON = 0.9;  // exploration probability

    // ========== Q-Table Initialization ==========
    function initQ() {
        QTable.clear();

        for (let tx of targetXs) {
            for (let ty of targetYs) {
                const qArray = [];
                for (let a = 0; a < angleOptions.length; a++) {
                    qArray[a] = [];
                    for (let v = 0; v < velocityOptions.length; v++) {
                        qArray[a][v] = 0; // start at 0
                    }
                }
                const key = `${tx},${ty}`;
                QTable.set(key, qArray);
            }
        }
    }

    // Helper to retrieve or initialize the Q-array for a given target
    function getQForTarget(tx, ty) {
        const key = `${tx},${ty}`;
        if (!QTable.has(key)) {
            const qArray = [];
            for (let a = 0; a < angleOptions.length; a++) {
                qArray[a] = [];
                for (let v = 0; v < velocityOptions.length; v++) {
                    qArray[a][v] = 0;
                }
            }
            QTable.set(key, qArray);
        }
        return QTable.get(key);
    }

    function chooseLessExplored(qArray) {
        const zeroActions = [];
            // Track the best (least negative) action among actions < 0
            let bestNegVal = -10000, bestA = 0, bestV = 0;

            for (let a = 0; a < angleOptions.length; a++) {
                for (let v = 0; v < velocityOptions.length; v++) {
                    const val = qArray[a][v];
                    if (val === 0) {
                        zeroActions.push({ aIndex: a, vIndex: v });
                    } else if (val < 0 && val > bestNegVal) {
                        bestNegVal = val;
                        bestA = a;
                        bestV = v;
                    }
                }
            }

            // If any zero-valued actions exist, pick one randomly
            if (zeroActions.length > 0) {
                return zeroActions[Math.floor(Math.random() * zeroActions.length)];
            }

            // Otherwise pick the least negative
            //return { aIndex: bestA, vIndex: bestV };
            return chooseBest(qArray);
    }

    function chooseBest(qArray) {
        let bestVal = -10000;
        let bestA = 0;
        let bestV = 0;
        for (let a = 0; a < angleOptions.length; a++) {
            for (let v = 0; v < velocityOptions.length; v++) {
                if (qArray[a][v] > bestVal) {
                    bestVal = qArray[a][v];
                    bestA = a;
                    bestV = v;
                }
            }
        }
        return { aIndex: bestA, vIndex: bestV };
    }

    // ========== Action Selection (ε-greedy) ==========
    function chooseAction(qArray) {
        if (Math.random() < EPSILON) {
            return chooseLessExplored(qArray);
        }
        return chooseBest(qArray);
    }

    // ========== Single-step Q-learning Update ==========
    function updateQ(qArray, aIndex, vIndex, reward) {
        const oldVal = qArray[aIndex][vIndex];
        const newVal = oldVal + ALPHA * (reward - oldVal);
        qArray[aIndex][vIndex] = newVal;
    }

    // ========== Main Training Loop ==========
    async function runAgentTraining(episodes) {
        initQ();

        for (let episode = 0; episode < episodes; episode++) {
            resetGame();
            let done = false;
            let steps = 0;
            let totalReward = 0;

            const initState = getState();
            const { targetX, targetY } = initState;
            const qArray = getQForTarget(targetX, targetY);

            // Choose an action
            const { aIndex, vIndex } = chooseAction(qArray);
            const angleDeg = angleOptions[aIndex];
            const velocity = velocityOptions[vIndex];

            // Perform the shot
            setAction(angleDeg, velocity);
            fireCannon();

            while (!done && steps < MAX_STEPS) {
                let headless = false;
                stepGame(headless = headless);

                const sNow = getState();
                done = sNow.isGameOver;
                if (done) {
                    totalReward = getReward();
                }
                steps++;

                // mild delay for visual
                if (!headless) { await new Promise(res => setTimeout(res, 0)); }
            }

            // Update Q
            updateQ(qArray, aIndex, vIndex, totalReward);
            if (episode % 100 == 0) {
                console.log(
                    `Episode ${episode + 1}, ` +
                    `Target=(${targetX},${targetY}), ` +
                    `Angle=${angleDeg}, Vel=${velocity}, ` +
                    `Reward=${totalReward.toFixed(2)}`
                );
            }

            // After each episode, draw Q-space if a container is present
            drawQSpaceIfPresent();
        }

        console.log("Training complete. Check QTable in the console if needed.");
    }

    // ========== Q-Space Visualization (internal to agent) ==========
    // We'll look for an element with id="qSpaceContainer". If missing, we do nothing.
    function drawQSpaceIfPresent() {
        const container = document.getElementById("qSpaceContainer");
        if (!container) return; // No container => skip visualization

        // Build a 4D array to represent Q:
        // Q4D[txIndex][tyIndex][aIndex][vIndex]
        const Q4D = [];
        for (let txI = 0; txI < targetXs.length; txI++) {
            Q4D[txI] = [];
            for (let tyI = 0; tyI < targetYs.length; tyI++) {
                const key = `${targetXs[txI]},${targetYs[tyI]}`;
                const qArr = QTable.get(key) || [];
                const plane = [];
                for (let a = 0; a < angleOptions.length; a++) {
                    plane[a] = [];
                    for (let v = 0; v < velocityOptions.length; v++) {
                        plane[a][v] = qArr[a][v] || 0;
                    }
                }
                Q4D[txI][tyI] = plane;
            }
        }

        container.innerHTML = "";

        for (let txI = 0; txI < targetXs.length; txI++) {
            for (let tyI = 0; tyI < targetYs.length; tyI++) {
                const stateWrapper = document.createElement("div");
                stateWrapper.style.border = "1px solid #aaa";
                stateWrapper.style.margin = "5px";
                stateWrapper.style.display = "inline-block";

                const header = document.createElement("div");
                header.style.background = "#444";
                header.style.fontSize = "14px";
                header.style.padding = "4px";
                header.textContent = `Target=(${targetXs[txI]}, ${targetYs[tyI]})`;
                stateWrapper.appendChild(header);

                const table = document.createElement("table");
                table.style.borderCollapse = "collapse";

                // Column header row
                const thead = document.createElement("thead");
                const headRow = document.createElement("tr");
                const blankTh = document.createElement("th");
                blankTh.textContent = "";
                blankTh.style.border = "1px solid #666";
                blankTh.style.padding = "4px";
                blankTh.style.background = "#222";
                headRow.appendChild(blankTh);

                for (let v = 0; v < velocityOptions.length; v++) {
                    const th = document.createElement("th");
                    th.textContent = `V=${velocityOptions[v]}`;
                    th.style.border = "1px solid #666";
                    th.style.padding = "4px";
                    th.style.background = "#222";
                    headRow.appendChild(th);
                }
                thead.appendChild(headRow);
                table.appendChild(thead);

                // Body
                const tbody = document.createElement("tbody");
                for (let a = 0; a < angleOptions.length; a++) {
                    const row = document.createElement("tr");

                    // Row label
                    const rowHeader = document.createElement("th");
                    rowHeader.textContent = `A=${angleOptions[a]}`;
                    rowHeader.style.border = "1px solid #666";
                    rowHeader.style.padding = "4px";
                    rowHeader.style.background = "#222";
                    row.appendChild(rowHeader);

                    for (let v = 0; v < velocityOptions.length; v++) {
                        const td = document.createElement("td");
                        td.style.border = "1px solid #666";
                        td.style.padding = "4px";
                        td.style.minWidth = "40px";

                        const qVal = Q4D[txI][tyI][a][v];
                        td.textContent = qVal.toFixed(1);

                        td.style.backgroundColor = valueToColor(qVal);

                        row.appendChild(td);
                    }
                    tbody.appendChild(row);
                }

                table.appendChild(tbody);
                stateWrapper.appendChild(table);
                container.appendChild(stateWrapper);
            }
        }
    }

    // Adjust color so zero is dark
    function valueToColor(val) {
        const MINVAL = -100, MAXVAL = 100;
        const clamped = Math.max(MINVAL, Math.min(MAXVAL, val));

        // If val is near zero, dark grey or black
        const epsilon = 1e-2;
        if (Math.abs(clamped) < epsilon) {
            return "#000"; // black for near zero
        }

        // otherwise do a linear blend from red to green
        const ratio = (clamped - MINVAL) / (MAXVAL - MINVAL);
        const red = Math.round((1 - ratio) * 255);
        const green = Math.round(ratio * 255);
        return `rgb(${red},${green},0)`;
    }

    window.runAgentTraining = runAgentTraining;

    setTimeout(() => {
        runAgentTraining(200);
    }, 100);
})();