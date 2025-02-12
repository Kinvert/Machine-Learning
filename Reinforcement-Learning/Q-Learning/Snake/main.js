
// Game Constants
const GRID_SIZE = 4;   // Number of grids
const CELL_SIZE = 16;  // Pixels per grid
let GAME_SPEED = 250; // ms pause between frames

// Rewards
const REWARDS = {
  MOVE: -0.01,
  APPLE: 1.0,
  COLLISION: -1.0,
};

// Q-Learning Parameters
let LEARNING_RATE = 0.25;
let DISCOUNT_FACTOR = 0.9;
let EXPLORATION_RATE = 0.1;

// Directions and Actions
const DIRECTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT'];
const ACTIONS = ['FORWARD', 'LEFT', 'RIGHT'];

function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

class SnakeGame {
  constructor() {
    this.reset();
    this.bestScore = 0;
    this.episodes = 0;
    this.wins = 0;
  }

  reset() {
    this.score = 0;
    this.snake = {
      body: [{x: 1, y: 1}],
      direction: 'RIGHT'
    };
    this.snake.body.push({x: 0, y: 1});
    this.apple = this.getRandomPosition();
    this.alive = true;
  }

  getRandomPosition() {
    let position;
    do {
      position = {
        x: Math.floor(Math.random() * GRID_SIZE),
        y: Math.floor(Math.random() * GRID_SIZE)
      };
    } while (this.snake.body.some(segment => 
      segment.x === position.x && segment.y === position.y));
    return position;
  }

  generateGridState() {
    // Initialize 4x4 grid with zeros
    let grid = Array(4).fill().map(() => Array(4).fill(0));

    // Place snake (-2 for head, -1 for body)
    this.snake.body.forEach((segment, index) => {
        grid[segment.y][segment.x] = index === 0 ? -2 : -1;
    });
    
    // Place apple (1)
    grid[this.apple.y][this.apple.x] = 1;
    
    // Flatten the grid to a string
    return grid.flat().join(',');
  }

  getState() {
    let thisState = this.generateGridState();
    const numbers = thisState.split(',').map(Number);
    console.log( `    State: ${numbers.slice(0,4)}
           ${numbers.slice(4,8)}
           ${numbers.slice(8,12)}
           ${numbers.slice(12,16)}\n    Direction: ${game.snake.direction}`);
    return thisState;
  }

  wouldCollide(position) {
    // Wall collision
    if (position.x < 0 || position.x >= GRID_SIZE || 
        position.y < 0 || position.y >= GRID_SIZE) {
      return true;
    }
    
    // Self collision
    return this.snake.body.some(segment => 
      segment.x === position.x && segment.y === position.y);
  }

  executeAction(action) {
    const currentDirIndex = DIRECTIONS.indexOf(this.snake.direction);
    let newDirIndex = currentDirIndex;
    
    // Convert relative action to absolute direction
    if (action === 'LEFT') newDirIndex = (currentDirIndex + 3) % 4;
    if (action === 'RIGHT') newDirIndex = (currentDirIndex + 1) % 4;
    
    this.snake.direction = DIRECTIONS[newDirIndex];
    
    // Move snake
    const head = {...this.snake.body[0]};
    switch(this.snake.direction) {
      case 'UP': head.y--; break;
      case 'DOWN': head.y++; break;
      case 'LEFT': head.x--; break;
      case 'RIGHT': head.x++; break;
    }
    
    // Check collision
    if (this.wouldCollide(head)) {
      this.alive = false;
      return REWARDS.COLLISION;
    }
    
    // Add new head
    this.snake.body.unshift(head);
    
    // Check apple
    if (head.x === this.apple.x && head.y === this.apple.y) {
      this.score++;
      if ( this.score >= 13) {this.wins++; this.reset();}
      this.apple = this.getRandomPosition();
      this.bestScore = Math.max(this.score, this.bestScore);
      return REWARDS.APPLE;
    }
    
    // Remove tail
    this.snake.body.pop();
    return REWARDS.MOVE;
  }
}

class QLearningAgent {
  constructor() {
    this.qTable = new Map();
    this.actions = ACTIONS;
    this.prepopulateStates();
  }

  // Helper to determine direction from body segments
  determineDirection(head, nextSegment) {
    if (head.x > nextSegment.x) return 'RIGHT';
    if (head.x < nextSegment.x) return 'LEFT';
    if (head.y > nextSegment.y) return 'DOWN';
    if (head.y < nextSegment.y) return 'UP';
    return null; // Should never happen with valid snake configurations
  }

  // Shared helper function to calculate Q-values for a given state and direction
  calculateStateQValues(grid2D, head, direction, apple, body, debug = false) {
    const qValues = {};
    
    ACTIONS.forEach(action => {
      let nextPos = {...head};
      let newDirection = direction;

      if (action === 'LEFT') {
        newDirection = DIRECTIONS[(DIRECTIONS.indexOf(direction) + 3) % 4];
      }
      if (action === 'RIGHT') {
        newDirection = DIRECTIONS[(DIRECTIONS.indexOf(direction) + 1) % 4];
      }

      switch(newDirection) {
        case 'UP': nextPos.y--; break;
        case 'DOWN': nextPos.y++; break;
        case 'LEFT': nextPos.x--; break;
        case 'RIGHT': nextPos.x++; break;
      }

      // Check if move is valid
      if (nextPos.x < 0 || nextPos.x >= GRID_SIZE || 
          nextPos.y < 0 || nextPos.y >= GRID_SIZE ||
          (body && body.some(segment => segment.x === nextPos.x && segment.y === nextPos.y))) {
        qValues[action] = -1;
      } 
      // Check if move captures apple
      else if (apple && nextPos.x === apple.x && nextPos.y === apple.y) {
        qValues[action] = 1;
      } 
      // Regular move
      else {
        qValues[action] = 0;
      }
    });

    return qValues;
  }

  // Convert grid string to 2D array and extract positions
  parseGridState(state) {
    const grid = state.split(',').map(Number);
    const grid2D = [];
    let head = null;
    let apple = null;
    const body = [];

    // Convert to 2D array and find positions
    for (let y = 0; y < GRID_SIZE; y++) {
      grid2D[y] = grid.slice(y * GRID_SIZE, (y + 1) * GRID_SIZE);
      for (let x = 0; x < GRID_SIZE; x++) {
        if (grid2D[y][x] === -2) {
          head = {x, y};
          body.unshift({x, y});
        } else if (grid2D[y][x] === -1) {
          body.push({x, y});
        } else if (grid2D[y][x] === 1) {
          apple = {x, y};
        }
      }
    }

    return {grid2D, head, apple, body};
  }

  getQValues(state) {
    if (!this.qTable.has(state)) {
      console.log('');
      console.log('UNKNOWN STATE');
      const {grid2D, head, apple, body} = this.parseGridState(state);
      
      // Determine direction for multi-segment snake
      let direction = game.snake.direction;
      //if (body.length > 1) {
      //  direction = this.determineDirection(body[0], body[1]);
      //}

      const qValues = this.calculateStateQValues(grid2D, head, direction, apple, body);
      console.log('ESTIMATED Q VALUES = ' + JSON.stringify(qValues));
      this.qTable.set(state, qValues);
    }
    
    const qValues = this.qTable.get(state);
    console.log(qValues);
    return qValues;
  }

  generatePossibleBodyPaths(headPos, length, maxLength = 4) {
    const paths = [];
    if (length <= 1) return [[headPos]];
    
    const directions = [
      {x: 0, y: 1},  // up
      {x: 0, y: -1}, // down
      {x: 1, y: 0},  // right
      {x: -1, y: 0}  // left
    ];
    
    for (const dir of directions) {
      const prevPos = {
        x: headPos.x - dir.x,
        y: headPos.y - dir.y
      };
      
      if (prevPos.x >= 0 && prevPos.x < maxLength &&
          prevPos.y >= 0 && prevPos.y < maxLength) {
        
        const subPaths = this.generatePossibleBodyPaths(prevPos, length - 1, maxLength);
        
        for (const subPath of subPaths) {
          const newPath = [headPos, ...subPath];
          let isValid = true;
          const visited = new Set();
          
          for (const pos of newPath) {
            const key = `${pos.x},${pos.y}`;
            if (visited.has(key)) {
              isValid = false;
              break;
            }
            visited.add(key);
          }
          
          if (isValid) {
            paths.push(newPath);
          }
        }
      }
    }
    
    return paths;
  }

  prepopulateStates() {
    const MAXL = 2;

    for (let length = 1; length <= MAXL; length++) {
      for (let headY = 0; headY < 4; headY++) {
        for (let headX = 0; headX < 4; headX++) {
          const possibleBodies = this.generatePossibleBodyPaths({x: headX, y: headY}, length);
          
          for (const body of possibleBodies) {
            for (let appleY = 0; appleY < 4; appleY++) {
              for (let appleX = 0; appleX < 4; appleX++) {
                if (body.some(segment => segment.x === appleX && segment.y === appleY)) {
                  continue;
                }

                let grid = Array(4).fill().map(() => Array(4).fill(0));
                body.forEach((segment, index) => {
                  grid[segment.y][segment.x] = index === 0 ? -2 : -1;
                });
                grid[appleY][appleX] = 1;

                let possibleDirections = DIRECTIONS;
                if (body.length > 1) {
                  const actualDirection = this.determineDirection(body[0], body[1]);
                  possibleDirections = [actualDirection];
                }

                possibleDirections.forEach(direction => {
                  const state = grid.flat().join(',');
                  const qValues = this.calculateStateQValues(
                    grid, 
                    body[0], 
                    direction,
                    {x: appleX, y: appleY},
                    body
                  );

                  // Set Q-values for this state
                  for (const [action, value] of Object.entries(qValues)) {
                    this.setQValue(state, action, value);
                  }
                });
              }
            }
          }
        }
      }
    }
  }

  setQValue(state, action, value, print=false) {
    if (!this.qTable.has(state)) {
      console.log('this state unknown')
      this.qTable.set(state, {});
    }
    if (print) {console.log(`    this.qTable.get(state)[action] ${action} = ` + this.qTable.get(state)[action]);}
    this.qTable.get(state)[action] = value;
    if (print) {console.log(`    this.qTable.get(state)[action] ${action} = ` + this.qTable.get(state)[action]);}
  }

  chooseAction(state) {
    // Exploration
    if (Math.random() < EXPLORATION_RATE) {
      const thisAction = this.actions[Math.floor(Math.random() * this.actions.length)];
      console.log(thisAction);
      return thisAction;
    }
    
    // Exploitation
    let qValues = this.getQValues(state);
    let bestAction = this.actions[0];
    let bestValue = -99;

    for (const action of this.actions) {
      const value = qValues[action];
      if (value > bestValue) {
        bestValue = value;
        bestAction = action;
      }
    }
    console.log(bestAction);
    return bestAction;
  }

  learn(state, action, reward, nextState) {
    console.log('LEARN=====')
    let qValues = this.getQValues(state);
    const currentQ = qValues[action] || 0;

    // Get max Q-value for next state
    let maxNextQ = Math.max(qValues['FORWARD'],qValues['LEFT'],qValues['RIGHT'],0) || 0;

    // Bellman equation
    const newQ = currentQ + LEARNING_RATE * (
      reward + DISCOUNT_FACTOR * maxNextQ - currentQ
    );

    this.setQValue(state, action, newQ, true);
  }
}

// Visualization class
class GameRenderer {
  constructor(canvas) {
    this.ctx = canvas.getContext('2d');
    this.canvas = canvas;
  }

  render(game) {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    
    // Draw apple
    this.ctx.fillStyle = 'red';
    this.ctx.fillRect(
      game.apple.x * CELL_SIZE,
      game.apple.y * CELL_SIZE,
      CELL_SIZE - 1,
      CELL_SIZE - 1
    );
    
    // Draw snake
    this.ctx.fillStyle = 'lime';
    game.snake.body.forEach((segment, i) => {
      this.ctx.fillStyle = i === 0 ? 'lime' : 'green';
      this.ctx.fillRect(
        segment.x * CELL_SIZE,
        segment.y * CELL_SIZE,
        CELL_SIZE - 1,
        CELL_SIZE - 1
      );
    });
  }
}

// Game Loop
const canvas = document.getElementById('game');
const stats = document.getElementById('stats');
const debug = document.getElementById('debug');
const game = new SnakeGame();
const agent = new QLearningAgent();
const renderer = new GameRenderer(canvas);

function updateStats() {
  if (game.wins > 0) { game.bestScore = 13; }
  stats.textContent = `Episodes: ${game.episodes} | Score: ${game.score} | Best: ${game.bestScore} | Wins: ${game.wins}`;
  let updatedState = game.getState();
  const numbers = updatedState.split(',').map(Number);
  const qValues = agent.getQValues(updatedState);
  const roundedQValues = Object.fromEntries(
    Object.entries(qValues).map(([key, value]) => [key, Number(value.toFixed(2))])
  );
  debug.textContent = `State: ${numbers.slice(0,4)}
       ${numbers.slice(4,8)}
       ${numbers.slice(8,12)}
       ${numbers.slice(12,16)}\nDirection: ${game.snake.direction}\nQValue: ${JSON.stringify(roundedQValues)}`;
}

function gameLoop() {
  console.log('================== BEGIN LOOP ======================')
  if (!game.alive) {
    game.episodes++;
    game.reset();
  }

  const currentState = game.getState();
  const action = agent.chooseAction(currentState);
  const reward = game.executeAction(action);
  const nextState = game.getState();
  
  agent.learn(currentState, action, reward, nextState);
  renderer.render(game);

  updateStats();
  console.log('================== END LOOP ======================')
  setTimeout(gameLoop, GAME_SPEED);
}

async function ready(fn) {
    await delay(100);
    if (document.readyState !== 'loading') {
        // Sliders
        const speedSlider = document.getElementById('speed-slider');
        const alphaSlider = document.getElementById('alpha-slider');
        const gammaSlider = document.getElementById('gamma-slider');
        const epsilonSlider = document.getElementById('epsilon-slider');

        const speedValue = document.getElementById('speed-value');
        const alphaValue = document.getElementById('alpha-value');
        const gammaValue = document.getElementById('gamma-value');
        const epsilonValue = document.getElementById('epsilon-value');

        speedSlider.addEventListener('input', () => {
            speedValue.textContent = speedSlider.value;
            GAME_SPEED = parseFloat(speedSlider.value); // Update speed globally
        });

        alphaSlider.addEventListener('input', () => {
            alphaValue.textContent = alphaSlider.value;
            LEARNING_RATE = parseFloat(alphaSlider.value); // Update alpha globally
        });

        gammaSlider.addEventListener('input', () => {
            gammaValue.textContent = gammaSlider.value;
            DISCOUNT_FACTOR = parseFloat(gammaSlider.value); // Update gamma globally
        });

        epsilonSlider.addEventListener('input', () => {
            epsilonValue.textContent = epsilonSlider.value;
            EXPLORATION_RATE = parseFloat(epsilonSlider.value); // Update epsilon globally
        });
    } else {
        document.addEventListener('DOMContentLoaded', fn);
    }
}

ready();

gameLoop();