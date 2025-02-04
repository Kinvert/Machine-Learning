const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');
const WIDTH = canvas.width;
const HEIGHT = canvas.height;

window.targetXs = [300, 500];
window.targetYs = [250, 350];

let cannonX = 50;
let cannonY = HEIGHT - 50;
let cannonAngleDeg = 45;
let cannonVelocity = 50;
let projectileX = cannonX;
let projectileY = cannonY;
let projectileVx = 0;
let projectileVy = 0;

const GRAVITY = 50;

let targetX = window.targetXs[0];
let targetY = window.targetYs[0];
const targetRadius = 15;

let closestDist = 10000;

let isFlying = false;
let isGameOver = false;

let DT = 0.02; // Larger is less accurate but faster
let currentTime = 0;
const maxFlightTime = 10;

/**
 * Resets all key variables to their initial values
 * 
 * Randomly selects a new (targetX, targetY) from the global arrays window.targetXs, window.targetYs.
 * Useful in Reinforcement Learning to begin a new episode with a fresh environment state.
 */
function resetGame(){
    cannonAngleDeg = 45;
    cannonVelocity = 50;

    cannonX = 50;
    cannonY = HEIGHT - 50;

    projectileX = cannonX;
    projectileY = cannonY;
    projectileVx = 0;
    projectileVy = 0;
    isFlying = false;
    isGameOver = false;
    currentTime = 0;

    closestDist = 10000;

    const xIndex = Math.floor(Math.random() * window.targetXs.length);
    const yIndex = Math.floor(Math.random() * window.targetYs.length);
    targetX = window.targetXs[xIndex];
    targetY = window.targetYs[yIndex];

    drawGame();
}

// For an RL agent to set angle/velocity
function setAction(angleDeg, velocity) {
    cannonAngleDeg = angleDeg;
    cannonVelocity = velocity;
    //console.log(`Calc = ${closestApproach(targetX, targetY, velocity, angleDeg, g=50)}`);
}

function stepGame(headless = false) {
    if (isGameOver) return;

    if (isFlying) {
        currentTime += DT;
        projectileVy += GRAVITY * DT;
        projectileX += projectileVx * DT;
        projectileY += projectileVy * DT;

        if (
            projectileX < 0 ||
            projectileX > targetX + targetRadius ||
            projectileY > HEIGHT ||
            currentTime > maxFlightTime
        ) {
            isGameOver = true;
        }

        const dx = projectileX - targetX;
        const dy = projectileY - targetY;
        const dist = Math.sqrt(dx * dx + dy * dy);

        if (dist < closestDist) {
            closestDist = dist;
        }
        if (dist < targetRadius) {
            isGameOver = true;
        }
    }

    if (!headless) { drawGame(); }
}

function fireCannon() {
    if (isFlying) return;

    const angleRad = (Math.PI / 180) * cannonAngleDeg;
    projectileVx = cannonVelocity * Math.cos(angleRad);
    projectileVy = -cannonVelocity * Math.sin(angleRad);

    projectileX = cannonX;
    projectileY = cannonY;

    isFlying = true;
    isGameOver = false;
    currentTime = 0;
}

function getState() {
    return {
        cannonAngleDeg,
        cannonVelocity,
        projectileX,
        projectileY,
        isFlying,
        isGameOver,
        targetX,
        targetY
    };
}

function getReward() {
    const dx = projectileX - targetX;
    const dy = projectileY - targetY;
    const dist = Math.sqrt(dx * dx + dy * dy);
    if (isGameOver && dist < targetRadius) {
        return 100;
    }
    return -1 * closestDist;
}
/*
function closestApproach(x_t, y_t, v0, angle, g = 9.81) { // doesnt work
    // Convert angle to radians
    let theta = angle * Math.PI / 180;

    // Components of initial velocity
    let vx0 = v0 * Math.cos(theta);
    let vy0 = v0 * Math.sin(theta);

    // Quadratic coefficients for dD²/dt = 0
    let a = vx0 ** 2 + vy0 ** 2 + g ** 2 / 4;
    let b = -2 * (x_t * vx0 + y_t * vy0 - vy0 * g);
    let c = x_t ** 2 + y_t ** 2;

    // Solve quadratic equation: a*t² + b*t + c = 0
    let discriminant = b ** 2 - 4 * a * c;

    if (discriminant < 0) {
        return Math.sqrt(c); // If no real solution, return initial distance
    }

    let t1 = (-b + Math.sqrt(discriminant)) / (2 * a);
    let t2 = (-b - Math.sqrt(discriminant)) / (2 * a);

    // We take the smallest non-negative time
    let t_min = Math.min(t1, t2);
    if (t_min < 0) t_min = Math.max(t1, t2);
    if (t_min < 0) return Math.sqrt(c); // If no valid time, return initial distance

    // Compute closest point (x_min, y_min)
    let x_min = vx0 * t_min;
    let y_min = vy0 * t_min - 0.5 * g * t_min ** 2;

    // Compute and return closest distance
    return Math.sqrt((x_min - x_t) ** 2 + (y_min - y_t) ** 2);
}
*/
function drawGame() {
    ctx.clearRect(0, 0, WIDTH, HEIGHT);

    // Cannon
    ctx.save();
    ctx.fillStyle = "lightgreen";
    ctx.fillRect(cannonX - 10, cannonY, 20, 10);
    const barrelLen = 30;
    const angleRad = (Math.PI / 180) * cannonAngleDeg;
    const endX = cannonX + barrelLen * Math.cos(angleRad);
    const endY = cannonY - barrelLen * Math.sin(angleRad);
    ctx.beginPath();
    ctx.moveTo(cannonX, cannonY);
    ctx.lineTo(endX, endY);
    ctx.strokeStyle = "lightgreen";
    ctx.lineWidth = 3;
    ctx.stroke();
    ctx.closePath();
    ctx.restore();

    // Projectile
    if (isFlying || !isGameOver) {
        ctx.beginPath();
        ctx.arc(projectileX, projectileY, 5, 0, 2 * Math.PI);
        ctx.fillStyle = "yellow";
        ctx.fill();
        ctx.closePath();
    }

    // Target
    ctx.beginPath();
    ctx.arc(targetX, targetY, targetRadius, 0, 2 * Math.PI);
    ctx.fillStyle = "red";
    ctx.fill();
    ctx.closePath();

    // Info text
    ctx.fillStyle = "#fff";
    ctx.font = "16px sans-serif";
    ctx.fillText(`Angle: ${cannonAngleDeg.toFixed(1)}°`, 10, 20);
    ctx.fillText(`Velocity: ${cannonVelocity.toFixed(1)}`, 10, 40);
    ctx.fillText(`Projectile X: ${projectileX.toFixed(1)}, Y: ${projectileY.toFixed(1)}`, 10, 60);
    ctx.fillText(`Target X: ${targetX.toFixed(1)}, Y: ${targetY.toFixed(1)}`, 10, 80);
    ctx.fillText(`Closest Dist: ${closestDist.toFixed(2)}`, 10, 100);
}

// Optional manual controls
document.addEventListener("keydown", (e) => {
    if (e.key === "ArrowUp") {
        cannonAngleDeg++;
    } else if (e.key === "ArrowDown") {
        cannonAngleDeg--;
    } else if (e.key === "ArrowRight") {
        cannonVelocity++;
    } else if (e.key === "ArrowLeft") {
        cannonVelocity = Math.max(1, cannonVelocity - 1);
    } else if (e.key === " ") {
        fireCannon();
    } else if (e.key === "r") {
        resetGame();
    }
    drawGame();
});

function mainLoop() {
    stepGame();
    requestAnimationFrame(mainLoop);
}

function valueToColor(qValue) {
    const MAXVAL = 100, MINVAL = -100;
    let val = Math.max(MINVAL, Math.min(MAXVAL, qValue));
    const ratio = (val - MINVAL) / (MAXVAL - MINVAL);
    const red = Math.round((1 - ratio) * 255);
    const green = Math.round(ratio * 255);
    return `rgb(${red},${green},0)`;
}

resetGame();
mainLoop();