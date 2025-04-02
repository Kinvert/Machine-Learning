class GameRenderer {
    constructor(boardEl, statsEl, debugEl) {
        this.boardEl = boardEl;
        this.statsEl = statsEl;
        this.debugEl = debugEl;
    }

    render(game) {
        this.boardEl.innerHTML = '';
        const decodedBoard = decodeState(game.board);
        [...decodedBoard].forEach((value, index) => {
            const cell = document.createElement('div');
            cell.className = 'cell';
            cell.id = `cell${index}`;

            cell.textContent = value === 'X' ? 'X' : value === 'O' ? 'O' : '';

            if (value === '-') {
                cell.addEventListener('click', () => makeMove(index));
            } else {
                cell.classList.add('taken');
            }

            this.boardEl.appendChild(cell);
        });

        const oldLine = document.getElementById('winLine');
        if (oldLine) {
            oldLine.remove();
        }

        if (game.winningCombo) {
            this.drawWinLine(game.winningCombo);
        }
    }

    drawWinLine(combo) {
        const line = document.createElement('div');
        line.id = 'winLine';
        line.style.position = 'absolute';
        line.style.backgroundColor = 'blue';
        line.style.height = '6px';
        line.style.width = '0px';
        line.style.transformOrigin = '0 50%';

        const cellSize = 100;
        const cellGap = 5;
        
        // Get actual grid position within the centered board
        const firstCell = this.boardEl.firstElementChild;
        const gridRect = firstCell.getBoundingClientRect();
        const boardRect = this.boardEl.getBoundingClientRect();
        
        const gridOffsetX = gridRect.left - boardRect.left;
        const gridOffsetY = gridRect.top - boardRect.top;

        const start = combo[0];
        const end = combo[2];

        const startRow = Math.floor(start / 3);
        const startCol = start % 3;
        const endRow   = Math.floor(end / 3);
        const endCol   = end % 3;

        const x1 = startCol * (cellSize + cellGap) + cellSize / 2 + gridOffsetX;
        const y1 = startRow * (cellSize + cellGap) + cellSize / 2 + gridOffsetY;
        const x2 = endCol   * (cellSize + cellGap) + cellSize / 2 + gridOffsetX;
        const y2 = endRow   * (cellSize + cellGap) + cellSize / 2 + gridOffsetY;

        const dx = x2 - x1;
        const dy = y2 - y1;
        const length = Math.sqrt(dx * dx + dy * dy);
        const angle = Math.atan2(dy, dx) * (180 / Math.PI);

        // Correct positioning inside the board
        line.style.left = `${x1}px`;
        line.style.top  = `${y1}px`;
        line.style.width = `${length}px`;
        line.style.transform = `rotate(${angle}deg)`;

        this.boardEl.appendChild(line);
    }

    updateStats(episodes, wins2, draws2, losses2) {
        this.statsEl.textContent = `Agent2 Episodes: ${episodes} | Wins: ${wins2} | Draws: ${draws2} | Losses: ${losses2}`;
    }
}
