# Q-Learning Tic Tac Toe

[:video_game: Live Game](https://kinvert.github.io/Machine-Learning/Reinforcement-Learning/Q-Learning/TicTacToe/index.html)

This is an interactive game of Tic Tac Toe.

Agent1 - Your teammate to give you suggestions, and opponent of Agent2. Takes first turn, is X.

Agent2 - Your opponent to play against, it trains against you and Agent2. Takes second turn, is O.

You can adjust how each agent learns:
- Epsilon = Exploration Rate
- Gamma = Discount Rate
- Alpha = Learning Rate
- Draw $ = Draw Reward

You can color the board according to Agent1's suggestions, Agent2's thinking, or no coloring.

You can choose whether or not to train the Agents on board rotations. Basically every game can have 3 other rotations and remain equivalent. Agents train faster on board rotations, however they look at everything semetrically and it can make the game a bit less interesting to play.

When training and watching the colors of the board change, what you are seeing is the Q Values for an empty board from the perspective of Agent2, and the brighter green the more that move is suggested. That said, if you train enough everything will likely become green since both Agents are smart enough to draw every game eventually.

<img alt="Q-Space" width="280px" src="https://github.com/Kinvert/Machine-Learning/blob/master/Reinforcement-Learning/Q-Learning/TicTacToe/tic-tac-toe-q-learning.png" />