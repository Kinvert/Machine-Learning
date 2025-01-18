# Choose the Opposite

[TRY IT LIVE](https://kinvert.github.io/Machine-Learning/Reinforcement-Learning/Q-Learning/Choose-the-Opposite/index.html)

I made this game since I could never get Reinforcement Learning to work. I wanted something simple that was easy to debug, and I knew it had to be easily learnable.

There are 3 boxes/buttons. To initialize the game, one of them is filled/clicked.

The agent's job is to fill all empty boxes, and ONLY fill empty boxes.

So if the initial state is 010, the computer could choose action 100. This would create state 110. Then from that state the agent could choose action 001 creating 111 without creating 210 or 120. 111 is the winning state.

I did this with a simple Q Space and the Bellman Equation.

Q Space:

<img alt="Q-Space" width="280px" src="https://github.com/Kinvert/Machine-Learning/blob/master/Reinforcement-Learning/Q-Learning/Choose-the-Opposite/Q-Space.png" />
