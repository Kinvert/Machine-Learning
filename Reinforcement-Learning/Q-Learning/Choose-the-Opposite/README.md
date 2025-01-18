# Choose the Opposite

I made this game since I could never get Reinforcement Learning to work. I wanted something simple that was easy to debug, and I knew it had to be easily learnable.

There are 3 boxes/buttons. To initialize the game, one of them is filled/clicked.

The agent's job is to fill all empty boxes, and ONLY fill empty boxes.

So if the initial state is 010, the computer could choose action 100. This would create state 110. Then from that state the agent could choose action 001 creating 111 without creating 210 or 120. 111 is the winning state.

I did this with a simple Q Space and the Bellman Equation.

Q Space:

<div id="q-space" style="grid-template-columns: repeat(3, 50px);"><div class="q-cell" style="background-color: rgb(201, 53, 0);">-0.58</div><div class="q-cell" style="background-color: rgb(83, 171, 0);">0.35</div><div class="q-cell" style="background-color: rgb(12, 242, 0);">0.90</div><div class="q-cell" style="background-color: rgb(120, 134, 0);">0.06</div><div class="q-cell" style="background-color: rgb(183, 71, 0);">-0.44</div><div class="q-cell" style="background-color: rgb(12, 242, 0);">0.90</div><div class="q-cell" style="background-color: rgb(86, 168, 0);">0.32</div><div class="q-cell" style="background-color: rgb(12, 242, 0);">0.90</div><div class="q-cell" style="background-color: rgb(159, 95, 0);">-0.25</div><div class="q-cell" style="background-color: rgb(201, 53, 0);">-0.58</div><div class="q-cell" style="background-color: rgb(183, 71, 0);">-0.44</div><div class="q-cell" style="background-color: rgb(30, 224, 0);">0.76</div><div class="q-cell" style="background-color: rgb(201, 53, 0);">-0.58</div><div class="q-cell" style="background-color: rgb(0, 254, 0);">1.00</div><div class="q-cell" style="background-color: rgb(183, 71, 0);">-0.44</div><div class="q-cell" style="background-color: rgb(0, 254, 0);">1.00</div><div class="q-cell" style="background-color: rgb(201, 53, 0);">-0.58</div><div class="q-cell" style="background-color: rgb(245, 9, 0);">-0.92</div></div>
