
SYSTEM_PROMPT = """\
Welcome to the MiniHack data collection and evaluation interface.
This interface is designed to collect data from MiniHack and evaluate the performance of reinforcement learning agents.
You are a helpful and honest judge of good gameplaying and progress in the NetHack game.
Always answer as helpfully as possible, while being truthful.
If you don't know the answer to a question, please don't share false information.
"""

PROMPT_ENV_DESCRIPTION = """\
Briefly and concisely describe the {} environment of MiniHack in a few words.
Mention its goal and its challenges."""

PROMPT_CREDIT_ASSIGNMENT = """\
I will present you with a sequence of observations and actions from the gameplay of MiniHack.
Recall that a single observation in MiniHack has four main parts:
a) The number of the timestep and the last action;
b) a *message* appearing on top of the screen;
c) a *grid of symbols* showing the positions of entities in the room and
d) a *set of statistics* at the bottom of the screen.

Write a brief and concise analysis describing the semantics of the trajectory strictly using information from the observations and your knowledge of MiniHack.

Finally, write a score for each action that describes the contribution of that action to achieving the goal.
For example, if a different action also would have led to the same observation at the next timestep, then the score of the action is 0.0.
You can choose any score between 0.0 and 1.0, writing {"timestep-t": X}, where t is the timestep of the action and X is the score you choose.
The sum of all scores must be 1.0.
We aim to find those unique actions that are crucial to the point that without them, the agent would not be able to reach the goal.

Timestep 0
Action: northeast

Hello Agent, welcome to NetHack!  You are a chaotic male human Rogue.








                                         --
                                         .|
                                        (.|
                                    ----..|
                                    |..@..|
                                    |.....|
                                    -------






Agent the Footpad              St:18/03 Dx:17 Co:10 In:12 Wi:11 Ch:8 Chaotic S:
Dlvl:1 $:0 HP:12(12) Pw:2(2) AC:7 Xp:1/0
Timestep 1
Action: northeast










                                       ----
                                       |..|
                                       +(.|
                                    ----@.|
                                    |..<..|
                                    |.....|
                                    -------






Agent the Footpad              St:18/03 Dx:17 Co:10 In:12 Wi:11 Ch:8 Chaotic S:
Dlvl:1 $:0 HP:12(12) Pw:2(2) AC:7 Xp:1/0
Timestep 2
Action: east










                                       ----
                                       |..|
                                       +(@|
                                    ----..|
                                    |..<..|
                                    |.....|
                                    -------






Agent the Footpad              St:18/03 Dx:17 Co:10 In:12 Wi:11 Ch:8 Chaotic S:
Dlvl:1 $:0 HP:12(12) Pw:2(2) AC:7 Xp:1/0
Timestep 3
Action: east

It's a wall.








                                       ----
                                       |..|
                                       +(@|
                                    ----..|
                                    |..<..|
                                    |.....|
                                    -------






Agent the Footpad              St:18/03 Dx:17 Co:10 In:12 Wi:11 Ch:8 Chaotic S:
Dlvl:1 $:0 HP:12(12) Pw:2(2) AC:7 Xp:1/0
Timestep 4
Action: south

It's a wall.








                                       ----
                                       |..|
                                       +(@|
                                    ----..|
                                    |..<..|
                                    |.....|
                                    -------






Agent the Footpad              St:18/03 Dx:17 Co:10 In:12 Wi:11 Ch:8 Chaotic S:
Dlvl:1 $:0 HP:12(12) Pw:2(2) AC:7 Xp:1/0
Timestep 5
Action: pickup










                                       ----
                                       |..|
                                       +(.|
                                    ----.@|
                                    |..<..|
                                    |.....|
                                    -------






Agent the Footpad              St:18/03 Dx:17 Co:10 In:12 Wi:11 Ch:8 Chaotic S:
Dlvl:1 $:0 HP:12(12) Pw:2(2) AC:7 Xp:1/0
Timestep 6
Action: northwest

There is nothing here to pick up.








                                       ----
                                       |..|
                                       +(.|
                                    ----.@|
                                    |..<..|
                                    |.....|
                                    -------






Agent the Footpad              St:18/03 Dx:17 Co:10 In:12 Wi:11 Ch:8 Chaotic S:
Dlvl:1 $:0 HP:12(12) Pw:2(2) AC:7 Xp:1/0
Timestep 7
Action: southwest

You see here a key named The Master Key of Thievery.








                                       ----
                                       |..|
                                       +@.|
                                    ----..|
                                    |..<..|
                                    |.....|
                                    -------






Agent the Footpad              St:18/03 Dx:17 Co:10 In:12 Wi:11 Ch:8 Chaotic S:
Dlvl:1 $:0 HP:12(12) Pw:2(2) AC:7 Xp:1/0
Timestep 8
Action: apply

It's a wall.








                                       ----
                                       |..|
                                       +@.|
                                    ----..|
                                    |..<..|
                                    |.....|
                                    -------






Agent the Footpad              St:18/03 Dx:17 Co:10 In:12 Wi:11 Ch:8 Chaotic S:
Dlvl:1 $:0 HP:12(12) Pw:2(2) AC:7 Xp:1/0
Timestep 9
Action: southwest

Never mind.








                                       ----
                                       |..|
                                       +@.|
                                    ----..|
                                    |..<..|
                                    |.....|
                                    -------






Agent the Footpad              St:18/03 Dx:17 Co:10 In:12 Wi:11 Ch:8 Chaotic S:
Dlvl:1 $:0 HP:12(12) Pw:2(2) AC:7 Xp:1/0
Timestep 10
Action:

It's a wall.








                                       ----
                                       |..|
                                       +@.|
                                    ----..|
                                    |..<..|
                                    |.....|
                                    -------






Agent the Footpad              St:18/03 Dx:17 Co:10 In:12 Wi:11 Ch:8 Chaotic S:
Dlvl:1 $:0 HP:12(12) Pw:2(2) AC:7 Xp:1/0
"""