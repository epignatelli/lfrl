PROMPT_ENV_DESCRIPTION = """\
Briefly and concisely describe the {} environment of MiniHack in a few words.
Mention its goal and its challenges."""

CREDIT_ASSIGNMENT_EXAMPLE = """\
I will present you with a sequence of observations and actions from the gameplay of MiniHack.
Recall that a single observation in MiniHack has four main parts:
a) The number of the timestep and the last action;
b) a *message* appearing on top of the screen;
c) a *grid of symbols* showing the positions of entities in the room and
d) a *set of statistics* at the bottom of the screen.

Write a brief and concise analysis describing the semantics of the trajectory strictly using information from the observations and your knowledge of MiniHack.

Finally, write a score for each action that describes the contribution of that action to progress in the game.
You can choose any score between 0.0 and 1.0, writing {"timestep-t": X}, where t is the timestep of the action and X is the score you choose.
The sum of all scores must be 1.0.
We aim to identify those unique actions that are crucial to the point that without them, the agent would not be able to reach the goal.
"""

COUNTERFACTUAL_EXAMPLE = """\
I will present you with a sequence of observations and actions from the gameplay of MiniHack.
Recall that a single observation in MiniHack has four main parts:
a) The number of the timestep and the last action;
b) a *message* appearing on top of the screen;
c) a *grid of symbols* showing the positions of entities in the room and
d) a *set of statistics* at the bottom of the screen.

The actions available to the agent at each timestep are the following:
- "North": the agent moves north
- "East": the agent moves east
- "South": the agent moves south
- "West": the agent moves west
- "Pickup": the agent picks up an item
- "Drop": the agent drops an item
- "Search": the agent interacts with the item in front of it

Your objective is two-fold:
1. To identify the actions in the trajecotry that are crucial to reach the goal.
2. To evaluate counterfactuals actions: if the agent had taken a different action at a given timestep, would it have reached the goal?

To do this, you will write a score for each action that describes the contribution (influence) of that action to reach the final observation.
Had that action changed, would the agent have reached the goal?
For example, if a different action also would have led to the same observation at the next timestep, then the score of the action is 0.0.
You can choose any score between 0.0 and 1.0, writing {"timestep-t": X}, where t is the timestep of the action and X is the score you choose.
The sum of all scores must be 1.0.
"""

TRAJECTORY_EXAMPLE = """\
- Timestep 0

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


- Timestep 1










                                       ----
                                       |..|
                                       +(.|
                                    ----@.|
                                    |..<..|
                                    |.....|
                                    -------






Agent the Footpad              St:18/03 Dx:17 Co:10 In:12 Wi:11 Ch:8 Chaotic S:
Dlvl:1 $:0 HP:12(12) Pw:2(2) AC:7 Xp:1/0


- Timestep 2










                                       ----
                                       |..|
                                       +(@|
                                    ----..|
                                    |..<..|
                                    |.....|
                                    -------






Agent the Footpad              St:18/03 Dx:17 Co:10 In:12 Wi:11 Ch:8 Chaotic S:
Dlvl:1 $:0 HP:12(12) Pw:2(2) AC:7 Xp:1/0


- Timestep 3

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


- Timestep 4

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


- Timestep 5










                                       ----
                                       |..|
                                       +(.|
                                    ----.@|
                                    |..<..|
                                    |.....|
                                    -------






Agent the Footpad              St:18/03 Dx:17 Co:10 In:12 Wi:11 Ch:8 Chaotic S:
Dlvl:1 $:0 HP:12(12) Pw:2(2) AC:7 Xp:1/0


- Timestep 6

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


- Timestep 7

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


- Timestep 8

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


- Timestep 9

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


- Timestep 10

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

PROMPT_CREDIT_ASSIGNMENT_EXAMPLE = CREDIT_ASSIGNMENT_EXAMPLE + TRAJECTORY_EXAMPLE
PROMPT_COUNTERFACTUAL_EXAMPLE = COUNTERFACTUAL_EXAMPLE + TRAJECTORY_EXAMPLE