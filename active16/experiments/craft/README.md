# The craft environment

The 'map' folder contains one standard map. Each grid position might contain a resource, workstation, wall, or being empty. A brief explanation of each symbol follows:

- 'A' is the agent
- 'X' is a wall
- 'a' is a tree or wood
- 'b' is a string
- 'c' is a factory or workbench (a place where player assembles parts)
- 'd' is grass
- 'e' is a piece of stone
- 'f' is iron
- 'g' is gold
- 'h' is gem

The 'reward_machines' folder contains 10 tasks for this environment. These tasks are based on the 10 tasks defined by [Andreas et al.](https://arxiv.org/abs/1611.01796) for the crafting environment. The 'tests' folder contains 11 testing scenarios.<br>
Each test is associated with one map and includes the path to the 10 tasks defined for the craft environment. It also includes the optimal number of steps needed to solve each task in the given map (we precomputed them using value iteration).<br>
We use the optimal number of steps to normalize the discounted rewards in our experiments.<br>
Example, t4.txt is building a bridge. The agent must first collect a tree (a) and a iron (f) in any order. Then they must go to a workbench (c)<br>
The 'options' folder is only used by the Hierarchical RL baselines. It defines a set of sensible options to tackle the tasks defined for this domain.<br>
<br>Note, that t1 to t5.txt are left as legacy from the Andreas paper mentioned above. t6 and onwards, were created for the Active Learning paper.<br>
<br> Note, we have changed the meaning of some symbols from their original definitions for the Active Learning Paper
<br> Below, we detail what each task means:<br>
t8.txt: Build a hammer. Get string (b),stone (e),iron(f), stone, and assemble in factory(c)<br>
t11.txt: Build a spear. Get stone, string, wood(a), iron, and assemble in factory.



