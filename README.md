# RL-Escape-Game
A Reinforcement-Learning based game in which an agent learns how to escape a castle with guards. This is part of the Graduate Foundation of AI course at Northeastern (CS5100). In the MFMC.py file, the Q_learning() function executes a specified number of episodes and has the agent follow an ε-greedy policy. 
$η = \frac{1}{1 + number of updates to Q_{opt(s,a)}}\\
{}\\
Estimate, Q_{opt(s,a)} = (1 −η)Q^{(t−1)}_{opt(s,a)} + η[R(s,a,s′) + γ ˆV (t−1)
opt (s′)]
where ˆVopt(s) = maxa′∈A
ˆQ(t−1)
opt (s′,a′)
At each time step, t, the action the agent plays from state s is chosen as follows:
πact(s) =
{
random(a ∈A); with probability P = ε
arg maxa∈A ˆQopt(s,a); with probability P = (1 −ε$
