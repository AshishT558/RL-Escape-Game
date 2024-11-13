# RL-Escape-Game
A Reinforcement-Learning based game in which an agent learns how to escape a castle with guards. This is part of the Graduate Foundation of AI course at Northeastern (CS5100). In the MFMC.py file, the Q_learning() function executes a specified number of episodes and has the agent follow an ε-greedy policy. 

### Update Step and Policy for Q-learning

The update step for the Q-learning estimate, \( \hat{Q}_{\text{opt}} \), is:

\[
\eta = \frac{1}{1 + \text{number of updates to } \hat{Q}_{\text{opt}}(s, a)}
\]

\[
\hat{Q}_{\text{opt}}^t(s, a) = (1 - \eta) \hat{Q}_{\text{opt}}^{(t-1)}(s, a) + \eta \left[ R(s, a, s') + \gamma \hat{V}_{\text{opt}}^{(t-1)}(s') \right]
\]

where:

\[
\hat{V}_{\text{opt}}(s) = \max_{a' \in A} \hat{Q}_{\text{opt}}^{(t-1)}(s', a')
\]

At each time step, \( t \), the action the agent plays from state \( s \) is chosen as follows:

\[
\pi_{\text{act}}(s) = 
\begin{cases} 
\text{random}(a \in A); & \text{with probability } P = \epsilon \\
\arg \max_{a \in A} \hat{Q}_{\text{opt}}(s, a); & \text{with probability } P = (1 - \epsilon)
\end{cases}
\]
