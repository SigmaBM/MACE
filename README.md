## Settling Decentralized Multi-Agent Coordinated Exploration by Novelty Sharing (MACE)

Code for "Settling Decentralized Multi-Agent Coordinated Exploration by Novelty Sharing" accepted by AAAI 2024. [[PDF]](https://arxiv.org/abs/2402.02097)

### MACE

Exploration in decentralized cooperative multi-agent reinforcement learning faces two challenges. One is that the novelty of global states is unavailable, while the novelty of local observations is biased. The other is how agents can explore in a coordinated way. To address these challenges, we propose MACE, a simple yet effective multi-agent coordinated exploration method. 

By communicating only local novelty, agents can take into account other agents’ local novelty $u_t^j$ to approximate the global novelty. 

$$
r_{\mathrm{nov}}^i\left(o_t^i, a_t^i\right)=\sum_j u_t^j
$$

Further, we newly introduce **weighted mutual information** to measure the influence of one agent’s action on other agents’ accumulated novelty. 

$$
\omega I\left(A_t^i ; Z_t^j \mid o_t^i\right)=\mathbb{E}_{a_t^i, z_t^j \mid o_t^i}\left[\omega(a_t^i, z_t^j) \log \frac{p(a_t^i, z_t^j \mid o_t^i)}{p(a_t^i \mid o_t^i) p(z_t^j \mid o_t^i)}\right]
$$

We convert it as an intrinsic reward in hindsight to encourage agents to exert more influence on other agents’ exploration and boost coordinated exploration. 

$$
r_{\mathrm{hin}}^{i \rightarrow j}\left(o_t^i, a_t^i, z_t^j\right)=z_t^j \log \frac{p(a_t^i \mid o_t^i, z_t^j)}{\pi^i(a_t^i \mid o_t^i)}
$$

We combine the two intrinsic rewards to get the final shaped reward.

$$
r_{\mathrm{s}}^i\left(o_t^i, a_t^i,\{z_t^j\}\_{j \neq i}\right)=r\_{\mathrm{ext}}+r_{\mathrm{nov}}^i\left(o_t^i, a_t^i\right)+\lambda \sum_{j \neq i} r_{\mathrm{hin}}^{i \rightarrow j}\left(o_t^i, a_t^i, z_t^j\right)=r_{\mathrm{ext}}+\sum_j u_t^j+\lambda \sum_{j \neq i} z_t^j \log \frac{p(a_t^i \mid o_t^i, z_t^j)}{\pi^i(a_t^i \mid o_t^i)}
$$


### Training

For GridWorld environment:

```shell
./scripts/train_gridworld.sh
```

For Overcooked environment:

```
./scripts/train_overcooked.sh
```

