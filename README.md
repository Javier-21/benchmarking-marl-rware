<h2 align="center">
    <p>Benchmarking of Multi Agent Reinforcement Learning in Robotic Warehouse</p>
</h2>

This repository contains the source code, execution scripts, and comparative analysis developed in this thesis to evaluate **Multi-Agent Reinforcement Learning (MARL)** algorithms within automated logistics environments. 

The primary objective of this study is to measure and benchmark the task performance, computational overhead, and generalization capabilities of **Transformer-based** architectures (such as MAT and SABLE) against traditional **PPO-based** paradigms (IPPO, MAPPO, MAGPO) using the partially observable **RWARE** robotic warehouse simulator.


## 📌 Table of Contents

* [1. Introduction](#introduction)
  * [Evaluated Algorithms](#evaluated-algorithms)
  * [RWARE Simulator](#rware-simulator)
  * [Key Research Questions](#key-research-questions)
* [2. Installation](#installation)
* [3. Experiments & Results](#experiments)
  * [Base Performance](#base-performance)
  * [Transfer Learning](#transfer-learning)
  * [Zero-Shot Generalization to Map Size](#zero-shot-generalization-to-map-size)
  * [Zero-Shot Generalization to Agent Counts](#zero-shot-generalization-to-agent-counts)
  * [Deactivation of Memory](#deactivation-of-memory)
* [4. Conclusion](#conclusion)
* [5. References](#references)

## 📖 Introduction
The increasing deployment of autonomous agents for task automation is driving the need to discover and develop robust and scalable multi-agent solutions. Recent advancements in Multi-Agent Reinforcement Learning (MARL) have led to the design of frameworks that solve cooperative problems within groups of coordinated agents. However, traditional approaches often struggle with non-stationarity and partial observability. This thesis evaluates state-of-the-art MARL algorithms within a robotic warehouse logistics context (RWARE), a partially observable environment where each agent can only observe a small fraction of the warehouse.

The evaluation focuses on the performance of Transformer-based architectures compared to established paradigms, such as Centralized Training with Decentralized Execution (CTDE) and Independent Reinforcement Learning (IRL). The main goal of this study is to analyze the trade-offs between task performance and resource consumption. Furthermore, the research investigates the Zero-shot Generalization of Transformer-based models, measuring their ability to transfer learned policies from a small-scale to large-scale warehouses without additional training.

This work aims to determine whether Transformer mechanisms can mitigate the scalability limits of current MARL methods. This evaluation seeks to provide a clear benchmark for selecting architectures that balance operational efficacy with computational cost in complex logistical systems.

### Evaluated Algorithms
To understand the impact of architectural choices on multi-agent coordination, we evaluate two distinct paradigms: **PPO-based** baselines and **Transformer-based** architectures. 

A key architectural distinction across these models is how they handle temporal information. Several algorithms offer both a **Feed-Forward (FF)** version for reactive behaviors and a **Recurrent/Memory (REC)** version designed to handle partial observability. However, their underlying mechanisms differ:
* **IPPO and MAPPO (REC):** Implement temporal memory using standard **GRU** modules.
* **MAGPO and SABLE (REC):** Manage temporal dependencies by processing and passing **hidden states** natively through their architectures instead of relying on traditional recurrent networks.

The algorithms, along with their respective script implementations in this repository, are categorized as follows:

<table>
  <thead>
    <tr>
      <th align="left">Paradigm</th>
      <th align="left">Algorithm</th>
      <th align="left">FF Version Script</th>
      <th align="left">REC Version Script</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3"><b>PPO-based</b></td>
      <td>IPPO</td>
      <td><code>ff_ippo.py</code></td>
      <td><code>rec_ippo.py</code></td>
    </tr>
    <tr>
      <td>MAPPO</td>
      <td><code>ff_mappo.py</code></td>
      <td><code>rec_mappo.py</code></td>
    </tr>
    <tr>
      <td>MAGPO</td>
      <td>-</td>
      <td><code>rec_magpo.py</code></td>
    </tr>
    <tr>
      <td rowspan="2"><b>Transformer</b></td>
      <td>MAT</td>
      <td><code>mat.py</code></td>
      <td>-</td>
    </tr>
    <tr>
      <td>SABLE</td>
      <td><code>ff_sable.py</code></td>
      <td><code>rec_sable.py</code></td>
    </tr>
  </tbody>
</table>

#### 1. PPO-Based Baselines
* **IPPO (Independent PPO):** A decentralized approach where each agent learns its own policy independently, treating other agents as part of the environment.
* **MAPPO (Multi-Agent PPO):** Incorporates the Centralized Training with Decentralized Execution (CTDE) framework, utilizing a shared centralized critic to guide individual agent updates.
* **MAGPO:** A framework based on a Teacher-Student architecture, where a centralized "teacher" policy guides and distills knowledge into decentralized "student" policies to improve coordination and training efficiency.

#### 2. Transformer-Based Architectures
* **MAT (Multi-Agent Transformer):** Reinterprets MARL as a sequence-to-sequence problem, framing multi-agent policy generation as a sequential decision-making process via Transformer blocks.
* **SABLE:** An optimized Transformer variant for multi-agent environments that integrates Retentive Networks (RetNet). It leverages specialized retention mechanisms to handle long-horizon dependencies through efficient hidden-state processing, bypassing the limitations of classic recurrent layers.


### RWARE Simulator

The **Robotic Warehouse (`rware`)** simulator utilized in this project is the Jumanji library implementation. It replicates demand-driven, automated mobile-robot fulfillment centers where multiple agents must cooperate to locate specific shelves, deliver them to workstations, and return them under strict spatial constraints.

Here is an example of the environment in action:

<!-- Replace the source path below with your actual gif/video path in your repo -->
<p align="center">
  <img src="render/rec_sable/rware_tiny-4ag.gif" width="500" alt="RWARE Simulator Demo">
</p>


#### Environment Elements & Color Coding
The warehouse map is a grid system that varies depending on the configuration. The layout features the following distinct elements:

*   **White Cells (Highways):** Open roads where agents can move freely without restrictions.
*   **Blue Squares (Standard Shelves):** Stationary storage racks that do not currently contain requested items.
*   **Green Squares (Target Shelves):** Requested shelves containing packages that must be picked up and delivered.
*   **Grey Squares (Workstations):** The delivery stations. Bringing a green shelf here completes a delivery task.
*   **Orange/Red Squares (Robotic Agents):** The MARL agents. They are **Orange** when moving freely and turn **Red** when carrying a shelf. By default, agents have a limited field of view, seeing only **1 cell away** in every direction (a $3 \times 3$ grid around them).


#### Simulation Logic & Constraints
1.  **Initialization:** Agents spawn at random locations at the start of an episode.
2.  **The Pickup Phase:** Agents can navigate anywhere to find and lift a Green Shelf. 
3.  **The Delivery Phase:** Once an agent loads a shelf, its movement is heavily restricted. It cannot cross cells occupied by other shelves and must navigate strictly through the Highways.
4.  **The Reset Loop:** When a shelf reaches the workstation, the task is completed, that shelf turns blue, and a new random blue shelf turns green elsewhere.
5.  **The Unload Constraint:** Agents cannot drop shelves on highway cells, they must return and unload them into valid shelf grid positions before picking up a new one.
6.  **Episode Termination:** To prevent infinite loops, episodes are hard-capped at 500 steps. Additionally, this Jumanji version strictly terminates the simulation immediately if a collision occurs between agents.

#### Reward Structure
RWARE is characterized by **extremely sparse, global, and shared rewards**:
*   A cooperative reward of **$+1$** is given to **all agents** simultaneously only when a target shelf is successfully delivered to a workstation.
*   Returning the empty shelf back to a legal grid cell is a required behavioral loop, but it yields **no immediate reward**.
*   There are no intermediate shaping rewards or negative step penalties.

#### Action Space
Each agent executes discrete decisions chosen from an individual action space of **5 valid actions**:

| Action ID | Action Name | Description |
| :---: | :--- | :--- |
| **0** | `No-Op` | The agent remains stationary during this time step. |
| **1** | `Turn Clockwise` | Rotates the agent's facing direction 90° to the right. |
| **2** | `Turn Counter-Clockwise` | Rotates the agent's facing direction 90° to the left. |
| **3** | `Forward` | Moves the agent one cell ahead in its current direction. |
| **4** | `Load / Unload` | **Loads** a shelf if standing on its cell (turning red), or **unloads** a carried shelf into a valid rack slot (turning orange). |

#### Observation Space
Each agent receives a flattened numerical vector containing **66 elements**, representing its own state and its $3 \times 3$ localized field of view (where the agent occupies the center cell):

*   **Agent State (Positions 1 to 8):**
    *   `2 integers`: Current coordinates $(X, Y)$.
    *   `1 binary`: Load status ($1$ if carrying a shelf, $0$ otherwise).
    *   `4 binaries`: One-hot encoded facing direction (North, South, East, West).
    *   `1 binary`: Highway status ($1$ if currently on a white cell, $0$ otherwise).
*   **Adjacent Cells Agents Data (Positions 9 to 48):**
    *   For each of the 8 visible adjacent cells, a 5-element block details:
        *   `1 binary`: Presence of another agent.
        *   `4 binaries`: One-hot encoded direction of that neighbor agent (all zeros if no agent is present).
*   **Adjacent Cells Shelves Data (Positions 49 to 66):**
    *   For the 9 cells in the field of view, a 2-element block details:
        *   `1 binary`: Presence of a shelf.
        *   `1 binary`: Status of the shelf ($1$ if it is a target/green shelf).



### Key Research Questions 
This benchmark is designed to answer critical questions regarding the current state of MARL:

1. **Performance & Scaling**: Which algorithmic paradigm performs better under tight coordination constraints and scaling agent counts?

2. **Transfer Learning**: Is transfer learning an effective and efficient approach to accelerate model training in warehouse environments?

3. **Zero-Shot Generalization**: Are modern MARL architectures capable of handling zero-shot generalization when environment factors (like grid size or agent density) change?


## 💻 Installation

## 📊 Experiments & Results

### Base Performance

In this experiment, we train all the algorithms from scratch under identical hyperparameter settings to evaluate their fundamental performance in the following environments: `tiny-2ag`, `tiny-4ag`, `small-4ag`, `medium-4ag` and `medium-6ag`. 

The table below displays the average number of delivered packages during inference:

<table>
  <thead>
    <tr>
      <th align="right">Alg. \ Env.</th>
      <th align="right">tiny-2ag</th>
      <th align="right">tiny-4ag</th>
      <th align="right">small-4ag</th>
      <th align="right">medium-4ag</th>
      <th align="right">medium-6ag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>ff ippo</b></td>
      <td align="right">13,14</td>
      <td align="right">14,48</td>
      <td align="right">5,85</td>
      <td align="right">3,39</td>
      <td align="right">0</td>
    </tr>
    <tr>
      <td><b>rec ippo</b></td>
      <td align="right">0,03</td>
      <td align="right">16,87</td>
      <td align="right">0,01</td>
      <td align="right">0</td>
      <td align="right">7,07</td>
    </tr>
    <tr>
      <td><b>ff mappo</b></td>
      <td align="right">14,67</td>
      <td align="right">18,97</td>
      <td align="right">6,79</td>
      <td align="right">4,18</td>
      <td align="right">5,08</td>
    </tr>
    <tr>
      <td><b>rec mappo</b></td>
      <td align="right">17,42</td>
      <td align="right">19,45</td>
      <td align="right">4,05</td>
      <td align="right">0</td>
      <td align="right">0,01</td>
    </tr>
    <tr>
      <td><b>rec magpo</b></td>
      <td align="right">1,98</td>
      <td align="right">21,37</td>
      <td align="right">0,03</td>
      <td align="right">0</td>
      <td align="right">0,02</td>
    </tr>
    <tr>
      <td><b>mat</b></td>
      <td align="right">15,81</td>
      <td align="right">22,66</td>
      <td align="right">13,24</td>
      <td align="right">8,4</td>
      <td align="right">5,62</td>
    </tr>
    <tr>
      <td><b>ff sable</b></td>
      <td align="right">16,27</td>
      <td align="right">27,34</td>
      <td align="right">7,81</td>
      <td align="right">0,02</td>
      <td align="right">0,33</td>
    </tr>
    <tr>
      <td><b>rec sable</b></td>
      <td align="right">16,97</td>
      <td align="right">25,07</td>
      <td align="right">4,92</td>
      <td align="right">0</td>
      <td align="right">0,01</td>
    </tr>
  </tbody>
</table>

* **Tiny Environments & Agent Density:** The `tiny` layout is simple enough for almost all paradigms to learn effectively. Scaling from 2 to 4 agents boosts performance across the board because doubling the robot density accelerates exploration and localized reward discovery. 
* **Transformer Coordination Advantage:** Transformer-based architectures significantly outclass classical models under high-density constraints. When doubling the agent count in `tiny`:
  * **MAT and SABLE** achieve a massive **40% to 70% performance surge**.
  * **IPPO and MAPPO** only show a modest **10% to 30% improvement**.
  This gap highlights the Transformers' superior spatial coordination, allowing them to optimize delivery routes while actively preventing collisions in congested corridors.
* **The Anomalous Case of PPO + Memory:** `rec ippo` and `rec magpo` fail completely with 2 agents in `tiny` maps but match the baseline average with 4 agents (positioning `rec magpo` as the top-performing PPO variant in `tiny-4ag`). This behavior suggests that higher agent density is mandatory for these specific memory architectures to trigger successful initial exploration.
* **The Exploration Bottleneck in Small Maps:** Scaling up to `small` layouts triggers a steep performance drop. **MAT stands out as the only architecture maintaining an acceptable delivery rate**. Standard recurrent models (with GRU or memory blocks) suffer severe performance deterioration here compared to their feed-forward counterparts. This likely  indicates a credit-assignment failure: due to delayed rewards in larger search spaces, the memory blocks wrongly associate long historical trajectories with low-value or useless states.
* **Medium Map Dynamics & Scalability Boundaries:** In `medium` environments, the performance of all memory-centric models alongside `ff sable` collapses close to zero. However, critical structural patterns emerge among the remaining baselines:
  * **IPPO:** Achieves minor success with 4 agents but drops to zero with 6, since treating independent agents as environmental noise compounds complexity exponentially as numbers grow.
  * **MAPPO:** Mitigates this breakdown through its shared centralized critic; elevating the agent count increases the simulation context, yielding slightly superior results.
  * **MAT:** Dominates the 4-agent variant but experiences a coordination bottleneck with 6 agents, equalizing its final score with MAPPO.
  * **rec ippo (Anomaly):** Behaves erratically, failing completely with 4 agents but outperforming other PPO variants with 6 agents. This reinforces the hypothesis that high agent density is the primary driver for successful exploration in memory-recurrent baselines.


The following table records the absolute training execution times for each algorithm across the evaluated simulator configurations:

<table>
  <thead>
    <tr>
      <th align="right">Alg. \ Env.</th>
      <th align="right">tiny-2ag</th>
      <th align="right">tiny-4ag</th>
      <th align="right">small-4ag</th>
      <th align="right">medium-4ag</th>
      <th align="right">medium-6ag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>ff ippo</b></td>
      <td align="right">0:09:37</td>
      <td align="right">0:10:42</td>
      <td align="right">0:14:37</td>
      <td align="right">0:19:36</td>
      <td align="right">0:21:11</td>
    </tr>
    <tr>
      <td><b>rec ippo</b></td>
      <td align="right">0:17:01</td>
      <td align="right">0:19:26</td>
      <td align="right">0:22:36</td>
      <td align="right">0:28:43</td>
      <td align="right">0:31:01</td>
    </tr>
    <tr>
      <td><b>ff mappo</b></td>
      <td align="right">0:09:35</td>
      <td align="right">0:10:38</td>
      <td align="right">0:14:49</td>
      <td align="right">0:20:31</td>
      <td align="right">0:21:40</td>
    </tr>
    <tr>
      <td><b>rec mappo</b></td>
      <td align="right">17:29</td>
      <td align="right">0:18:42</td>
      <td align="right">0:22:35</td>
      <td align="right">0:27:35</td>
      <td align="right">0:31:33</td>
    </tr>
    <tr>
      <td><b>rec magpo</b></td>
      <td align="right">0:17:54</td>
      <td align="right">0:21:48</td>
      <td align="right">0:25:27</td>
      <td align="right">0:30:07</td>
      <td align="right">0:36:12</td>
    </tr>
    <tr>
      <td><b>mat</b></td>
      <td align="right">0:13:24</td>
      <td align="right">0:16:10</td>
      <td align="right">0:19:51</td>
      <td align="right">0:25:34</td>
      <td align="right">0:27:57</td>
    </tr>
    <tr>
      <td><b>ff sable</b></td>
      <td align="right">0:13:23</td>
      <td align="right">0:15:34</td>
      <td align="right">0:19:33</td>
      <td align="right">0:24:37</td>
      <td align="right">0:27:48</td>
    </tr>
    <tr>
      <td><b>rec sable</b></td>
      <td align="right">0:13:49</td>
      <td align="right">0:17:12</td>
      <td align="right">0:20:51</td>
      <td align="right">0:25:37</td>
      <td align="right">0:31:01</td>
    </tr>
  </tbody>
</table>

* **Map Size Penalty:** As the simulator complexity scales, training times increase significantly. Map dimensions act as the primary computational bottleneck due to the intense model-environment interaction loop, which inherently decreases the steps-per-second (SPS) processing throughput.
* **Algorithmic Overhead:** Classical, feed-forward PPO baselines remain the fastest configurations. However, introducing recurrent memory layers inflicts a massive temporal penalty. Overall, **MAGPO** emerges as the most computationally expensive architecture. 
* **The Transformer Sweet Spot:** Transformer-based models manage a well-balanced computational profile. While marginally slower than reactive, feed-forward PPO networks, they execute significantly faster than traditional recurrent baseline architectures (GRU-based models).

#### Execution & Replication Script
To replicate these baseline experiments, use the following execution template.

```bash
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

python mava/systems/mat/anakin/mat.py
    env=rware
    env/scenario=tiny-2ag
    logger.loggers.tensorboard.enabled=true
    logger.loggers.json.enabled=true
    logger.checkpointing.save_model=true
    logger.checkpointing.save_args.save_interval_steps=10
    arch.num_envs=32
    arch.num_evaluation=244
    arch.num_absolute_metric_eval_episodes=640
    system.num_updates=2000
```

The first line is instructions to prevent JAX from pre-allocating 90% of the GPU VRAM. Given our hardware resources, this parameter was mandatory to avoid system deadlocks. The rest of the command handles the execution of the selected algorithm—which in this example is MAT utilizing the Anakin architecture. The subsequent configuration variables manage the environment setup, logging, and hyperparameters.

For this specific benchmark, the only variable modified across runs is env/scenario to swap the target map configuration. During the training pipeline, 32 environment scenarios run simultaneously, the policy is evaluated 244 times, and the network undergoes a total of 2,000 system updates. Finally, to test model inference, 640 absolute evaluation episodes are executed.


---

### Transfer learning
This section analyzes the impact of transferring knowledge across different warehouse configurations. The evaluation tracks how policies trained in smaller layouts scale up to more complex environments.

>❗This transfer learning analysis excludes `rec ippo` and `rec magpo` due to poor initial performance in tiny layouts. <br> ❗A strict requirement for policy transfer is that the network parameter count must remain completely independent of the agent density. This rule makes MAPPO structurally incompatible because its centralized critic network grows proportionally with the agent count. This variation prevents a pretrained 2 agent model from loading into a 4 agent environment.

The following tables showcase the average reward achieved when training from scratch compared to initializing the weights with prior knowledge.

<table width="100%">
  <!-- FILA 1: Tiny y Small -->
  <tr>
    <td width="50%" valign="top">
      <h4>Tiny Layout (4 Agents)</h4>
      <p>Models are initialized using weights from the tiny 2ag environment.</p>
      <table>
        <thead>
          <tr>
            <th align="left">Algorithm</th>
            <th align="center">Base Performance</th>
            <th align="center">Transfer Learning</th>
          </tr>
        </thead>
        <tbody>
          <tr><td>ff ippo</td><td align="center">14.48</td><td align="center">23.47</td></tr>
          <tr><td>mat</td><td align="center">22.66</td><td align="center">27.38</td></tr>
          <tr><td>ff sable</td><td align="center">27.34</td><td align="center">34.01</td></tr>
          <tr><td>rec sable</td><td align="center">25.07</td><td align="center">35.64</td></tr>
        </tbody>
      </table>
    </td>
    <td width="50%" valign="top">
      <h4>Small Layout (4 Agents)</h4>
      <p>Models are initialized using weights from the previously retrained tiny 4ag environment.</p>
      <table>
        <thead>
          <tr>
            <th align="left">Algorithm</th>
            <th align="center">Base Performance</th>
            <th align="center">Transfer Learning</th>
          </tr>
        </thead>
        <tbody>
          <tr><td>ff ippo</td><td align="center">5.85</td><td align="center">0.00</td></tr>
          <tr><td>mat</td><td align="center">13.24</td><td align="center">17.49</td></tr>
          <tr><td>ff sable</td><td align="center">7.81</td><td align="center">0.00</td></tr>
          <tr><td>rec sable</td><td align="center">4.92</td><td align="center">21.54</td></tr>
        </tbody>
      </table>
    </td>
  </tr>
  <!-- FILA 2: Medium 4ag y Medium 6ag -->
  <tr>
    <td width="50%" valign="top">
      <h4>Medium Layout (4 Agents)</h4>
      <p>Models are initialized selectively based on the best results from the small 4ag phase. The models ff ippo and ff sable load their base configurations while mat and rec sable load their TL weights.</p>
      <table>
        <thead>
          <tr>
            <th align="left">Algorithm</th>
            <th align="center">Base Performance</th>
            <th align="center">Transfer Learning</th>
          </tr>
        </thead>
        <tbody>
          <tr><td>ff ippo</td><td align="center">3.39</td><td align="center">7.49</td></tr>
          <tr><td>mat</td><td align="center">8.40</td><td align="center">12.94</td></tr>
          <tr><td>ff sable</td><td align="center">0.02</td><td align="center">11.34</td></tr>
          <tr><td>rec sable</td><td align="center">0.00</td><td align="center">16.71</td></tr>
        </tbody>
      </table>
    </td>
    <td width="50%" valign="top">
      <h4>Medium Layout (6 Agents)</h4>
      <p>Models are initialized using weights directly from the medium 4ag transfer learning phase.</p>
      <table>
        <thead>
          <tr>
            <th align="left">Algorithm</th>
            <th align="center">Base Performance</th>
            <th align="center">Transfer Learning</th>
          </tr>
        </thead>
        <tbody>
          <tr><td>ff ippo</td><td align="center">0.00</td><td align="center">11.21</td></tr>
          <tr><td>mat</td><td align="center">5.62</td><td align="center">18.10</td></tr>
          <tr><td>ff sable</td><td align="center">0.33</td><td align="center">17.97</td></tr>
          <tr><td>rec sable</td><td align="center">0.01</td><td align="center">25.30</td></tr>
        </tbody>
      </table>
    </td>
  </tr>
</table>

* **Tiny 4ag Environments:** Significant performance spikes appear across all models. Initializing weights via tiny 2ag yields a 60% performance boost for `ff ippo`, matching the baseline of Transformer architectures. The memory version of SABLE improves by over 40%, securing the top spot for this configuration.

* **Small 4ag Environments:** Results reveal a high degree of instability. Both `ff ippo` and `ff sable` fail to learn entirely. This suggests that rigid feed forward architectures overfit to the initial map dimensions, restricting their ability to explore new layouts (more information in section [Overfitting to Map Size](#overfitting-to-map-size)). Conversely, `mat` and `rec sable` successfully leverage prior knowledge, with `rec sable` gaining over 15 reward points.

* **Medium 4ag Environments:** Every single algorithm successfully outperforms its respective baseline. While `ff ippo` doubles its score, it remains the lowest performing option. `mat` yields a steady but moderate improvement due to its high initial baseline. Both SABLE architectures jump from a baseline of zero to remarkable performance levels, especially the recurrent version.

* **Medium 6ag Environments:** While `mat` was the only algorithm capable of scoring above zero from scratch, transfer learning enables every model to surpass the 10 point threshold. `rec sable` achieves the highest reward by a wide margin, proving that weight initialization gives memory structures an outstanding training advantage.

---

<p align="center">
  <!-- Place your 4 evaluation graphs here -->
  <img src="charts\202604_tl_vs_hard_ff_ippo.png" width="45%" alt="IPPO Comparison">
  <img src="charts\202604_tl_vs_hard_mat.png" width="45%" alt="MAT Comparison">
  <br>
  <img src="charts\202604_tl_vs_hard_ff_sable.png" width="45%" alt="FF SABLE Comparison">
  <img src="charts\202604_tl_vs_hard_rec_sable.png" width="45%" alt="REC SABLE Comparison">
</p>

* **ff ippo:** The environment complexity prevents standard learning. Without weight initialization, exploration fails completely to discover rewarding states. Transfer learning starts the policy at a reward near 10 and maintains a slight positive trend above that value.

* **mat:** Demonstrates exceptional exploration across all three training modes. The transfer learning curve starts high at 15 points and grows gradually to 18. Intensive training shows an abrupt learning spike after checkpoint 100 followed by a plateau, eventually converging logarithmically with the transfer learning results.

* **ff sable:** Exhibits dynamics similar to MAT. The transfer learning run starts near 15 and increases steadily to 18. Intensive training triggers a linear improvement after checkpoint 50 to match the TL score. The base run delays its learning until checkpoint 250, indicating that it would require significantly more iterations to converge.

* **rec sable:** Faces notable exploration bottlenecks when training from scratch. The transfer learning run starts at 20 points and scales up to 26, showing the highest growth rate. Intensive training remains flat until checkpoint 170 before climbing linearly past 18 points. The base setup only begins to learn around checkpoint 350, highlighting a severe dependency on heavy compute resources.

---

### Zero-Shot Generalization to Map Size

#### Overfitting to Map Size

### Zero-Shot Generalization to Agent Counts

### Deactivation of Memory

## 📝 Conclusion

## 📚 References

The MAVA framework was developed by [InstaDeep](https://instadeep.com/). I have modified some parts so as to solve incompatibilities with my configuration or add new features, but the main algorithms' core belongs to InstaDeep's work. This [link](https://github.com/instadeepai/Mava/tree/main) redirects to the original MAVA repository.

I cite their [technical report][Paper]:

```bibtex
@article{dekock2023mava,
    title={Mava: a research library for distributed multi-agent reinforcement learning in JAX},
    author={Ruan de Kock and Omayma Mahjoub and Sasha Abramowitz and Wiem Khlifi and Callum Rhys Tilbury
    and Claude Formanek and Andries P. Smit and Arnu Pretorius},
    year={2023},
    journal={arXiv preprint arXiv:2107.01460},
    url={https://arxiv.org/pdf/2107.01460.pdf},
}
```


[Paper]: https://arxiv.org/pdf/2107.01460.pdf
