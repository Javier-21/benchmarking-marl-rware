<h2 align="center">
    <p>Benchmarking of Multi Agent Reinforcement Learning in Robotic Warehouse</p>
</h2>


## Introduction
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


## Installation

## Experiments

### Base performance

### Transfer learning

### Zero-shot generalization

## Conclusion

## References

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
