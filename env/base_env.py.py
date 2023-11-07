import gym
from gym import spaces
import numpy as np
from abc import ABC, abstractmethod

class SlicePlacementEnv(gym.Env, ABC):
    """
    Base class for multi-domain slice placement environments.
    Contains shared logic for:
      - initializing and resetting capacities
      - managing the high-level state‐vector structure
      - handling done/termination logic
    Subclasses must implement:
      - _sample_demands()
      - _build_initial_state()
      - step()
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        # Node/link capacities from config
        self.node_cpu_capacity = config['node_cpu_capacity']
        self.node_ram_capacity = config['node_ram_capacity']
        self.link_bw_capacity = config['link_bw_capacity']

        # Curriculum‐specific slice‐request count per episode
        self.NSPR_numbers_0 = config['NSPR_numbers']

        # Reward shaping parameters
        self.reward_per_vnf = config.get('reward_per_vnf', 10)
        self.reward_final_chain = config.get('reward_final_chain', 100)
        self.penalty_rejection = config.get('penalty_rejection', -200)

        # Maximum possible number of VNFs in any chain (for zero‐padding)
        # We assume a fixed upper bound of 10 (as used in the original code)
        self.MAX_CHAIN_LEN = config.get('max_chain_length', 10)

        # Placeholders to be initialized in reset()
        self.CPU_CAPACITY = None   # shape: (num_nodes,)
        self.RAM_CAPACITY = None   # shape: (num_nodes,)
        self.LINK_CAPACITY = None  # shape: (num_edges,)

        # Track how many requests remain in current episode
        self.NSPR_numbers = None

        # Bookkeeping for logging metrics at episode end
        self.episode_rewards = []
        self.episode_acceptances = []
        self.episode_loads = []

        # The current state vector
        self.state = None

    @abstractmethod
    def _sample_demands(self):
        """
        Sample:
          - self.R_num_vnf: array of length NSPR_numbers_0, each entry = # of VNFs in that slice
          - self.CPU_demands: list of np.arrays of length R_num_vnf[i]
          - self.RAM_demands: list of np.arrays of length R_num_vnf[i]
          - self.BW_demands: list of np.arrays of length (R_num_vnf[i] - 1)
          - self.Delay_demands: list of np.arrays of length (R_num_vnf[i] - 1)
        according to the ranges in self.config.
        """
        raise NotImplementedError

    @abstractmethod
    def _build_initial_state(self):
        """
        Construct the initial state vector for timestep = 0. Should include:
          [ t=0 ]
          + [placement_vector of length MAX_CHAIN_LEN, filled with 1000]
          + [NSPR_numbers]
          + [cost_placeholder = 0]
          + [acceptance_ratio = 0.0]
          + [padded CPU_demands for slice 0 (length MAX_CHAIN_LEN, zeros after actual)]
          + [padded RAM_demands for slice 0 (length MAX_CHAIN_LEN, zeros after actual)]
          + [padded BW_demands for slice 0 (length MAX_CHAIN_LEN, zeros after actual)]
          + [CPU_CAPACITY vector (num_nodes,)]
          + [RAM_CAPACITY vector (num_nodes,)]
          + [LINK_CAPACITY vector (num_edges,)]
          + [load_percentage = 0.0]
        Assign the result to self.state.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, action):
        """
        Execute one placement action (node index) for the current VNF:
          1. Check feasibility (CPU, RAM, bandwidth, delay)
          2. Update capacities, placement_vector, possibly mark reject/accept
          3. Compute reward
          4. Transition to next VNF or next slice
          5. If all slices in this episode placed (NSPR_numbers == 0), set done=True, log metrics
          6. Construct new state vector, return (self.state, reward, done, info)
        """
        raise NotImplementedError

    def reset(self):
        """
        Reset capacities, sample a fresh batch of slice demands, and build initial state.
        Returns:
            np.ndarray: the initial observation vector.
        """
        # 1. Reinitialize resource capacities
        self.CPU_CAPACITY = np.full(self.config['num_nodes'],
                                    self.node_cpu_capacity, dtype=np.int32)
        self.RAM_CAPACITY = np.full(self.config['num_nodes'],
                                    self.node_ram_capacity, dtype=np.int32)
        self.LINK_CAPACITY = np.full(self.config['num_edges'],
                                     self.link_bw_capacity, dtype=np.int32)

        # 2. Reset the number of slice requests remaining
        self.NSPR_numbers = self.NSPR_numbers_0

        # 3. Clear out per-episode logs
        self.episode_rewards = []
        self.episode_acceptances = []
        self.episode_loads = []

        # 4. Sample new demands for all slices in this episode
        self._sample_demands()

        # 5. Build the very first state vector (t = 0)
        self._build_initial_state()
        return np.array(self.state, dtype=np.float32)

    def render(self, mode='human'):
        # Optional: visualize the placement or current capacities
        pass

    def close(self):
        # Optional: any cleanup
        pass
