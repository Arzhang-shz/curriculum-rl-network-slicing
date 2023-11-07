import numpy as np
import networkx as nx
from gym import spaces
from .base_env import SlicePlacementEnv
from .utils import (
    find_edge_index,
    compute_path_delay,
    deduct_path_bandwidth,
    restore_path_bandwidth,
    compute_cpu_ram_update
)

# Hardcode the 43 undirected edges between 33 nodes (0..32). Each pair is [u, v].
EDGE_LIST = [
    [0, 24], [0, 1],  [0, 6],  [1, 6],
    [2, 24], [2, 21], [3, 24], [3, 4],
    [4, 20], [5, 8],  [5, 20], [6, 7],
    [7, 16], [7, 20], [8, 25], [8, 20],
    [9, 20], [10, 20],[10, 13],[11, 21],
    [12, 21],[13, 21],[13, 14],[15, 16],
    [16, 17],[16, 22],[17, 18],[18, 20],
    [19, 20],[19, 21],[20, 21],[21, 24],
    [21, 27],[21, 28],[21, 29],[21, 30],
    [23, 24],[23, 32],[25, 26],[26, 27],
    [28, 29],[30, 31],[31, 32]
]

class PlacementEnv(SlicePlacementEnv):
    """
    Gym environment for multi-domain slice placement (the original 'euav' logic, refactored).
    """

    def __init__(self, config: dict):
        # Add fixed graph parameters into config if not already present
        config = config.copy()
        config.setdefault('num_nodes', 33)
        config.setdefault('num_edges', len(EDGE_LIST))

        # Initialize base class
        super().__init__(config)

        # Build adjacency list from EDGE_LIST
        self.adjacency_list = {i: [] for i in range(self.config['num_nodes'])}
        for (u, v) in EDGE_LIST:
            self.adjacency_list[u].append(v)
            self.adjacency_list[v].append(u)

        # Pre‐create a NetworkX graph for shortest‐path queries
        self.G = nx.Graph()
        self.G.add_nodes_from(range(self.config['num_nodes']))
        for (u, v) in EDGE_LIST:
            self.G.add_edge(u, v)

        # Observation and action spaces
        # action_space: pick a node index in [0 .. num_nodes-1]
        self.action_space = spaces.Discrete(self.config['num_nodes'])

        # observation_space: a flat Box of length:
        #  1 (time) +
        #  MAX_CHAIN_LEN (placement_vector) +
        #  1 (NSPR_numbers) +
        #  1 (cost placeholder) +
        #  1 (acceptance ratio) +
        #  MAX_CHAIN_LEN (CPU demands) +
        #  MAX_CHAIN_LEN (RAM demands) +
        #  MAX_CHAIN_LEN (BW demands) +
        #  num_nodes (CPU_CAPACITY) +
        #  num_nodes (RAM_CAPACITY) +
        #  num_edges (LINK_CAPACITY) +
        #  1 (load percentage)
        obs_dim = (
            1
            + self.MAX_CHAIN_LEN
            + 1
            + 1
            + 1
            + self.MAX_CHAIN_LEN
            + self.MAX_CHAIN_LEN
            + self.MAX_CHAIN_LEN
            + self.config['num_nodes']
            + self.config['num_nodes']
            + self.config['num_edges']
            + 1
        )
        low = np.repeat(-10000.0, obs_dim)
        high = np.repeat(+10000.0, obs_dim)
        self.observation_space = spaces.Box(low=low.astype(np.float32),
                                            high=high.astype(np.float32),
                                            dtype=np.float32)

    def _sample_demands(self):
        """
        Sample slice‐request demands for the entire episode:
          - self.R_num_vnf: array of length NSPR_numbers_0
          - For each slice i:
                CPU_demands[i]: np.array shape=(R_num_vnf[i],) drawn from cpu_demand_range
                RAM_demands[i]: np.array shape=(R_num_vnf[i],) drawn from ram_demand_range
                BW_demands[i]: np.array shape=(R_num_vnf[i] - 1,) drawn from bw_demand_range
                Delay_demands[i]: np.array shape=(R_num_vnf[i] - 1,) drawn from 
                                 (delay_mult * Uniform(delay_range[0], delay_range[1]))
        """
        # Unpack ranges from config
        vnf_min, vnf_max = self.config['vnf_num_range']
        cpu_min, cpu_max = self.config['cpu_demand_range']
        ram_min, ram_max = self.config['ram_demand_range']
        bw_min, bw_max = self.config['bw_demand_range']
        delay_min, delay_max = self.config['delay_range']
        delay_mult = self.config['delay_mult']

        # 1. Sample how many VNFs each slice has
        self.R_num_vnf = np.random.randint(vnf_min, vnf_max + 1, size=self.NSPR_numbers_0)

        # Initialize demand lists
        self.CPU_demands = []
        self.RAM_demands = []
        self.BW_demands = []
        self.Delay_demands = []

        for chain_len in self.R_num_vnf:
            # CPU demands for each VNF in this chain
            cpu_vec = np.random.randint(cpu_min, cpu_max + 1, size=chain_len)
            self.CPU_demands.append(cpu_vec)

            # RAM demands
            ram_vec = np.random.randint(ram_min, ram_max + 1, size=chain_len)
            self.RAM_demands.append(ram_vec)

            # Bandwidth demands for each inter-VNF link (chain_len - 1 values)
            if chain_len > 1:
                bw_vec = np.random.randint(bw_min, bw_max + 1, size=(chain_len - 1))
                delay_vec = delay_mult * np.random.randint(delay_min, delay_max + 1, size=(chain_len - 1))
            else:
                bw_vec = np.array([], dtype=int)
                delay_vec = np.array([], dtype=int)

            self.BW_demands.append(bw_vec)
            self.Delay_demands.append(delay_vec)

    def _build_initial_state(self):
        """
        Compose the initial state vector at t = 0. The first slice to place is index 0.
        """
        t = 0
        # (1) placement_vector: length MAX_CHAIN_LEN, fill with 1000 (meaning "not placed yet")
        placement_vector = [1000] * self.MAX_CHAIN_LEN

        # (2) how many slices remain to be placed
        nspr = self.NSPR_numbers

        # (3) cost placeholder (not used in step logic, but kept for compatibility)
        cost_placeholder = 0.0

        # (4) acceptance ratio so far (0.0)
        acceptance_ratio = 0.0

        # (5) demands for the first slice (index 0) zero-padded to length MAX_CHAIN_LEN
        cpu_vec = self.CPU_demands[0].tolist()
        ram_vec = self.RAM_demands[0].tolist()
        bw_vec = self.BW_demands[0].tolist()

        cpu_padded = cpu_vec + [0] * (self.MAX_CHAIN_LEN - len(cpu_vec))
        ram_padded = ram_vec + [0] * (self.MAX_CHAIN_LEN - len(ram_vec))
        bw_padded = bw_vec + [0] * (self.MAX_CHAIN_LEN - len(bw_vec))

        # (6) current capacities
        cpu_cap_list = self.CPU_CAPACITY.tolist()
        ram_cap_list = self.RAM_CAPACITY.tolist()
        link_cap_list = self.LINK_CAPACITY.tolist()

        # (7) current load percentage = 0.0
        load_pct = 0.0

        # Build the full state vector in order
        self.state = (
            [t]
            + placement_vector
            + [nspr]
            + [cost_placeholder]
            + [acceptance_ratio]
            + cpu_padded
            + ram_padded
            + bw_padded
            + cpu_cap_list
            + ram_cap_list
            + link_cap_list
            + [load_pct]
        )

    def step(self, action: int):
        """
        Execute one action: place the next VNF of the current slice on node 'action'.
        Returns: (next_state, reward, done, info)
        """
        # Unpack previous state
        prev_state = self.state.copy()
        # Convert to list for easy indexing
        prev_state_list = list(prev_state)

        # Indices in the state vector
        #  0               = t
        #  1 .. MAX_CHAIN_LEN      = placement_vector (length MAX_CHAIN_LEN)
        #  1-based indexing helps keep track; in Python use offset
        offset_placement = 1
        offset_nspr = 1 + self.MAX_CHAIN_LEN
        offset_cost = offset_nspr + 1
        offset_accept = offset_cost + 1
        offset_cpu_demand = offset_accept + 1
        offset_ram_demand = offset_cpu_demand + self.MAX_CHAIN_LEN
        offset_bw_demand = offset_ram_demand + self.MAX_CHAIN_LEN
        offset_cpu_cap = offset_bw_demand + self.MAX_CHAIN_LEN
        offset_ram_cap = offset_cpu_cap + self.config['num_nodes']
        offset_link_cap = offset_ram_cap + self.config['num_nodes']
        offset_load = offset_link_cap + self.config['num_edges']

        # Extract sub-vectors
        placement_vector = prev_state_list[offset_placement: offset_placement + self.MAX_CHAIN_LEN]
        nspr = int(prev_state_list[offset_nspr])
        # cost_placeholder = prev_state_list[offset_cost]       # Not used
        acceptance_ratio = prev_state_list[offset_accept]
        cpu_demand_padded = prev_state_list[offset_cpu_demand: offset_cpu_demand + self.MAX_CHAIN_LEN]
        ram_demand_padded = prev_state_list[offset_ram_demand: offset_ram_demand + self.MAX_CHAIN_LEN]
        bw_demand_padded = prev_state_list[offset_bw_demand: offset_bw_demand + self.MAX_CHAIN_LEN]
        cpu_cap_vec = np.array(prev_state_list[offset_cpu_cap: offset_cpu_cap + self.config['num_nodes']], dtype=np.int32)
        ram_cap_vec = np.array(prev_state_list[offset_ram_cap: offset_ram_cap + self.config['num_nodes']], dtype=np.int32)
        link_cap_vec = np.array(prev_state_list[offset_link_cap: offset_link_cap + self.config['num_edges']], dtype=np.int32)

        # 1. Determine which VNF in the current slice we are placing
        try:
            first_avb_index = placement_vector.index(1000)
        except ValueError:
            # Should not occur if reset and step logic are correct
            first_avb_index = self.MAX_CHAIN_LEN - 1

        # Demand for this VNF
        vnf_cpu_req = int(cpu_demand_padded[first_avb_index])
        vnf_ram_req = int(ram_demand_padded[first_avb_index])

        # If not the first VNF, get previous demands for BW and Delay
        if first_avb_index > 0:
            prev_bw_req = int(bw_demand_padded[first_avb_index - 1])
            prev_delay_req = int(self.Delay_demands[self.NSPR_numbers_0 - nspr][first_avb_index - 1])
        else:
            prev_bw_req = None
            prev_delay_req = None

        reject = False
        reward = 0.0

        # 2. Check CPU/RAM feasibility on chosen node
        #    If not enough, immediately reject this chain
        if cpu_cap_vec[action] < vnf_cpu_req or ram_cap_vec[action] < vnf_ram_req:
            reject = True
        else:
            # Tentatively deduct CPU & RAM
            cpu_cap_vec[action] -= vnf_cpu_req
            ram_cap_vec[action] -= vnf_ram_req

        # 3. If this is not the first VNF, check delay + bandwidth on the path
        if (not reject) and (first_avb_index > 0):
            prev_node = placement_vector[first_avb_index - 1]
            # Shortest path using prebuilt NX graph
            try:
                path = nx.shortest_path(self.G, source=prev_node, target=action)
            except nx.NetworkXNoPath:
                reject = True
            if not reject:
                # Delay check
                total_delay = compute_path_delay(EDGE_LIST, path, np.array(self.config['delay_vector']))
                if total_delay > prev_delay_req:
                    reject = True
                else:
                    # Bandwidth check & deduct
                    new_link_cap_vec, ok = deduct_path_bandwidth(EDGE_LIST, path, link_cap_vec, prev_bw_req)
                    if not ok:
                        reject = True
                    else:
                        link_cap_vec = new_link_cap_vec

        # 4. Assign reward or reject penalty
        if reject:
            # Revert partial CPU/RAM deduction for this VNF
            cpu_cap_vec[action] += vnf_cpu_req
            ram_cap_vec[action] += vnf_ram_req

            # If any previous VNFs were placed in this chain, revert their resource deductions:
            # We look at every placed VNF index < first_avb_index
            for idx in range(first_avb_index):
                placed_node = placement_vector[idx]
                prev_cpu_req = int(cpu_demand_padded[idx])
                prev_ram_req = int(ram_demand_padded[idx])
                # Return CPU & RAM
                cpu_cap_vec[placed_node] += prev_cpu_req
                ram_cap_vec[placed_node] += prev_ram_req

            # Also restore any link bandwidth deducted in this partial chain
            # We must reconstruct each sub-path from VNF idx→VNF idx+1, for idx = 0..first_avb_index-2
            for idx in range(first_avb_index - 1):
                u = placement_vector[idx]
                v = placement_vector[idx + 1]
                try:
                    path_restore = nx.shortest_path(self.G, source=u, target=v)
                    link_restore_amount = int(bw_demand_padded[idx])
                    link_cap_vec = restore_path_bandwidth(EDGE_LIST, path_restore, link_cap_vec, link_restore_amount)
                except nx.NetworkXNoPath:
                    pass  # Already no path, skip

            reward = self.penalty_rejection
            # Clear placement vector for this chain
            for idx in range(self.MAX_CHAIN_LEN):
                placement_vector[idx] = 1000
            # Move on to next slice
            nspr -= 1
        else:
            # Successful placement of this VNF
            placement_vector[first_avb_index] = action

            # If this was the last VNF of the chain:
            if first_avb_index == (self.R_num_vnf[self.NSPR_numbers_0 - nspr] - 1):
                # Completed chain
                reward = self.reward_final_chain
                # Reset placement vector for the next chain
                for idx in range(self.MAX_CHAIN_LEN):
                    placement_vector[idx] = 1000
                # Increase acceptance ratio
                acceptance_ratio += 1.0 / float(self.NSPR_numbers_0)
                # Move on to next slice
                nspr -= 1
            else:
                # Not last VNF → intermediate placement reward
                reward = self.reward_per_vnf

        # 5. Compute new load percentage
        total_resources = (
            self.config['num_nodes'] * self.node_cpu_capacity
            + self.config['num_nodes'] * self.node_ram_capacity
            + self.config['num_edges'] * self.link_bw_capacity
        )
        used_cpu = self.config['num_nodes'] * self.node_cpu_capacity - cpu_cap_vec.sum()
        used_ram = self.config['num_nodes'] * self.node_ram_capacity - ram_cap_vec.sum()
        used_bw = self.config['num_edges'] * self.link_bw_capacity - link_cap_vec.sum()
        load_pct = 100.0 * (used_cpu + used_ram + used_bw) / float(total_resources)

        # 6. Build the next state vector
        t_next = prev_state_list[0] + 1
        # Next slice demands (if nspr > 0)
        if nspr > 0:
            # Identify index of the next slice: it is (NSPR_numbers_0 - nspr)
            next_idx = self.NSPR_numbers_0 - nspr
            cpu_vec_next = self.CPU_demands[next_idx].tolist()
            ram_vec_next = self.RAM_demands[next_idx].tolist()
            bw_vec_next = self.BW_demands[next_idx].tolist()
        else:
            cpu_vec_next = []
            ram_vec_next = []
            bw_vec_next = []

        cpu_padded_next = cpu_vec_next + [0] * (self.MAX_CHAIN_LEN - len(cpu_vec_next))
        ram_padded_next = ram_vec_next + [0] * (self.MAX_CHAIN_LEN - len(ram_vec_next))
        bw_padded_next = bw_vec_next + [0] * (self.MAX_CHAIN_LEN - len(bw_vec_next))

        new_state = (
            [t_next]
            + placement_vector
            + [nspr]
            + [0.0]                         # cost placeholder
            + [acceptance_ratio]
            + cpu_padded_next
            + ram_padded_next
            + bw_padded_next
            + cpu_cap_vec.tolist()
            + ram_cap_vec.tolist()
            + link_cap_vec.tolist()
            + [load_pct]
        )

        # Update internal state
        self.state = new_state
        self.CPU_CAPACITY = cpu_cap_vec
        self.RAM_CAPACITY = ram_cap_vec
        self.LINK_CAPACITY = link_cap_vec
        self.NSPR_numbers = nspr

        # 7. Check for end-of-episode
        done = False
        if nspr <= 0:
            done = True
            # Log per-episode metrics
            self.episode_rewards.append(sum(self.episode_rewards) + reward)
            self.episode_acceptances.append(acceptance_ratio)
            self.episode_loads.append(load_pct)
        else:
            # Accumulate reward for partial-episode logging
            self.episode_rewards.append(reward)

        # 8. Return (state, reward, done, info)
        info = {}
        return np.array(self.state, dtype=np.float32), reward, done, info
