# transformer_solver/pocat_env.py

import torch
from tensordict import TensorDict
from torchrl.envs import EnvBase
from typing import Optional, Dict, Union

from torchrl.data import Unbounded, UnboundedDiscrete, Composite

from .definitions import (
    SCALAR_PROMPT_FEATURE_DIM, FEATURE_DIM, FEATURE_INDEX,
    NODE_TYPE_BATTERY, NODE_TYPE_IC, NODE_TYPE_LOAD
)


# --- [핵심] 하이브리드 보상 가중치 상수 ---
# R_action: 즉각적인 IC 비용에 대한 가중치 (0.0으로 설정 시 순수 R_path)
REWARD_WEIGHT_ACTION = 0
# R_path: 경로 완성 시 누적 비용(staging_cost)에 대한 가중치
REWARD_WEIGHT_PATH = 1.0
# 스텝 페널티: 경로를 불필요하게 길게 만드는 것을 방지
STEP_PENALTY = 0
# R_fail: 실패 시 페널티
FAILURE_PENALTY = -100.0
# 👈 [암전류] 초과된 암전류 1A당 페널티 (음수 보상) 크기
PENALTY_WEIGHT_SLEEP = 1000.0 # 예: 1mA(0.001A) 초과 시 -10.0의 페널티

BATTERY_NODE_IDX = 0

class PocatEnv(EnvBase):
    name = "pocat"

    def __init__(self, generator_params: dict = {}, device: str = "cpu", **kwargs):
        super().__init__(device=device)
        from .env_generator import PocatGenerator
        self.generator = PocatGenerator(**generator_params)
        
        # 버퍼는 _ensure_buffers에서 동적으로 생성되므로 __init__에서는 None으로 초기화
        self.register_buffer("arange_nodes", None, persistent=False)
        self.register_buffer("node_type_tensor", None, persistent=False)
        self.register_buffer("rail_types", None, persistent=False)

        self._make_spec()
        self._load_constraint_info()

    # --- [개선] 버퍼 크기 동기화 함수 추가 ---
    def _ensure_buffers(self, td: TensorDict):
        """에피소드마다 그래프/로드 수가 바뀔 경우를 대비해 버퍼를 동기화합니다."""
        num_nodes = td["nodes"].shape[1]

        if self.arange_nodes is None or self.arange_nodes.numel() != num_nodes:
            self.arange_nodes = torch.arange(num_nodes, device=self.device)
        
        # node_type_tensor는 config에서 오므로 고정, __init__에서 한 번만 생성되도록 수정
        if (self.node_type_tensor is None) or (self.node_type_tensor.numel() != num_nodes):
            node_types_list = [self.generator.config.node_types[i] for i in range(num_nodes)]
            self.node_type_tensor = torch.tensor(node_types_list, dtype=torch.long, device=self.device)

        # rail_types도 config에서 오므로 고정
        if (self.rail_types is None) or (self.rail_types.numel() != self.generator.num_loads):
            rail_type_map = {"exclusive_supplier": 1, "exclusive_path": 2}
            load_configs = self.generator.config.loads
            rail_types_list = [rail_type_map.get(cfg.get("independent_rail_type"), 0) for cfg in load_configs]
            self.rail_types = torch.tensor(rail_types_list, dtype=torch.long, device=self.device) if rail_types_list else torch.tensor([], dtype=torch.long, device=self.device)

    def _make_spec(self):
        """환경의 observation, action, reward 스펙을 정의합니다."""
        num_nodes = self.generator.num_nodes
        
        self.observation_spec = Composite({
            "nodes": Unbounded(shape=(num_nodes, FEATURE_DIM)),
            "scalar_prompt_features": Unbounded(shape=(SCALAR_PROMPT_FEATURE_DIM,)),
            "matrix_prompt_features": Unbounded(shape=(num_nodes, num_nodes)),
            "connectivity_matrix": Unbounded(shape=(num_nodes, num_nodes), dtype=torch.bool),
            "adj_matrix": Unbounded(shape=(num_nodes, num_nodes), dtype=torch.bool),
            "unconnected_loads_mask": Unbounded(shape=(num_nodes,), dtype=torch.bool),
            "trajectory_head": UnboundedDiscrete(shape=(1,)),
            "step_count": UnboundedDiscrete(shape=(1,)),
            # --- 👇 [여기에 새로운 상태 명세를 추가합니다] ---
            "current_cost": Unbounded(shape=(1,)),
            "staging_cost": Unbounded(shape=(1,)), # *현재 구축 중인* 경로의 누적 비용
            "is_used_ic_mask": Unbounded(shape=(num_nodes,), dtype=torch.bool),
            "current_target_load": UnboundedDiscrete(shape=(1,)),
            "is_exclusive_mask": Unbounded(shape=(num_nodes,), dtype=torch.long), # 👈 [신규] 0: Normal, 1: Supplier, 2: Path
        })
        
        self.action_spec = UnboundedDiscrete(shape=(1,))
        self.reward_spec = Unbounded(shape=(1,))

    def _set_seed(self, seed: Optional[int] = None):
        if seed is not None:
            torch.manual_seed(seed)

    # 💡 **[변경 3]** 제약조건 정보를 미리 가공하여 저장하는 헬퍼 함수
    def _load_constraint_info(self):
        """config 파일에서 제약조건 정보를 로드하고 마스킹에 사용하기 쉽게 가공합니다."""
        self.node_name_to_idx = {name: i for i, name in enumerate(self.generator.config.node_names)}
        
        # Independent Rail 정보
        self.exclusive_supplier_loads = set()
        self.exclusive_path_loads = set()

        loads_config = self.generator.config.loads
        if loads_config:
            load_start_idx = 1 + self.generator.num_ics
            for i, load_cfg in enumerate(loads_config):
                load_idx = load_start_idx + i
                if load_cfg.get("independent_rail_type") == "exclusive_supplier":
                    self.exclusive_supplier_loads.add(load_idx)
                elif load_cfg.get("independent_rail_type") == "exclusive_path":
                    self.exclusive_path_loads.add(load_idx)
            # set에 정보가 채워진 후 tensor를 생성합니다.
            if self.exclusive_path_loads:
                self.exclusive_path_loads_tensor = torch.tensor(
                    sorted(self.exclusive_path_loads), dtype=torch.long, device=self.device
                )
        if self.exclusive_supplier_loads:
            self.exclusive_supplier_loads_tensor = torch.tensor(
                sorted(self.exclusive_supplier_loads), dtype=torch.long, device=self.device
            )
        else:
            self.exclusive_supplier_loads_tensor = torch.tensor([], dtype=torch.long, device=self.device)
        if not self.exclusive_path_loads:
            self.exclusive_path_loads_tensor = torch.tensor([], dtype=torch.long, device=self.device)

        # Power Sequence 정보에 f 플래그(동시 허용 여부) 추가
        self.power_sequences = []
        for seq in self.generator.config.constraints.get("power_sequences", []):
            f_flag = seq.get("f", 1)
            j_idx = self.node_name_to_idx.get(seq['j'])
            k_idx = self.node_name_to_idx.get(seq['k'])
            if j_idx is not None and k_idx is not None:
                self.power_sequences.append((j_idx, k_idx, f_flag))

    def select_start_nodes(self, td: TensorDict):
        node_types = td["nodes"][0, :, FEATURE_INDEX["node_type"][0]:FEATURE_INDEX["node_type"][1]].argmax(-1)
        start_nodes_idx = torch.where(node_types == NODE_TYPE_LOAD)[0]
        return len(start_nodes_idx), start_nodes_idx
    
    def _trace_path_batch(self, start_nodes: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        """배치 전체에 대해 start_node들의 모든 조상을 찾아 마스크로 반환합니다."""
        batch_size, num_nodes, _ = adj_matrix.shape
        path_mask = torch.zeros(batch_size, num_nodes, dtype=torch.bool, device=self.device)

        # start_nodes가 비어있지 않을 때만 scatter_ 실행
        if start_nodes.numel() > 0:
            path_mask.scatter_(1, start_nodes.unsqueeze(-1), True)

        # 행렬 곱셈을 이용해 그래프를 거슬러 올라가며 모든 조상을 찾습니다.
        for _ in range(num_nodes):
            # 현재 경로에 포함된 노드들의 부모를 찾습니다.
            parents_mask = (
                # Use the transpose to follow incoming edges when accumulating parents.
                adj_matrix.transpose(-1, -2).float() @ path_mask.float().unsqueeze(-1)
            ).squeeze(-1).bool()            # 더 이상 새로운 부모가 없으면 (경로의 끝에 도달하면) 종료합니다.
            if (parents_mask & ~path_mask).sum() == 0: break
            # 새로 찾은 부모들을 경로 마스크에 추가합니다.
            path_mask |= parents_mask
        return path_mask

    def _reset(self, td: Optional[TensorDict] = None, **kwargs) -> TensorDict:
        batch_size = kwargs.get("batch_size", self.batch_size)
        if td is None:
            batch_size = kwargs.get("batch_size", self.batch_size)
            if isinstance(batch_size, tuple): batch_size = batch_size[0]
            td_initial = self.generator(batch_size=batch_size).to(self.device)
        # td가 인자로 들어오면, 그 td를 초기 상태로 사용합니다.
        else:
            td_initial = td
            # 배치 크기도 들어온 td에서 가져옵니다.
            batch_size = td_initial.batch_size[0]

        num_nodes = td_initial["nodes"].shape[1]

        # --- 💡 1. Trajectory 기반 상태(state) 재정의 ---
        reset_td = TensorDict({
            "nodes": td_initial["nodes"],
            "scalar_prompt_features": td_initial["scalar_prompt_features"],
            "matrix_prompt_features": td_initial["matrix_prompt_features"],
            "connectivity_matrix": td_initial["connectivity_matrix"],
            "adj_matrix": torch.zeros(batch_size, num_nodes, num_nodes, dtype=torch.bool, device=self.device),
            "trajectory_head": torch.full((batch_size, 1), BATTERY_NODE_IDX, dtype=torch.long, device=self.device),
            "unconnected_loads_mask": torch.ones(batch_size, num_nodes, dtype=torch.bool, device=self.device),
            "step_count": torch.zeros(batch_size, 1, dtype=torch.long, device=self.device),
            # --- 👇 [여기에 새로운 상태 초기값을 추가합니다] ---
            "current_cost": torch.zeros(batch_size, 1, dtype=torch.float32, device=self.device),
            "staging_cost": torch.zeros(batch_size, 1, dtype=torch.float32, device=self.device), #
            "is_used_ic_mask": torch.zeros(batch_size, num_nodes, dtype=torch.bool, device=self.device),
            "current_target_load": torch.full((batch_size, 1), -1, dtype=torch.long, device=self.device),
            "is_exclusive_mask": torch.zeros(batch_size, num_nodes, dtype=torch.long, device=self.device), # 👈 [신규] 0으로 초기화
        }, batch_size=[batch_size], device=self.device)
       
        # 배터리(인덱스 0)는 항상 메인 트리에 포함
        node_types = td_initial["nodes"][0, :, FEATURE_INDEX["node_type"][0]:FEATURE_INDEX["node_type"][1]].argmax(-1)
        is_load = node_types == NODE_TYPE_LOAD
        reset_td["unconnected_loads_mask"][:, ~is_load] = False
        reset_td.set("done", torch.zeros(batch_size, 1, dtype=torch.bool, device=self.device))
        self._ensure_buffers(reset_td)
        return reset_td

    # 💡 추가된 step 메소드: 배치 크기 검사를 우회합니다.
    def step(self, tensordict: TensorDict) -> TensorDict:
        return self._step(tensordict)

    def _calculate_power_loss(self, ic_node_features: torch.Tensor, i_out: torch.Tensor) -> torch.Tensor:
        ic_type = ic_node_features[:, :, FEATURE_INDEX["ic_type_idx"]]
        vin = ic_node_features[:, :, FEATURE_INDEX["vin_min"]]
        vout = ic_node_features[:, :, FEATURE_INDEX["vout_min"]]

        power_loss = torch.zeros_like(i_out)
        
        # LDO
        ldo_mask = ic_type == 1.0
        if ldo_mask.any():
            op_current = ic_node_features[:, :, FEATURE_INDEX["op_current"]]
            power_loss[ldo_mask] = (vin[ldo_mask] - vout[ldo_mask]) * i_out[ldo_mask] + vin[ldo_mask] * op_current[ldo_mask]
        
        # Buck
        buck_mask = ic_type == 2.0
        if buck_mask.any():
            s, e = FEATURE_INDEX["efficiency_params"]
            a, b, c = ic_node_features[:, :, s:e].permute(2, 0, 1)
            i_out_buck = i_out[buck_mask]
            power_loss[buck_mask] = a[buck_mask] * (i_out_buck**2) + b[buck_mask] * i_out_buck + c[buck_mask]
            
        return power_loss

    def _step(self, td: TensorDict) -> TensorDict:
        batch_size, num_nodes, _ = td["nodes"].shape
        action = td["action"].reshape(batch_size)
        current_head = td["trajectory_head"].reshape(batch_size)

        # --- 👇 [핵심 버그 수정 1] ---
        # 이미 'done' 상태인 샘플을 식별합니다. (B,)
        is_already_done = td["done"].squeeze(-1)
        
        # 만약 모든 샘플이 이미 'done'이면, 즉시 0점짜리 리워드를 반환합니다.
        if is_already_done.all():
            return TensorDict({
                "next": td, 
                "reward": torch.zeros(batch_size, device=self.device), 
                "done": td["done"]}, batch_size=td.batch_size)
        # --- 수정 완료 ---

        # 💡 [핵심 수정] 얕은 복사 대신, 수정될 텐서만 깊은 복사(deep copy)
        next_obs = td.clone() # 껍데기는 얕은 복사
        next_obs["nodes"] = td["nodes"].clone()
        next_obs["adj_matrix"] = td["adj_matrix"].clone()
        next_obs["is_used_ic_mask"] = td["is_used_ic_mask"].clone()
        next_obs["current_target_load"] = td["current_target_load"].clone()
        # ---  staging_cost 복제 및 스텝 리워드 초기화 ---
        next_obs["current_cost"] = td["current_cost"].clone()
        next_obs["staging_cost"] = td["staging_cost"].clone()
        # 기본 보상: 작은 스텝 페널티 (경로를 짧게 만들도록 유도)
        step_reward = torch.full((batch_size,), STEP_PENALTY, dtype=torch.float32, device=self.device)
        # --- 수정 완료 ---
        batch_indices = torch.arange(batch_size, device=self.device)

        # 1. 액션 타입: [Select New Load]
        head_is_battery = current_head == BATTERY_NODE_IDX
        if head_is_battery.any():
            # [Select New Load]
            battery_rows = batch_indices[head_is_battery]
            action_from_battery = action[head_is_battery]

            # --- 👇 [핵심 버그 수정] ---
            # 액션이 실제 '로드'인 경우와 '배터리(0)'(종료)인 경우를 분리
            is_load_selection = (action_from_battery != BATTERY_NODE_IDX)
            if is_load_selection.any():
                load_rows = battery_rows[is_load_selection]
                selected_load = action_from_battery[is_load_selection]

                next_obs["trajectory_head"][load_rows, 0] = selected_load
                next_obs["unconnected_loads_mask"][load_rows, selected_load] = False
                next_obs["current_target_load"][load_rows, 0] = selected_load
                next_obs["staging_cost"][load_rows] = 0.0
                
                # '표기' 시작: 로드의 독립 조건을 is_exclusive_mask에 기록
                load_start_idx = 1 + self.generator.num_ics
                load_indices_in_config = selected_load - load_start_idx
                rail_types_to_set = self.rail_types[load_indices_in_config]
                next_obs["is_exclusive_mask"][load_rows, selected_load] = rail_types_to_set

            # (이 스텝의 보상은 STEP_PENALTY만 적용됨)

            # (액션이 0번(배터리)인 경우는 아무 작업도 하지 않고 'Find Parent'로 넘어감)

        # 2. 액션 타입: [Find Parent]
        head_is_node = ~head_is_battery
        if head_is_node.any():
            # [Find Parent]
            node_rows = batch_indices[head_is_node]
            child_node = current_head[head_is_node]
            parent_node = action[head_is_node]

            # 2. 연결 정보 및 사용 여부 업데이트
            next_obs["adj_matrix"][node_rows, parent_node, child_node] = True
            node_types = td["nodes"][0, :, FEATURE_INDEX["node_type"][0]:FEATURE_INDEX["node_type"][1]].argmax(-1)
            # --- 👇 [핵심 수정 3 & Point 1, 3 Fix] '표기' 전파: 자식의 독립 조건을 부모가 물려받음 ---
            # (B_act,)
            child_status = next_obs["is_exclusive_mask"][node_rows, child_node]
            # 'child_status'가 0 (Normal)이 아닌 경우에만 '표기' 로직 실행
            if (child_status > 0).any():
                parent_status = next_obs["is_exclusive_mask"][node_rows, parent_node]

                # [Point 1, 5 Fix] dtype을 child_status.dtype (long)으로 통일
                zero_tensor = torch.tensor(0, device=self.device, dtype=child_status.dtype)
                # 'Path(2)'만 상위로 전파됨.
                status_to_propagate = torch.where(child_status == 2, child_status, zero_tensor)
                # 부모는 (1)자신의 상태, (2)전파된 Path, (3)자식의 Supplier(1) 상태 중 가장 높은 값을 가짐.
                status_from_child = torch.where(child_status == 1, child_status, status_to_propagate)
                new_parent_status = torch.max(parent_status, status_from_child)
                next_obs["is_exclusive_mask"][node_rows, parent_node] = new_parent_status
            # --- '표기' 전파 완료 ---
            is_parent_ic = (node_types[parent_node] == NODE_TYPE_IC)
            if is_parent_ic.any():
                ic_rows = node_rows[is_parent_ic]
                ic_indices = parent_node[is_parent_ic]
                next_obs["is_used_ic_mask"][ic_rows, ic_indices] = True


            # 4. 다음 헤드 결정 (비용 계산 후 수행)
            parent_is_battery = (parent_node == BATTERY_NODE_IDX)
            next_obs["trajectory_head"][node_rows, 0] = torch.where(parent_is_battery, BATTERY_NODE_IDX, parent_node)
            if parent_is_battery.any():
                finished_rows = node_rows[parent_is_battery]
                next_obs["current_target_load"][finished_rows, 0] = -1

        # 5. 전류, 온도, 비용 업데이트
        # 1. 초기 전류 수요는 Load의 active_current로 설정
        current_demands = next_obs["nodes"][..., FEATURE_INDEX["current_active"]].clone()
        ic_mask_b_n = (next_obs["nodes"][..., FEATURE_INDEX["node_type"][0] + NODE_TYPE_IC] == 1.0)
        current_demands[ic_mask_b_n] = 0.0 # (B, N) 마스크를 (B, N) 텐서에 직접 적용
        
        adj_matrix_T = next_obs["adj_matrix"].float().transpose(-1, -2)

        # 2. 트리 레벨 수만큼 반복하여 전류를 위로 전파
        for _ in range(num_nodes):
            # 각 노드의 출력 전류(I_out)는 모든 자식 노드들의 수요(current_demands) 합
            i_out = (adj_matrix_T @ current_demands.unsqueeze(-1)).squeeze(-1)
            
            # 각 IC의 입력 전류(I_in) 계산
            op_current = next_obs["nodes"][..., FEATURE_INDEX["op_current"]]
            
            # LDO I_in = I_out + I_op
            i_in_ldo = i_out + op_current
            
            # Buck의 I_in = P_in / V_in + I_op = (P_out / eff) / V_in + I_op
            vout = next_obs["nodes"][..., FEATURE_INDEX["vout_min"]]
            vin = next_obs["nodes"][..., FEATURE_INDEX["vin_min"]]
            p_out_buck = vout * i_out
            # 환경 내에서는 복잡한 효율 곡선 대신 단순화된 고정 효율(예: 90%) 사용
            eff = 0.9 
            # vin이 0인 경우를 방지
            safe_vin = torch.where(vin > 0, vin, 1e-6)
            i_in_buck = (p_out_buck / eff) / safe_vin + op_current
            # 다음 반복을 위해 IC 노드의 수요를 새로 계산된 I_in 값으로 업데이트
            new_demands = current_demands.clone()
            ldo_mask_b = (next_obs["nodes"][..., FEATURE_INDEX["ic_type_idx"]] == 1.0)
            buck_mask_b = (next_obs["nodes"][..., FEATURE_INDEX["ic_type_idx"]] == 2.0)
            
            new_demands[ldo_mask_b] = i_in_ldo[ldo_mask_b]
            new_demands[buck_mask_b] = i_in_buck[buck_mask_b]
            
            # 더 이상 수요가 변하지 않으면(계산 완료) 루프 종료
            if torch.allclose(current_demands, new_demands):
                break
            current_demands = new_demands
            
        # 3. 최종적으로 계산된 current_out을 사용하여 전력 손실 및 온도 계산
        final_i_out = (adj_matrix_T @ current_demands.unsqueeze(-1)).squeeze(-1)
        next_obs["nodes"][..., FEATURE_INDEX["current_out"]] = final_i_out
        power_loss = self._calculate_power_loss(next_obs["nodes"], final_i_out)
        theta_ja = next_obs["nodes"][..., FEATURE_INDEX["theta_ja"]]
        ambient_temp = self.generator.config.constraints.get("ambient_temperature", 25.0)
        new_temp = ambient_temp + power_loss * theta_ja
        next_obs["nodes"][..., FEATURE_INDEX["junction_temp"]] = new_temp
        
        node_costs = next_obs["nodes"][:, :, FEATURE_INDEX["cost"]]
        # 이번 스텝으로 인해 *전체 비용*이 증가한 양
        previous_total_cost = (td["is_used_ic_mask"].float() * node_costs).sum(dim=1, keepdim=True)
        new_total_cost = (next_obs["is_used_ic_mask"].float() * node_costs).sum(dim=1, keepdim=True)
        total_cost_increase = new_total_cost - previous_total_cost # (B, 1)

        # [Find Parent] 모드였던 인스턴스에 대해서만 R_action, R_path 적용
        if head_is_node.any():
            # 3a. [공통] staging_cost에 비용 증가분을 누적
            next_obs["staging_cost"][node_rows] += total_cost_increase[node_rows]

            # 3b. R_action (액션별 비용) 보상을 스텝 보상에 추가
            #    (total_cost_increase는 (B,1) -> (B_act,)로 변환)
            step_reward[node_rows] += REWARD_WEIGHT_ACTION * (-total_cost_increase[node_rows].squeeze(-1))

            # 3c. R_path (경로별 비용) 보상
            finished_rows = node_rows[parent_is_battery]
            if finished_rows.numel() > 0:
                next_obs["trajectory_head"][finished_rows, 0] = BATTERY_NODE_IDX
                next_obs["current_target_load"][finished_rows, 0] = -1

                # 경로 완성 시, 누적된 staging_cost를 R_path 보상으로 추가
                sub_trajectory_total_cost = next_obs["staging_cost"][finished_rows]
                step_reward[finished_rows] += REWARD_WEIGHT_PATH * (-sub_trajectory_total_cost.squeeze(-1))

                # staging_cost를 0으로 리셋하고, current_cost(최종비용)에 반영
                next_obs["current_cost"][finished_rows] += sub_trajectory_total_cost
                next_obs["staging_cost"][finished_rows] = 0.0

            # 3d. 경로가 진행 중인 인스턴스
            in_progress_rows = node_rows[~parent_is_battery]
            if in_progress_rows.numel() > 0:
                next_obs["trajectory_head"][in_progress_rows, 0] = parent_node[~parent_is_battery]
                # (보상은 이미 STEP_PENALTY + R_action 으로 설정됨)
        # --- 수정 완료 ---


        next_obs.set("step_count", td["step_count"] + 1)


        # 6. 종료 조건
        next_mask = self.get_action_mask(next_obs)
        is_stuck_or_finished = ~next_mask.any(dim=-1)
        all_loads_connected = (next_obs["unconnected_loads_mask"].sum(dim=1) == 0)
        trajectory_finished = (next_obs["trajectory_head"].squeeze(-1) == BATTERY_NODE_IDX)
        done_successfully = all_loads_connected & trajectory_finished
        max_steps = 2 * self.generator.num_nodes
        timed_out = (next_obs["step_count"] > max_steps).squeeze(-1)
        is_done = done_successfully | timed_out | is_stuck_or_finished
        next_obs["done"] = is_done.unsqueeze(-1)
        
        # --- 👇 [핵심 버그 수정 1] get_reward 호출 및 상태 덮어쓰기 ---
        final_reward = self.get_reward(
            next_obs,
            step_reward, # (B,) 텐서 (STEP_PENALTY + R_action + R_path)
            done_successfully,
            timed_out,
            is_stuck_or_finished
        )
        
        # 이미 'done'이었던 샘플들은 보상을 0으로 강제하고, 상태를 덮어쓰지 않습니다.
        if is_already_done.any():
            final_reward[is_already_done] = 0.0
            next_obs[is_already_done] = td[is_already_done]
        # --- 수정 완료 ---

        return TensorDict({
            "next": next_obs,
            "reward": final_reward.unsqueeze(-1),
            "done": next_obs["done"], # 'is_already_done' 샘플도 'done=True'로 유지됨
        }, batch_size=batch_size)
        
# 💡 *** 여기가 핵심 수정 부분입니다 (get_action_mask) ***
    def get_action_mask(self, td: TensorDict, debug: bool = False) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        self._ensure_buffers(td) # 맨 앞에서 버퍼 동기화
        
        batch_size, num_nodes, _ = td["nodes"].shape
        mask = torch.zeros(batch_size, num_nodes, dtype=torch.bool, device=self.device)
        current_head = td["trajectory_head"].squeeze(-1)
        
        reasons = {} # 디버그 이유 저장

        # --- 1. [Select New Load] 모드 마스킹 ---
        head_is_battery = (current_head == BATTERY_NODE_IDX)
        if head_is_battery.any():
            all_has_unconnected = td["unconnected_loads_mask"].any(dim=-1)
            is_active = head_is_battery & all_has_unconnected
            is_finished = head_is_battery & ~all_has_unconnected
            
            mask[is_active] = td["unconnected_loads_mask"][is_active]
            mask[is_finished, BATTERY_NODE_IDX] = True
            
            if debug:
                reasons = {"Unconnected Load": td["unconnected_loads_mask"]}
            # (중요) head_is_battery와 head_is_node는 상호 배타적이므로,
            # [Find Parent] 로직이 실행될 수 있도록 여기서 return하지 않습니다.


        # --- 2. [Find Parent] 모드 마스킹 (모든 조건을 한번에 계산) ---
        head_is_node = ~head_is_battery
        if head_is_node.any():
            b_idx_node = torch.where(head_is_node)[0]
            child_nodes = current_head[head_is_node]
            node_types_tensor = self.node_type_tensor # (N,)
            B_act = len(b_idx_node) # (B_act,)
            
            # --- [공통 마스크] ---
            # (1, N_nodes) -> (B_act, N_nodes)
            is_battery_mask = (self.arange_nodes.unsqueeze(0) == BATTERY_NODE_IDX).expand(B_act, -1)
            # (1, N_nodes) -> (B_act, N_nodes)
            not_load_parent = (self.node_type_tensor.unsqueeze(0) != NODE_TYPE_LOAD).expand(B_act, -1)
            # (B_act, 1) -> (B_act, N_nodes)
            not_self_parent = (self.arange_nodes.unsqueeze(0) != child_nodes.unsqueeze(1))

            
            # --- 1. 전압 호환성 ---
            # (B, N, N) -> (B_act, N, N)
            connectivity = td["connectivity_matrix"][b_idx_node] 
            # (B_act, 1, 1) -> (B_act, N, 1)
            child_indices_exp = child_nodes.view(-1, 1, 1).expand(-1, num_nodes, 1)
            # (B_act, N)
            volt_ok = torch.gather(connectivity, 2, child_indices_exp).squeeze(-1)

            # --- 2. 사이클 방지 ---
            # (B_act, N)
            path_mask = self._trace_path_batch(child_nodes, td["adj_matrix"][b_idx_node])
            cycle_ok = ~path_mask

            # --- 3. 전류 한계 ---
            nodes_slice = td["nodes"][b_idx_node] # (B_act, N, D)
            rows = torch.arange(B_act, device=self.device) # (B_act,)
            # (B_act, N)
            remaining_capacity = nodes_slice[:, :, FEATURE_INDEX["i_limit"]] - nodes_slice[:, :, FEATURE_INDEX["current_out"]]
            # (B_act,) -> (B_act, 1)
            child_current_draw = nodes_slice[rows, child_nodes, FEATURE_INDEX["current_active"]].unsqueeze(1)
            # (B_act, N)
            current_ok = (remaining_capacity >= child_current_draw) | is_battery_mask

            
            # --- 4. [버그 수정] 독립(Exclusive) 조건 ---
            # (a) 현재 Head(child_nodes)의 상태 식별
            # (B_act,)
            head_status = td["is_exclusive_mask"][b_idx_node, child_nodes]
            
            # [Point 2 Fix] Head가 로드인지 IC인지 구별
            # (B_act,)
            head_is_load = (node_types_tensor[child_nodes] == NODE_TYPE_LOAD)
            # (b) 후보 부모(Parent)의 상태 및 자식 유무 스캔
            # (B_act, N_nodes)
            parent_status = td["is_exclusive_mask"][b_idx_node]
            parent_is_exclusive = (parent_status > 0)
            
            load_start_idx = 1 + self.generator.num_ics
            load_end_idx = load_start_idx + self.generator.num_loads
            # (B_act, N_nodes) - 이 부모가 'IC' 자식을 가졌는가?
            has_ic_child = td["adj_matrix"][b_idx_node, :, 1:load_start_idx].any(dim=-1)
            # (B_act, N_nodes) - 이 부모가 'Load' 자식을 가졌는가?
            has_load_child = td["adj_matrix"][b_idx_node, :, load_start_idx:load_end_idx].any(dim=-1)
            # (B_act, N_nodes) - 부모가 *어떤* 자식이라도 가졌는가? (엣지의 합 > 0)
            parent_has_any_child = has_ic_child | has_load_child
            
            # (c) 님의 규칙 정의 (True = 위반)
            # 규칙 1: Head가 'Path' (로드든 IC든) -> 부모는 자식이 없어야 함.
            violation_Rule1 = (head_status.unsqueeze(-1) == 2) & parent_has_any_child
            # 규칙 2: Head가 'Supplier Load' -> 부모는 자식이 없어야 함.
            violation_Rule2 = ((head_status == 1) & head_is_load).unsqueeze(-1) & parent_has_any_child
            # 규칙 3: Head가 'Normal' (Load/IC) 또는 'Supplier IC' -> 부모는 Exclusive이면 안 됨.
            violation_Rule3 = ((head_status == 0) | ((head_status == 1) & ~head_is_load)).unsqueeze(-1) & parent_is_exclusive
            # (d) 위반 사항들을 종합 (True = 금지)
            violations = violation_Rule1 | violation_Rule2 | violation_Rule3
            
            # (e) 규칙 4 (Battery)는 항상 허용
            exclusive_ok = torch.logical_not(violations) | is_battery_mask
            # --- [버그 수정 완료] ---
            # --- 5. 최종 결합 ---
            can_be_parent = (
                not_load_parent & not_self_parent & volt_ok & cycle_ok & 
                current_ok & exclusive_ok 
            )

            # --- 6. Power Sequence (루프 필요) ---
            for j_idx, k_idx, f_flag in self.power_sequences:
                # Case 1: 현재 child가 'k'일 때 (k의 부모를 찾는 중)
                is_k_mask = (child_nodes == k_idx)
                if is_k_mask.any():
                    instances_to_check = torch.where(is_k_mask)[0]
                    b_idx_check = b_idx_node[instances_to_check]
                    adj_j = td["adj_matrix"][b_idx_check, :, j_idx]
                    parent_exists = adj_j.any(dim=-1)

                    if parent_exists.any():
                        b_constr = b_idx_check[parent_exists]
                        inst_constr = instances_to_check[parent_exists]
                        parent_of_j_idx = adj_j[parent_exists].long().argmax(-1)
                        
                        anc_mask = self._trace_path_batch(parent_of_j_idx, td["adj_matrix"][b_constr])
                        anc_mask[:, BATTERY_NODE_IDX] = False # 조상 마스크에서 배터리 제외
                        can_be_parent[inst_constr] &= ~anc_mask
                        
                        if f_flag == 1:
                            same_parent_mask = (self.arange_nodes == parent_of_j_idx.unsqueeze(1))
                            can_be_parent[inst_constr] &= ~same_parent_mask

                # Case 2: 현재 child가 'j'일 때 (j의 부모를 찾는 중)
                is_j_mask = (child_nodes == j_idx)
                if is_j_mask.any():
                    instances_to_check = torch.where(is_j_mask)[0]
                    b_idx_check = b_idx_node[instances_to_check]
                    adj_k = td["adj_matrix"][b_idx_check, :, k_idx]
                    parent_exists = adj_k.any(dim=-1)

                    if parent_exists.any():
                        b_constr = b_idx_check[parent_exists]
                        inst_constr = instances_to_check[parent_exists]
                        parent_of_k_idx = adj_k[parent_exists].long().argmax(-1)
                        
                        anc_mask = self._trace_path_batch(parent_of_k_idx, td["adj_matrix"][b_constr])
                        anc_mask[:, BATTERY_NODE_IDX] = False # 조상 마스크에서 배터리 제외
                        can_be_parent[inst_constr] &= ~anc_mask
                        
                        if f_flag == 1:
                            same_parent_mask = (self.arange_nodes == parent_of_k_idx.unsqueeze(1))
                            can_be_parent[inst_constr] &= ~same_parent_mask

            # 최종 마스크를 전체 배치 마스크에 적용
            mask[head_is_node] = can_be_parent

            if debug:
                # [Find Parent] 모드 이유를 덮어쓰기
                reasons.update({ 
                     "Not Load": not_load_parent,
                     "Not Self": not_self_parent,
                     "Volt OK": volt_ok,
                     "Cycle OK": cycle_ok,
                     "Current OK": current_ok,
                     "Exclusive OK": exclusive_ok, # 수정된 최종 로직
                     "Sequence OK": can_be_parent # Power Sequence까지 적용된 최종본
                 })

        # --- 3. 최종 반환 ---
        if debug:
            return {"mask": mask, "reasons": reasons}
            
        return mask # 디버그 모드가 아닐 때
    # 👈 [암전류] 헬퍼 함수 (OR-Tools 로직 기반)
    def _calculate_total_sleep_current(self, td: TensorDict) -> torch.Tensor:
        """
        성공한 샘플들(td)의 최종 암전류 합계를 계산합니다.
        """
        batch_size, num_nodes, _ = td["nodes"].shape
        adj_matrix = td["adj_matrix"].float()
        adj_matrix_T = adj_matrix.transpose(-1, -2) # (c, p)

        # 1. "Always-On" 상태를 배터리까지 전파 (B, N)
        always_on_loads = (td["nodes"][..., FEATURE_INDEX["always_on_in_sleep"]] == 1.0)
        always_on_nodes = always_on_loads.clone()
        always_on_nodes[:, BATTERY_NODE_IDX] = True # 배터리는 항상 AO
        
        for _ in range(num_nodes):
            # (B,N,N) @ (B,N,1) -> (B,N,1) -> (B,N)
            parents_mask = (adj_matrix_T @ always_on_nodes.float().unsqueeze(-1)).squeeze(-1).bool()
            if (parents_mask & ~always_on_nodes).sum() == 0: break
            always_on_nodes |= parents_mask
        
        # 2. IC 자체 암전류 소모 계산 (B, N)
        is_ao = always_on_nodes
        is_used = td["is_used_ic_mask"]
        # (B,N,N) @ (B,N,1) -> (B,N) : 내 부모(p)가 AO(is_ao)인가?
        parent_is_ao = (adj_matrix_T @ is_ao.float().unsqueeze(-1)).squeeze(-1).bool()

        op_current = td["nodes"][..., FEATURE_INDEX["op_current"]]
        quiescent_current = td["nodes"][..., FEATURE_INDEX["quiescent_current"]]
        shutdown_current = td["nodes"][..., FEATURE_INDEX["shutdown_current"]]
        
        # shutdown_current가 0(미정의)이면 quiescent_current 사용
        use_ishut_current = torch.where(shutdown_current > 1e-9, shutdown_current, quiescent_current)

        ic_self_sleep = torch.zeros(batch_size, num_nodes, device=self.device)
        
        # 규칙 1: IC가 AO 경로상에 있음 -> Iop 소모
        ic_self_sleep[is_ao & is_used] = op_current[is_ao & is_used]
        # 규칙 2: IC가 AO가 아니지만, 부모가 AO -> I_shut/Iq 소모
        ic_self_sleep[~is_ao & is_used & parent_is_ao] = use_ishut_current[~is_ao & is_used & parent_is_ao]
        # 규칙 3: 그 외 (부모도 AO 아님) -> 0 소모

        # 3. 로드 암전류 소모 계산 (B, N)
        # 원본 td["nodes"]가 오염되지 않도록 .clone() 사용
        load_sleep_draw_base = td["nodes"][..., FEATURE_INDEX["current_sleep"]].clone()
        load_sleep_draw = load_sleep_draw_base * always_on_nodes.float()
        # AO 경로가 아닌 로드는 전류 0
        load_sleep_draw[~always_on_nodes] = 0.0

        # 4. 전류 수요 전파 (LDO 방식: I_in = I_out + I_self)
        current_demands_sleep = load_sleep_draw + ic_self_sleep
        ic_mask = (td["nodes"][..., FEATURE_INDEX["node_type"][0] + NODE_TYPE_IC] == 1.0)
        
        for _ in range(num_nodes):
            # I_out = sum(children's I_in)
            i_out_sleep = (adj_matrix_T @ current_demands_sleep.unsqueeze(-1)).squeeze(-1)
            
            # I_in = I_self + I_out (for ICs)
            new_demands_sleep = load_sleep_draw + ic_self_sleep
            new_demands_sleep[ic_mask] += i_out_sleep[ic_mask] # IC의 수요에 I_out을 더함
            
            if torch.allclose(current_demands_sleep, new_demands_sleep):
                break
            current_demands_sleep = new_demands_sleep

        # 5. 배터리에서 나가는 총 암전류 계산
        battery_children_mask = adj_matrix[:, BATTERY_NODE_IDX, :] # (B, N)
        total_sleep_current = (current_demands_sleep * battery_children_mask).sum(dim=1)
        
        return total_sleep_current # (B,)


    
    # --- 👇 [핵심 5] get_reward 함수 시그니처 변경 ---
    def get_reward(self,
                   td: TensorDict,
                   step_reward: torch.Tensor, # _step에서 계산된 기본 보상
                   done_successfully: torch.Tensor,
                   timed_out: torch.Tensor,
                   is_stuck_or_finished: torch.Tensor) -> torch.Tensor:
        

        """
        보상을 계산합니다.
        - 기본 보상: _step에서 계산된 값 (스텝 페널티 + R_action + R_path)
        - 최종 보상: *실패* 페널티 (경로 실패, 암전류 위반)
        """

        reward = step_reward.clone()

        # --- 👇 [암전류] 암전류 제약 검사 ---
        if done_successfully.any():
            td_success = td[done_successfully]
            total_sleep_current = self._calculate_total_sleep_current(td_success)
            
            # scalar_prompt index 1 is max_sleep_current
            max_sleep_current = td_success["scalar_prompt_features"][:, 1]
            
            # --- 👇 [힌지 페널티 수정] ---
            # (B_success,)
            violation_amount = total_sleep_current - max_sleep_current
            # Hinge Loss: max(0, violation_amount)
            hinge_violation = torch.relu(violation_amount) # 0 미만(위반 안 함)은 0으로 처리

            # 증분형(Incremental) 페널티 계산 (양수 값)
            sleep_penalty = PENALTY_WEIGHT_SLEEP * hinge_violation
            
            # [Point 2 Fix] reward[done_successfully]에 직접 페널티 차감 (음수 보상에 더함)
            reward[done_successfully] -= sleep_penalty
            # --- [힌지 페널티] 수정 완료 ---

        # --- [암전류] 검사 완료 ---
        # R_fail (실패 페널티)
        # 실패 시, 이전까지의 보상을 모두 덮어쓰고 강력한 페널티를 부여

        failed = (timed_out | is_stuck_or_finished) & ~done_successfully
        if failed.any():
            reward[failed] = FAILURE_PENALTY            
        return reward # (B,) 텐서. 호출부(_step)에서 (B, 1)로 unsqueeze함.