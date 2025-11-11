# V6/generate_expert_data.py
# (V6/transformer_solver/ 및 V6/or_tools_solver/ 와 동일한 위치에 저장)

import json
import sys
import os
import argparse
from dataclasses import asdict
from collections import defaultdict
from ortools.sat.python import cp_model
import torch # PocatGenerator가 torch.device를 사용하므로 임포트 필요

# --- 1. OR-Tools Solver 모듈 임포트 ---
from or_tools_solver.config_loader import load_configuration_from_file
from or_tools_solver.core import expand_ic_instances, create_solver_model
from common.ic_preprocessor import prune_dominated_ic_instances
# --- 👇 [신규] 시각화 함수 임포트 ---
from or_tools_solver.solution_visualizer import print_and_visualize_one_solution

# --- 2. Transformer Solver 모듈 임포트 (매핑 및 후처리용) ---
from transformer_solver.env_generator import PocatGenerator
from transformer_solver.solver_env import REWARD_WEIGHT_PATH, BATTERY_NODE_IDX
from transformer_solver.definitions import FEATURE_INDEX, NODE_TYPE_LOAD


def generate_expert_solution(config_filename, output_filename, max_sleep_override=None):
    """
    OR-Tools를 실행하여 최적해를 찾고, 
    Transformer가 학습할 수 있는 '정답지' (Bottom-Up 액션 시퀀스 + 최종 보상)를 생성합니다.
    (최적해의 시각화 .png 파일도 함께 생성합니다)
    """
    print(f"📖 설정 파일 '{config_filename}' 로딩...")
    
    # 1. 설정 로드 (OR-Tools 기준)
    try:
        battery, available_ics, loads, constraints = load_configuration_from_file(config_filename)
    except FileNotFoundError:
        print(f"❌ 오류: 설정 파일 '{config_filename}'을(를) 찾을 수 없습니다.", file=sys.stderr)
        return
        
    if max_sleep_override is not None:
        constraints['max_sleep_current'] = max_sleep_override
        print(f"   - 암전류 제약 오버라이드: {max_sleep_override} A")

    # 2. 전처리 (or_tools_solver/main.py와 동일)
    candidate_ics, ic_groups = expand_ic_instances(available_ics, loads, battery, constraints)
    
    print("\n🔪 Dominance Pruning 전처리 시작...")
    candidate_ics_dicts = [asdict(ic) for ic in candidate_ics]
    pruned_ics_dicts, _ = prune_dominated_ic_instances(candidate_ics_dicts)
    
    candidate_ics_map = {ic.name: ic for ic in candidate_ics}
    pruned_candidate_names = {ic_dict['name'] for ic_dict in pruned_ics_dicts}
    candidate_ics = [ic for name, ic in candidate_ics_map.items() if name in pruned_candidate_names]
    
    original_count = len(candidate_ics_dicts)
    pruned_count = len(candidate_ics)
    print(f"   - {original_count - pruned_count}개 IC 인스턴스 제거 완료. (남은 후보: {pruned_count}개)")

    sanitized_ic_groups = {}
    # --- 👇 [버그 수정] .values() -> .items() ---
    for group_key, group_list in ic_groups.items():
        sanitized_group_list = [name for name in group_list if name in pruned_candidate_names]
        if len(sanitized_group_list) > 1:
            sanitized_ic_groups[group_key] = sanitized_group_list

    # 3. OR-Tools 모델 생성 및 최적해 탐색
    print("\n🧠 OR-Tools 모델 생성 및 최적해 탐색 시작...")
    model, edges, ic_is_used = create_solver_model(candidate_ics, loads, battery, constraints, sanitized_ic_groups)
    
    solver = cp_model.CpSolver()
    solver.parameters.log_search_progress = True
    solver.parameters.max_time_in_seconds = 300.0 # 5분 타임아웃
    
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print("\n❌ OR-Tools가 유효한 솔루션을 찾지 못했습니다.", file=sys.stderr)
        return

    print(f"\n🎉 탐색 완료! (상태: {solver.StatusName(status)})")
    
    # OR-Tools가 찾은 최적해 (이름 기반)
    base_solution = {
        "cost": solver.ObjectiveValue() / 10000, # Cost (예: 14.38)
        "active_edges": [(p, c) for (p, c), var in edges.items() if solver.Value(var)],
        # (시각화 함수가 'used_ic_names'도 사용하므로 추가)
        "used_ic_names": {name for name, var in ic_is_used.items() if solver.Value(var)},
    }

    # --- 👇 [신규] 4. "정답지" 시각화 (요청 사항) ---
    print("\n🖼️ '정답지' 솔루션 시각화 생성...")
    
    # [신규] JSON 파일이 저장될 'expert_data' 폴더 경로를 추출
    visualization_dir = os.path.dirname(output_filename)
    if not visualization_dir: # output_filename이 'dataset.json'처럼 경로 없이 파일명만 있을 경우
        visualization_dir = "." # 현재 폴더에 저장


    print_and_visualize_one_solution(
        base_solution, 
        candidate_ics, 
        loads, 
        battery, 
        constraints, 
        solution_index=0, # (정답지는 0번 인덱스로 저장)
        custom_output_dir=visualization_dir
    )
    print(f"   - 시각화 이미지 저장 완료. (경로: {visualization_dir})")
    # --- [신규] 시각화 완료 ---


    # --- 5. [핵심] Transformer용 "정답지" 변환 ---
    print(f"\n💾 '정답지' 데이터 생성 시작 (Transformer 매핑)...")
    
    # 5a. Transformer의 노드-인덱스 매핑 로드
    print(f"   - Transformer의 노드-인덱스 매핑 로드 중...")
    transformer_generator = PocatGenerator(config_file_path=config_filename)
    node_name_to_idx = {name: i for i, name in enumerate(transformer_generator.config.node_names)}
    
    # 5b. OR-Tools 엣지(이름) -> Transformer 엣지(인덱스) 변환
    try:
        active_edges_indices = [
            (node_name_to_idx[p], node_name_to_idx[c]) 
            for p, c in base_solution['active_edges']
        ]
    except KeyError as e:
        print(f"❌ 이름-인덱스 매핑 실패: {e}.", file=sys.stderr)
        print("   OR-Tools와 Transformer의 노드 이름이 일치하는지 확인하세요.", file=sys.stderr)
        return

    # 5c. Transformer가 사용할 최종 보상(Reward) 계산
    target_reward = base_solution['cost'] * -REWARD_WEIGHT_PATH

    # 5d. 엣지 리스트를 역방향 맵(child -> parent)으로 변환
    child_to_parent_map = {c_idx: p_idx for p_idx, c_idx in active_edges_indices}

    # 5e. Transformer의 로드(Leaf) 노드 찾기
    load_start_idx = 1 + transformer_generator.num_ics
    load_end_idx = load_start_idx + transformer_generator.num_loads
    load_indices = list(range(load_start_idx, load_end_idx))

    transformer_action_sequence = []
    
    # 5f. 각 로드부터 배터리까지 역추적하여 "경로" 생성
    for load_idx in load_indices:
        
        if load_idx not in child_to_parent_map:
            continue
            
        current_node_idx = load_idx
        path_actions = []
        
        path_actions.append(current_node_idx) 
        
        while current_node_idx in child_to_parent_map:
            parent_node_idx = child_to_parent_map[current_node_idx]
            path_actions.append(parent_node_idx)
            current_node_idx = parent_node_idx
            
            if current_node_idx == BATTERY_NODE_IDX: # 0 = BATTERY_NODE_IDX
                break
        
        transformer_action_sequence.append(path_actions)
        
    print(f"   - {len(load_indices)}개 로드를 {len(transformer_action_sequence)}개의 액션 시퀀스로 변환 완료.")

    expert_data_entry = {
        "config_file": config_filename,
        "cost": base_solution['cost'],
        "target_reward": target_reward,
        "action_sequences": transformer_action_sequence
    }

    # 5g. JSON 파일로 저장 (덮어쓰기 대신 '추가' 방식)
    output_dir = os.path.dirname(output_filename)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    data_list = []
    if os.path.exists(output_filename):
        try:
            with open(output_filename, 'r', encoding='utf-8') as f:
                data_list = json.load(f)
            if not isinstance(data_list, list):
                data_list = []
        except json.JSONDecodeError:
            data_list = []
            
    data_list.append(expert_data_entry)
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, indent=2)
    
    print(f"✅ '정답지' 1개를 {output_filename}에 추가했습니다. (현재 파일에 총 {len(data_list)}개)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pocat Expert Data Generator (OR-Tools to Transformer)")
    parser.add_argument(
        "--config_file", 
        type=str, 
        required=True, 
        help="Path to the configuration file (.json) to solve. (예: configs/config_6.json)"
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        required=True, 
        help="Path to the output expert data JSON file (e.g., expert_data/dataset.json)."
    )
    parser.add_argument(
        "--max_sleep_current", 
        type=float, 
        default=None, 
        help="Override the max_sleep_current constraint (in Amperes)."
    )
    args = parser.parse_args()
    
    generate_expert_solution(args.config_file, args.output_file, args.max_sleep_current)