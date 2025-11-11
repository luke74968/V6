# transformer_solver/debug_env.py
import torch
import argparse
import sys
import os

# 프로젝트 루트 경로를 sys.path에 추가하여 모듈을 찾을 수 있도록 함
# 💡 [수정] 경로 추가 로직을 좀 더 안정적으로 변경
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformer_solver.solver_env import PocatEnv, BATTERY_NODE_IDX
from transformer_solver.definitions import FEATURE_INDEX, NODE_TYPE_IC

def run_interactive_debugger(config_file):
    """대화형으로 Power Tree를 만들며 마스킹 로직을 디버깅하는 스크립트"""
    
    # 1. 환경 초기화
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = PocatEnv(generator_params={"config_file_path": config_file}, device=device)
    td = env.reset(batch_size=1)
    
    node_names = env.generator.config.node_names
    num_nodes = env.generator.num_nodes # 💡 [신규] num_nodes 추가
    node_name_to_idx = {name: i for i, name in enumerate(node_names)}
    
    # 💡 [신규] 노드 타입 피처 인덱스 미리 정의
    nt_s, nt_e = FEATURE_INDEX["node_type"]

    print("="*50)
    print("🚀 POCAT Transformer Interactive Debugger 🚀")
    print(f"Config: {config_file}")
    print("목표: OR-Tools의 최적해 경로를 따라가며 마스킹이 올바른지 확인하세요.")
    print("액션은 노드의 '이름'으로 입력합니다 (예: MCU_Main, LDO_X_Gen@5.0Vin_3.3Vout_copy1).")
    print("'exit'를 입력하면 종료합니다.")
    print("="*50)

    step = 0
    while not td["done"].all():
        step += 1
        current_head_idx = td["trajectory_head"].item()
        current_head_name = node_names[current_head_idx]
        
        print(f"\n--- Step {step} ---")
        if current_head_idx == BATTERY_NODE_IDX:
            print("🌲 Head: 🔋 Battery (Action: Select a new load)")
        else:
            # --- [신규] 현재 헤드의 소모 전류(mA)를 가져옵니다 ---
            current_active = td["nodes"][0, current_head_idx, FEATURE_INDEX["current_active"]].item()
            current_active_ma = current_active * 1000
            target_load_idx = td["current_target_load"].item()
            target_load_name = "None" if target_load_idx == -1 else node_names[target_load_idx]
            print(f"🌲 Head: 🔌 {current_head_name} (Consumes: {current_active_ma:.1f}mA) (Targeting: {target_load_name})")
            print(f"🌲 Head: 🔌 {current_head_name} (Targeting: {target_load_name})")
            print("Action: Find a parent for the current head")
            
        # 2. 마스킹 정보 가져오기 (debug=True 사용)
        mask_info = env.get_action_mask(td, debug=True)
        final_mask = mask_info["mask"][0]
        reasons = mask_info["reasons"]
        
        valid_actions = []
        
        # 3. 모든 노드에 대한 마스킹 결과와 이유 출력
        print("\n--- Masking Details ---")
        
        # --- [FIX] "Select New Load" 모드인지 "Find Parent" 모드인지 확인 ---
        is_find_parent_mode = "Not Load" in reasons
        
        if not is_find_parent_mode:
             print(f"{'Node Name':<50} | {'VALID?':<8}")
             print("-" * 61)
        else:
            # 1. 헤더 정의 ("Find Parent" 모드일 때만)
            new_cols = ["I_now(mA)", "Tj_now(C)", "I_sim(mA)", "Tj_sim(C)"]
            col_widths = [13, 10, 13, 10] # 💡 6자리 소수점을 위해 너비 증가
            # 💡 [FIX] bool_reasons를 여기서 정의
            bool_reasons = [k for k in reasons.keys() if k not in ["Sim I_out", "Sim Tj"]]
            
            header_parts = [f"{'Node Name':<50}", f"{'VALID?':<8}"]
            header_parts.extend(f"{k:<12}" for k in bool_reasons)
            header_parts.extend(f"{k:<{w}}" for k, w in zip(new_cols, col_widths))
            
            header = " | ".join(header_parts)
            print(header)
            print("-" * len(header))

        for idx, name in enumerate(node_names):
            is_valid = final_mask[idx].item()
            if is_valid:
                valid_actions.append(name)

            # --- [FIX] 모드에 따라 분기 ---
            if not is_find_parent_mode:
                # "Select New Load" 모드
                if is_valid: 
                    print(f"{name:<50} | {'✅ YES':<8}")
                continue # 👈 [중요] 다음 노드로 바로 넘어감
            else:
                # "Find Parent" 모드
                
                # 1. 현재 값 가져오기
                current_i_out = td["nodes"][0, idx, FEATURE_INDEX["current_out"]].item()
                current_tj = td["nodes"][0, idx, FEATURE_INDEX["junction_temp"]].item()
                
                # 2. boolean 이유 문자열 생성 (bool_reasons가 여기서 보장됨)
                reason_str_parts = []
                for k in bool_reasons:
                    tensor = reasons[k]
                    # debug_env는 B=1 이므로 [0, idx]로 접근
                    value = tensor[0, idx].item() 
                    reason_str_parts.append(f"{('✅' if value else '❌'):<12}")
                reason_str = " | ".join(reason_str_parts)
                
                # 3. 시뮬레이션 값 가져오기
                sim_i_out = reasons.get("Sim I_out", torch.empty(1, num_nodes).fill_(-1.0))[0, idx].item()
                sim_tj = reasons.get("Sim Tj", torch.empty(1, num_nodes).fill_(-1.0))[0, idx].item()
                
                # 4. 시뮬레이션 값 포맷팅 (시뮬레이션 안 한 노드는 '----' 표시)
                sim_i_str = f"{sim_i_out*1000:10.6f}" if sim_i_out != -1.0 else "----"
                sim_tj_str = f"{sim_tj:7.1f}" if sim_tj != -1.0 else "----"
                
                # 5. 현재 값 포맷팅 (IC 노드만 의미 있으므로 IC만 표시)
                node_type = td["nodes"][0, idx, nt_s:nt_e].argmax().item()
                is_ic = (node_type == NODE_TYPE_IC)
                
                curr_i_str = f"{current_i_out*1000:10.6f}" if is_ic else "----"
                curr_tj_str = f"{current_tj:7.1f}" if is_ic else "----"

                # 6. 최종 행 출력
                row_parts = [
                    f"{name:<50}",
                    f"{('✅ YES' if is_valid else '❌ NO'):<8}",
                    f"{reason_str}",
                    f"{curr_i_str:<13}",
                    f"{curr_tj_str:<10}",
                    f"{sim_i_str:<13}",
                    f"{sim_tj_str:<10}"
                ]
                print(" | ".join(row_parts))
                
        # ... (이하 valid_actions 출력 및 사용자 입력 로직은 동일) ...

        print("\n--- Valid Actions ---")
        if not valid_actions:
            print("❌ No valid actions found! The environment is stuck.")
            break
            
        for name in sorted(valid_actions):
            print(f"- {name}")

        # 4. 사용자로부터 액션 입력받기
        while True:
            action_name = input("\nEnter action (node name): ")
            if action_name.lower() == 'exit':
                print("Debugger terminated.")
                return
                
            if action_name in valid_actions:
                action_idx = node_name_to_idx[action_name]
                break
            else:
                print(f"❌ Invalid action '{action_name}'. Please choose from the 'Valid Actions' list.")

        # 5. 환경 스텝 실행
        action_tensor = torch.tensor([[action_idx]], dtype=torch.long, device=device)
        td.set("action", action_tensor)
        output = env.step(td)
        td = output["next"]

    print("\n🎉 Power Tree construction finished!")
    final_reward = output['reward'].item() if output['reward'].numel() == 1 else output['reward'][0].item()
    print(f"Final Cost: ${-final_reward:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive Debugger for POCAT Transformer Env")
    parser.add_argument("config_file", type=str, help="Path to the configuration file (.json) to debug.")
    args = parser.parse_args()
    
    run_interactive_debugger(args.config_file)