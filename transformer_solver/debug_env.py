# transformer_solver/debug_env.py
import torch
import argparse
import sys
import os

# 프로젝트 루트 경로를 sys.path에 추가하여 모듈을 찾을 수 있도록 함
# 💡 [수정] 경로 추가 로직을 좀 더 안정적으로 변경
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformer_solver.solver_env import PocatEnv, BATTERY_NODE_IDX

def run_interactive_debugger(config_file):
    """대화형으로 Power Tree를 만들며 마스킹 로직을 디버깅하는 스크립트"""
    
    # 1. 환경 초기화
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = PocatEnv(generator_params={"config_file_path": config_file}, device=device)
    td = env.reset(batch_size=1)
    
    node_names = env.generator.config.node_names
    node_name_to_idx = {name: i for i, name in enumerate(node_names)}

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
            target_load_idx = td["current_target_load"].item()
            target_load_name = "None" if target_load_idx == -1 else node_names[target_load_idx]
            print(f"🌲 Head: 🔌 {current_head_name} (Targeting: {target_load_name})")
            print("Action: Find a parent for the current head")
            
        # 2. 마스킹 정보 가져오기 (debug=True 사용)
        mask_info = env.get_action_mask(td, debug=True)
        final_mask = mask_info["mask"][0]
        reasons = mask_info["reasons"]
        
        valid_actions = []
        
        # 3. 모든 노드에 대한 마스킹 결과와 이유 출력
        print("\n--- Masking Details ---")
        
        # 💡 [수정] reasons 딕셔너리가 비어있지 않은지 확인 (중요)
        if not reasons:
             print(f"{'Node Name':<50} | {'VALID?':<8}")
             print("-" * 61)
        else:
            header = f"{'Node Name':<50} | {'VALID?':<8} | " + " | ".join(f"{k:<12}" for k in reasons.keys())
            print(header)
            print("-" * len(header))

        for idx, name in enumerate(node_names):
            is_valid = final_mask[idx].item()
            if is_valid:
                valid_actions.append(name)

            # 💡 [수정] reasons가 비어있을 때(예: [Select Load] 모드)와
            #          reasons가 있을 때([Find Parent] 모드)를 분리하여 처리
            if not reasons:
                if is_valid: # [Select Load] 모드일 경우 유효한 것만 출력
                    print(f"{name:<50} | {'✅ YES':<8}")
                continue
            else:
                # --- 👇 [핵심 버그 수정] ---
                # reasons[k][idx] -> reasons[k][0, idx]로 수정
                reason_str = " | ".join(f"{('✅' if reasons[k][0, idx] else '❌'):<12}" for k in reasons.keys())
                # --- 수정 완료 ---
                print(f"{name:<50} | {('✅ YES' if is_valid else '❌ NO'):<8} | {reason_str}")

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
    # 💡 [수정] reward가 스칼라가 아닐 수 있으므로 .item() 추가
    final_reward = output['reward'].item() if output['reward'].numel() == 1 else output['reward'][0].item()
    print(f"Final Cost: ${-final_reward:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive Debugger for POCAT Transformer Env")
    parser.add_argument("config_file", type=str, help="Path to the configuration file (.json) to debug.")
    args = parser.parse_args()
    
    run_interactive_debugger(args.config_file)