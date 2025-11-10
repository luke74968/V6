# transformer_solver/expert_dataset.py

import json
import torch
from torch.utils.data import Dataset
from tensordict import TensorDict
from tqdm import tqdm
from typing import Tuple, List

from .solver_env import PocatEnv
from .solver_env import BATTERY_NODE_IDX

# --- 👇 [신규] 커스텀 Collate 함수 ---
def expert_collate_fn(batch: List[Tuple[TensorDict, torch.Tensor]]) -> Tuple[TensorDict, torch.Tensor]:
    """
    Custom collate function to stack TensorDicts and Tensors from the ExpertReplayDataset.
    PyTorch의 default_collate는 TensorDict를 처리하지 못하므로 이 함수가 필요합니다.
    """
    # batch는 튜플 리스트입니다: [(td_batch_1, reward_batch_1), (td_batch_2, reward_batch_2), ...]
    
    # 1. 튜플 리스트를 두 개의 리스트로 분리합니다.
    td_list = [item[0] for item in batch]
    reward_list = [item[1] for item in batch]
    
    # 2. torch.stack()을 사용하여 TensorDict 리스트를 하나의 (B, ...) TensorDict로 묶습니다.
    batched_tds = torch.stack(td_list, dim=0)
    
    # 3. torch.cat()을 사용하여 보상 리스트를 (B, 1) 텐서로 묶습니다.
    # (각 reward는 [1, 1] 형태이므로 cat(dim=0)을 사용)
    batched_rewards = torch.cat(reward_list, dim=0)
    
    return batched_tds, batched_rewards


class ExpertReplayDataset(Dataset):
    """
    'generate_expert_data.py'로 생성된 "정답지" JSON 파일을 읽어옵니다.
    JSON 안의 'action_sequences'를 환경에서 한 스텝씩 "리플레이(Replay)"하여,
    모든 (상태, 최종_보상) 페어(pair)를 메모리에 로드하는 지도학습용 데이터셋입니다.
    """
    def __init__(self, expert_data_path: str, env: PocatEnv, device: str = "cpu"):
        self.env = env
        self.generator = env.generator
        self.device = device
        self.replay_buffer = []

        print(f"\n🧠 '정답지' 리플레이 데이터셋 생성 중...")
        print(f"   - 정답지 파일 로드: {expert_data_path}")
        
        try:
            with open(expert_data_path, 'r', encoding='utf-8') as f:
                expert_traces = json.load(f)
            if not isinstance(expert_traces, list):
                expert_traces = []
        except Exception as e:
            print(f"❌ '정답지' 파일 로드 실패: {e}")
            expert_traces = []

        # tqdm으로 리플레이 진행 상황 표시
        pbar = tqdm(expert_traces, desc="   - OR-Tools 경로 리플레이 중")
        for trace in pbar:
            config_file = trace["config_file"]
            target_reward = trace["target_reward"]
            action_sequences = trace["action_sequences"] # [ [207, 181, 0], [208, 176, 0], ... ]
            
            # 1. 정답지와 동일한 'config'로 환경 텐서 생성
            # (PocatEnv는 이미 올바른 config로 초기화되었지만, 
            #  V7(일반화)을 대비해 generator를 다시 호출하는 것이 더 견고함)
            try:
                # (주의: 이 로직은 generator가 config_file 경로를 받아 다시 초기화할 수 있다고 가정)
                # 현재 V6 구조에서는 self.generator를 그냥 사용해도 됩니다.
                # generator = PocatGenerator(config_file_path=config_file)
                generator = self.generator 
                
                # (B, 1) 크기의 보상 텐서 준비
                target_reward_tensor = torch.tensor([[target_reward]], dtype=torch.float32, device=self.device)

                # 2. 모든 경로(Load)를 순회
                for path_actions in action_sequences:
                    # 3. 환경 리셋
                    # (배치 크기 1로 새 문제지 생성)
                    td_initial = generator(batch_size=1).to(self.device)
                    td = self.env._reset(td_initial)
                    
                    # 4. '정답지'의 Bottom-Up 경로를 한 스텝씩 리플레이
                    # path_actions 예시: [207, 181, 0]
                    for action_idx in path_actions:
                        
                        # (A) 리플레이: 현재 상태(td)와 정답 보상(target_reward)을 버퍼에 저장
                        # .clone()으로 텐서의 현재 스냅샷을 저장
                        self.replay_buffer.append((td.clone(), target_reward_tensor.clone()))
                        
                        # (B) 다음 스텝으로 이동
                        action_tensor = torch.tensor([[action_idx]], dtype=torch.long, device=self.device)
                        td.set("action", action_tensor)
                        td = self.env.step(td)["next"]
                        
                        if td["done"].item():
                            # (경로 완성 (head=0) 또는 실패 시 루프 중단)
                            break
                            
            except Exception as e:
                print(f"❌ 리플레이 중 오류 발생 (Trace: {config_file}): {e}")
                
        if not self.replay_buffer:
            print("⚠️ 경고: '정답지' 리플레이 결과, 유효한 (상태, 보상) 데이터가 0개입니다.")
        else:
            print(f"✅ '정답지' 리플레이 완료. 총 {len(self.replay_buffer)}개의 (상태, 보상) 페어 생성.")

    def __len__(self) -> int:
        return len(self.replay_buffer)

    def __getitem__(self, idx: int) -> Tuple[TensorDict, torch.Tensor]:
        # (state_td, target_reward) 반환
        # [BUG FIX] .squeeze(0)를 제거해야 PyTorch의 default_collate 함수가
        # (B=1, ...) 텐서들을 (B=batch_size, ...)로 올바르게 stack/collate 할 수 있습니다.
        # .squeeze(0)를 하면 non-batch 텐서가 되어 collate가 __iter__를 호출해 StopIteration이 발생합니다.
        return self.replay_buffer[idx] # (td [B=1], reward [B=1])