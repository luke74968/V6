0. Ubuntu-24.04 에서 구현
# python3-venv 설치 필요 
sudo apt install -y python3-venv
1. 초기 셋업
# 가상환경 생성
python3 -m venv .venv
# 가상환경 활성화
# window용 
# 현재 터미널의 정책만 임시로 변경
Set-ExecutionPolicy Bypass -Scope Process
# 가상 환경 활성화
.\.venv\Scripts\activate.ps1
# Linux용 
# 가상 환경 활성화
source .venv/bin/activate
# PyTorch 설치 (CUDA 12.9 기준)
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu129
# 기타 라이브러리 설치 
pip install -r requirements.lock.txt


2. 실행
# OR-TOOLS Solver 실행
# (가상환경 .venv가 활성화된 상태에서)
# (pocat_solver_v6/ 폴더에서 실행)
# python -m [모듈이름] [설정파일] [옵션]
python3 -m or_tools_solver.main configs/config_6.json --max_sleep_current 0.001
python -m or_tools_solver.main config.json --max_sleep_current 0.01


# Tronsformer based solver 학습 실행
# 예시 1 (상세 로그)
python3 -m transformer_solver.run --config_file configs/config_6.json --config_yaml configs/config.yaml --batch_size 1 --log_idx 8 --log_mode detail --decode_type sampling
# 예시 2 (진행률)
python3 -m transformer_solver.run --config_file configs/config_6.json --config_yaml configs/config.yaml --batch_size 256 --log_idx 8 --log_mode progress --decode_type sampling
# 예시 3 (POMO 샘플링)
python3 -m transformer_solver.run --config_file configs/config_6.json --config_yaml configs/config.yaml --batch_size 2 --log_idx 0 --log_mode progress --decode_type sampling --num_pomo_samples 48


# 학습된 결과로 Tronsformer based solver test 실행
# 예시 1
python3 -m transformer_solver.run --test_only --config_file configs/config_4.json --log_mode detail --log_idx 0 --load_path "transformer_solver/result/2025-0923-174528/epoch-5.pth"
# 예시 2 (best_cost.pth 사용)
python3 -m transformer_solver.run --test_only --config_file configs/config_6.json --log_mode detail --log_idx 0 --load_path "transformer_solver/result/2025-1013-133337/best_cost.pth"


# 디버그 모드 
# 대화형으로 마스킹 로직을 테스트합니다.
python3 -m transformer_solver.debug_env configs/config_6.json

# git hub 사용방법
# git add . or git add 파일명
# git commit -m "description"
# git push origin master