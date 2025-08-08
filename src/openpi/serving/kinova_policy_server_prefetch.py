import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import socket
import pickle
import struct
import numpy as np
import cv2

from openpi.training import config
from openpi.policies import policy_config
from openpi.shared import download

import threading
import queue
import time

# 프롬프트 → 체크포인트 경로 맵핑
TASK_TO_CKPT = {
    "pick up warm water from floor": "/app/checkpoints/pi0_libero_low_mem_finetune/my_experiment_tm_all/48000",
    "place warm water to the right side of empty plate": "/app/checkpoints/pi0_libero_low_mem_finetune/my_experiment_tm_all/48000",
    "pick up strawberry from fruit plate": "/app/checkpoints/pi0_libero_low_mem_finetune/my_experiment_tm_all/48000",
    "place strawberry to the left side of empty plate": "/app/checkpoints/pi0_libero_low_mem_finetune/my_experiment_tm_all/48000",
    # ... 더 추가
}

MAX_POLICY_CACHE = 2

class PolicyManager:
    def __init__(self):
        self.cache = {}
        self.usage_queue = queue.Queue()  # 순서 관리

    def get_policy(self, ckpt_path):
        # 이미 로드되어 있으면 사용
        if ckpt_path in self.cache:
            # 사용 순서 갱신
            self.usage_queue.queue.remove(ckpt_path)
            self.usage_queue.put(ckpt_path)
            return self.cache[ckpt_path]
        # 새로 로드
        if len(self.cache) >= MAX_POLICY_CACHE:
            # 오래된 policy 메모리에서 내리기
            to_remove = self.usage_queue.get()
            print(f"💡 Unloading policy: {to_remove}")
            del self.cache[to_remove]
        print(f"💡 Loading policy: {ckpt_path}")
        cfg = config.get_config("pi0_libero_low_mem_finetune")
        policy = policy_config.create_trained_policy(cfg, ckpt_path)
        self.cache[ckpt_path] = policy
        self.usage_queue.put(ckpt_path)
        return policy

policy_manager = PolicyManager()

# 소켓 서버 설정
HOST = '127.0.0.1'
PORT = 12345

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)
print(f"📡 서버 대기 중... 포트 {PORT}")

def recvall(sock, count):
    buf = b""
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

def preprocess(img):
    #return cv2.resize(img,(256, 256));
    return img

while True:
    print("⏳ 클라이언트 접속 대기 중...")
    conn, addr = server_socket.accept()
    print(f"✅ [접속됨] 클라이언트 {addr}")
    try:
        while True:
            data_len = recvall(conn, 4)
            if not data_len:
                print("🔌 클라이언트 연결 종료 감지")
                break
            msg_len = struct.unpack('>I', data_len)[0]
            data = recvall(conn, msg_len)
            if not data:
                print("⚠️ 데이터 수신 실패")
                break

            input_payload = pickle.loads(data)
            prompt = input_payload["task"]
            next_prompt = input_payload.get("next_task", None)
            print(f"🔍 프롬프트: {prompt}, 다음 프롬프트: {next_prompt}")

            # 현재 태스크 체크포인트
            ckpt_path = TASK_TO_CKPT.get(prompt)
            if ckpt_path is None:
                print(f"❌ 등록되지 않은 태스크: {prompt}")
                continue

            # 다음 태스크 체크포인트 미리 로드 (background로!)
            if next_prompt:
                next_ckpt = TASK_TO_CKPT.get(next_prompt)
                if next_ckpt and next_ckpt not in policy_manager.cache:
                    def prefetch_policy(path):
                        # 0.5초 지연 넣어서 race condition 방지 (optional)
                        time.sleep(0.5)
                        try:
                            policy_manager.get_policy(path)
                        except Exception as e:
                            print(f"❗ Prefetch 실패: {e}")
       
                    threading.Thread(target=prefetch_policy, args=(next_ckpt,), daemon=True).start()

            # 현재 policy 인퍼런스
            policy = policy_manager.get_policy(ckpt_path)
            input_dict = {
                "observation/image": preprocess(input_payload["image"]),
                "observation/wrist_image": preprocess(input_payload["wrist_image"]),
                "observation/state": np.array(input_payload["state"], dtype=np.float32),
                "prompt": np.array([prompt])
            }
            output = policy.infer(input_dict)
            action = output["actions"][:45:3, :8].tolist()

            result = pickle.dumps(action)
            conn.sendall(struct.pack('>I', len(result)) + result)

    except Exception as e:
        print(f"❌ 예외 발생: {e}")
    finally:
        print("🔁 클라이언트 연결 종료. 서버 대기로 복귀.\n")
        conn.close()