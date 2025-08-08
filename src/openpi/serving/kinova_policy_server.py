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

# 모델 사전 로드
cfg = config.get_config("pi0_libero_low_mem_finetune")
checkpoint_dir = download.maybe_download("/app/checkpoints/pi0_libero_low_mem_finetune/my_experiment_hair_straightener_70_fast/29999")
#cfg = config.get_config("pi0_fast_libero_low_mem_finetune")
#checkpoint_dir = download.maybe_download("/app/checkpoints/pi0_fast_libero_low_mem_finetune/my_experiment/20000")
policy = policy_config.create_trained_policy(cfg, checkpoint_dir)

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

            input_dict = {
                "observation/image": preprocess(input_payload["image"]),
                "observation/wrist_image": preprocess(input_payload["wrist_image"]),
                "observation/state": np.array(input_payload["state"], dtype=np.float32),
                "prompt": np.array([input_payload["task"]])
            }

            output = policy.infer(input_dict)
            action = output["actions"][:10, :8].tolist()
            print('추론 시간: ', output["policy_timing"]["infer_ms"], 'ms')

            result = pickle.dumps(action)
            conn.sendall(struct.pack('>I', len(result)) + result)

    except Exception as e:
        print(f"❌ 예외 발생: {e}")

    finally:
        print("🔁 클라이언트 연결 종료. 서버 대기로 복귀.\n")
        conn.close()
