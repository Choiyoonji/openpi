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
# cfg = config.get_config("pi0_libero_low_mem_finetune")
# checkpoint_dir = download.maybe_download("/app/checkpoints/pi0_libero_low_mem_finetune/my_experiment_tm_li_90/29999")
# policy = policy_config.create_trained_policy(cfg, checkpoint_dir)

cfg_list = [
    "pi0_libero_low_mem_finetune",
    "pi0_libero_low_mem_finetune2",
    "pi0_libero_low_mem_finetune3",
]

# checkpoint_dir_list = [
#     download.maybe_download("/app/checkpoints/all_se_checkpoints/my_experiment_se_tu_50/29999"),
#     download.maybe_download("/app/checkpoints/all_se_checkpoints/my_experiment_se_pi_50/29999"),
#     download.maybe_download("/app/checkpoints/all_se_checkpoints/my_experiment_se_pl_50/29999"),
# ]

checkpoint_dir_list = [
    download.maybe_download("/app/checkpoints/e50/my_experiment_tm_se_tu_50/29999"),
    download.maybe_download("/app/checkpoints/e50/my_experiment_tm_se_pi_50/29999"),
    download.maybe_download("/app/checkpoints/e50/my_experiment_tm_se_pl_50/29999"),
]


policy_list = [
    policy_config.create_trained_policy(config.get_config(cfg_list[0]), checkpoint_dir_list[0]),
    policy_config.create_trained_policy(config.get_config(cfg_list[1]), checkpoint_dir_list[1]),
    policy_config.create_trained_policy(config.get_config(cfg_list[2]), checkpoint_dir_list[2]),
]

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
                # "observation/right_wrist_image": preprocess(input_payload["right_image"]),
                "observation/state": np.array(input_payload["state"], dtype=np.float32),
                "prompt": np.array([input_payload["task"]])
            }

            policy = policy_list[input_payload["task_index"]]  # 클라이언트로부터 받은 정책 인덱스에 따라 정책 선택
            output = policy.infer(input_dict)
            action = output["actions"][:, :8].tolist()
            print(input_payload["task_index"], ' 추론 시간: ', output["policy_timing"]["infer_ms"], 'ms')

            result = pickle.dumps(action)
            conn.sendall(struct.pack('>I', len(result)) + result)

    except Exception as e:
        print(f"❌ 예외 발생: {e}")

    finally:
        print("🔁 클라이언트 연결 종료. 서버 대기로 복귀.\n")
        conn.close()
