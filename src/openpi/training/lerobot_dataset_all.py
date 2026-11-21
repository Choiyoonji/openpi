import json
import cv2
from pathlib import Path
from PIL import Image
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, HF_LEROBOT_HOME
import shutil

# 데이터셋 경로 설정
base_root = Path("/media/choiyj/mnt_store/Combined")
REPO_NAME = "physical-intelligence/kinova/pour_into_hot_co" # libero_(tm or rgb)_(li or se or ...)_(pl or pi or tu or all)_(70 or 50 or 30)
output_path = HF_LEROBOT_HOME / REPO_NAME
target_task = "Empty_the_scoop_into_the_hot_water" # 하위 작업 폴더명

target_mega = "Make_a_green_tea"  # None or "Make_a_vitamin_drink"  # 대분류 폴더명 설정, None이면 전체 탐색

if output_path.exists():
    shutil.rmtree(output_path)

dataset = LeRobotDataset.create(
    repo_id=REPO_NAME,
    robot_type="kinova_gen3",
    fps=15,
    features={
        "image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
        "wrist_image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
        "right_wrist_image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
        "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
        "actions": {"dtype": "float32", "shape": (8,), "names": ["actions"]},
    },
    #PRE TODO: 1,0ìœ¼ë¡œ ë°”ê¿”ì„œ ë„ì „í•´ë³¼ê²ƒ
    image_writer_threads=10,
    image_writer_processes=5,
)

for task_dir in base_root.iterdir():
    if not task_dir.is_dir():
        continue

    if target_mega is not None and task_dir.name != target_mega:
        continue
    
    for dataset_dir in task_dir.glob("dataset_*"):
        json_path = dataset_dir / target_task / "robot_data.json"

        cam1_path = dataset_dir / "camera1.mp4"
        cam2_path = dataset_dir / "camera2.mp4"
        cam3_path = dataset_dir / "camera3.mp4"

        if not (json_path.exists() and cam1_path.exists() and cam2_path.exists() and cam3_path.exists()):
        # if not (json_path.exists() and cam1_path.exists() and cam3_path.exists()):
            continue

        try:
            with open(json_path, "r") as f:
                robot_json = json.load(f)
            robot_data = robot_json["robot_data"]
            task_prompt = robot_json["task"]

            cap_cam1 = cv2.VideoCapture(str(cam1_path))
            cap_cam2 = cv2.VideoCapture(str(cam2_path))
            cap_cam3 = cv2.VideoCapture(str(cam3_path))

            for i, frame in enumerate(robot_data):
                try:
                    idx_cam1 = frame["frame_idx_cam1"] - 1
                    idx_cam2 = frame["frame_idx_cam2"] - 1
                    idx_cam3 = frame["frame_idx_cam3"] - 1

                    cap_cam1.set(cv2.CAP_PROP_POS_FRAMES, idx_cam1)
                    ret1, f1 = cap_cam1.read()
                    cap_cam2.set(cv2.CAP_PROP_POS_FRAMES, idx_cam2)
                    ret2, f2 = cap_cam2.read()
                    cap_cam3.set(cv2.CAP_PROP_POS_FRAMES, idx_cam3)
                    ret3, f3 = cap_cam3.read()

                    if not (ret1 and ret2 and ret3):
                        raise ValueError("Frame read error")

                    # wrist_img = Image.fromarray(f1).resize((256, 256))
                    wrist_img = Image.fromarray(cv2.cvtColor(f1, cv2.COLOR_BGR2RGB)).resize((256, 256)) # D435
                    main_img = Image.fromarray(cv2.cvtColor(f2, cv2.COLOR_BGR2RGB)).resize((256, 256)) # L515
                    right_wrist_img = Image.fromarray(cv2.cvtColor(f3, cv2.COLOR_BGR2RGB)).resize((256, 256)) # Temp

                    state = frame["joint_angles_deg"] + [0.0] + [frame["gripper_position"]]
                    if frame["action_joint_angles"] is not None:
                        end_state_flag = 1.0 if frame.get("end_state", False) else 0.0
                        actions = frame["action_joint_angles"] + [frame["action_gripper_position"], end_state_flag]
                    else:
                        actions = [0.0] * 8

                    dataset.add_frame({
                        "image": main_img,
                        "wrist_image": np.zeros_like(wrist_img),  # Assuming wrist image is not available
                        "right_wrist_image": right_wrist_img,
                        "state": np.array(state, dtype=np.float32),
                        "actions": np.array(actions, dtype=np.float32),
                        "task":task_prompt,
                    })

                    # dataset.add_frame({
                    #     "image": main_img,
                    #     "wrist_image": wrist_img,
                    #     "right_wrist_image": np.zeros_like(right_wrist_img),  # Assuming right wrist image is not available
                    #     "state": np.array(state, dtype=np.float32),
                    #     "actions": np.array(actions, dtype=np.float32),
                    #     "task":task_prompt,
                    # })

                except Exception as e:
                    print(f"Frame {i} skipped in {dataset_dir.name}: {e}")

            print(f"✅ Loaded {dataset_dir}")
            cap_cam1.release()
            cap_cam2.release()
            cap_cam3.release()

            dataset.save_episode()

        except Exception as e:
            print(f"Error in {dataset_dir}: {e}")
            
        try:
            with open(json_path, "r") as f:
                robot_json = json.load(f)
            robot_data = robot_json["robot_data"]
            task_prompt = robot_json["task"]

            cap_cam1 = cv2.VideoCapture(str(cam1_path))
            cap_cam2 = cv2.VideoCapture(str(cam2_path))
            cap_cam3 = cv2.VideoCapture(str(cam3_path))

            for i, frame in enumerate(robot_data):
                try:
                    idx_cam1 = frame["frame_idx_cam1"] - 1
                    idx_cam2 = frame["frame_idx_cam2"] - 1
                    idx_cam3 = frame["frame_idx_cam3"] - 1

                    cap_cam1.set(cv2.CAP_PROP_POS_FRAMES, idx_cam1)
                    ret1, f1 = cap_cam1.read()
                    cap_cam2.set(cv2.CAP_PROP_POS_FRAMES, idx_cam2)
                    ret2, f2 = cap_cam2.read()
                    cap_cam3.set(cv2.CAP_PROP_POS_FRAMES, idx_cam3)
                    ret3, f3 = cap_cam3.read()

                    if not (ret1 and ret2 and ret3):
                        raise ValueError("Frame read error")

                    # wrist_img = Image.fromarray(f1).resize((256, 256))
                    wrist_img = Image.fromarray(cv2.cvtColor(f1, cv2.COLOR_BGR2RGB)).resize((256, 256)) # D435
                    main_img = Image.fromarray(cv2.cvtColor(f2, cv2.COLOR_BGR2RGB)).resize((256, 256)) # L515
                    right_wrist_img = Image.fromarray(cv2.cvtColor(f3, cv2.COLOR_BGR2RGB)).resize((256, 256)) # Temp

                    state = frame["joint_angles_deg"] + [0.0] + [frame["gripper_position"]]
                    if frame["action_joint_angles"] is not None:
                        end_state_flag = 1.0 if frame.get("end_state", False) else 0.0
                        actions = frame["action_joint_angles"] + [frame["action_gripper_position"], end_state_flag]
                    else:
                        actions = [0.0] * 8

                    # dataset.add_frame({
                    #     "image": main_img,
                    #     "wrist_image": np.zeros_like(wrist_img),  # Assuming wrist image is not available
                    #     "right_wrist_image": right_wrist_img,
                    #     "state": np.array(state, dtype=np.float32),
                    #     "actions": np.array(actions, dtype=np.float32),
                    #     "task":task_prompt,
                    # })

                    dataset.add_frame({
                        "image": main_img,
                        "wrist_image": wrist_img,
                        "right_wrist_image": np.zeros_like(right_wrist_img),  # Assuming right wrist image is not available
                        "state": np.array(state, dtype=np.float32),
                        "actions": np.array(actions, dtype=np.float32),
                        "task":task_prompt,
                    })

                except Exception as e:
                    print(f"Frame {i} skipped in {dataset_dir.name}: {e}")

            print(f"✅ Loaded {dataset_dir}")
            cap_cam1.release()
            cap_cam2.release()
            cap_cam3.release()

            dataset.save_episode()

        except Exception as e:
            print(f"Error in {dataset_dir}: {e}")
#TODO
#dataset.consolidate()
print(" Dataset saved and consolidated.")