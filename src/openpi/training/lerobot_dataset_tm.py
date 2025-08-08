import json
import cv2
from pathlib import Path
from PIL import Image
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, HF_LEROBOT_HOME
import shutil

# 데이터셋 경로 설정
base_root = Path("/media/choiyj/mnt_store/Thermal")
REPO_NAME = "physical-intelligence/libero_tm_test_50"
output_path = HF_LEROBOT_HOME / REPO_NAME

if output_path.exists():
    shutil.rmtree(output_path)

dataset = LeRobotDataset.create(
    repo_id=REPO_NAME,
    robot_type="kinova_gen3",
    fps=15,
    features={
        "image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
        "wrist_image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
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

    for dataset_dir in task_dir.glob("dataset_*"):
        json_path = dataset_dir / "robot_data.json"

        cam1_path = dataset_dir / "camera1.avi"
        cam2_path = dataset_dir / "camera2.avi"
        
        if not (json_path.exists() and cam1_path.exists() and cam2_path.exists()):
            continue

        try:
            with open(json_path, "r") as f:
                robot_json = json.load(f)
            robot_data = robot_json["robot_data"]
            # task_prompt = "Turn off the hair straightener if it is hot."
            # task_num = robot_json["task"]
            
            # task_map = {
            #     1: "Turn off the power strip switches for battery chargers with overheated batteries",
            #     2: "Turn off the main power switch of the power strip",
            #     3: "turn off hair straightener",
            #     4: "pick up unplugged wire from floor",
            #     5: "place unplugged wire to the right side of power strip",
            # }
            task_prompt = robot_json["task"]


            cap_cam1 = cv2.VideoCapture(str(cam1_path))
            cap_cam2 = cv2.VideoCapture(str(cam2_path))

            for i, frame in enumerate(robot_data):
                try:
                    #dddd
                    end_state = frame["end_state"]

                    idx_cam1 = frame["frame_idx_cam1"] - 1
                    idx_cam2 = frame["frame_idx_cam2"] - 1

                    cap_cam1.set(cv2.CAP_PROP_POS_FRAMES, idx_cam1)
                    ret1, f1 = cap_cam1.read()
                    cap_cam2.set(cv2.CAP_PROP_POS_FRAMES, idx_cam2)
                    ret2, f2 = cap_cam2.read()

                    if not (ret1 and ret2):
                        raise ValueError("Frame read error")

                    main_img = Image.fromarray(cv2.cvtColor(f2, cv2.COLOR_BGR2RGB)).resize((256, 256))
                    wrist_img= Image.fromarray(cv2.cvtColor(f1, cv2.COLOR_BGR2RGB)).resize((256, 256))

                    state = frame["joint_angles_deg"] + [0.0] + [frame["gripper_position"]]
                    if frame["action_joint_angles"] is not None:
                        end_state_flag = 1.0 if frame.get("end_state", False) else 0.0
                        actions = frame["action_joint_angles"] + [frame["action_gripper_position"], end_state_flag]
                    else:
                        actions = [0.0] * 8

                    dataset.add_frame({
                        "image": main_img,
                        "wrist_image": wrist_img,
                        "state": np.array(state, dtype=np.float32),
                        "actions": np.array(actions, dtype=np.float32),
                        "task":task_prompt,
                    })

                except Exception as e:
                    print(f"Frame {i} skipped in {dataset_dir.name}: {e}")

            print(f"âœ… Loaded {dataset_dir}")
            cap_cam1.release()
            cap_cam2.release()

            dataset.save_episode()

        except Exception as e:
            print(f"Error in {dataset_dir}: {e}")
#TODO
#dataset.consolidate()
print(" Dataset saved and consolidated.")