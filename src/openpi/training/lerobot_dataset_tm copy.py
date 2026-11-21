import json
import cv2
from pathlib import Path
from PIL import Image
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, HF_LEROBOT_HOME
import shutil

# 데이터셋 경로 설정
base_root = Path("/media/choiyj/mnt_store/power_strip_50")
REPO_NAME = "physical-intelligence/libero_se_tu_50" # libero_(tm or rgb)_(li or se or ...)_(pl or pi or tu or all)_(70 or 50 or 30)
output_path = HF_LEROBOT_HOME / REPO_NAME
target_task = "Turn_off_the_hot_hair_straightener"

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

rgb_dir = base_root / "OnlyRGB"
tm_dir = base_root / "Thermal"
for task_dir in rgb_dir.iterdir():
    if not task_dir.is_dir():
        continue
    if task_dir.name != target_task:
        continue
    tm_task_dir = tm_dir / task_dir.name
    if not tm_task_dir.exists():
        continue
    print(f"Processing task: {task_dir.name}")
    for dataset_dir in task_dir.glob("dataset_*"):
        json_path1 = dataset_dir / "robot_data.json"
        json_path2 = tm_task_dir / dataset_dir.name / "robot_data.json"

        cam1_path = dataset_dir / "camera1.avi"
        cam2_path = dataset_dir / "camera2.avi"
        cam3_path = tm_task_dir / dataset_dir.name / "camera2.avi"

        if not (json_path1.exists() and cam1_path.exists() and cam2_path.exists() and cam3_path.exists()):
            continue

        try:
            with open(json_path1, "r") as f:
                robot_json_rgb = json.load(f)
            with open(json_path2, "r") as f:
                robot_json_tm = json.load(f)
            robot_data_rgb = robot_json_rgb["robot_data"]
            robot_data_tm = robot_json_tm["robot_data"]


            task_prompt = robot_json_rgb["task"]


            cap_cam1 = cv2.VideoCapture(str(cam1_path))
            cap_cam2 = cv2.VideoCapture(str(cam2_path))
            cap_cam3 = cv2.VideoCapture(str(cam3_path))

            for i in range(len(robot_data_rgb)):
                try:
                    frame_rgb = robot_data_rgb[i]
                    frame_tm = robot_data_tm[i]

                    #dddd
                    end_state = frame_rgb["end_state"]

                    idx_cam1 = frame_rgb["frame_idx_cam1"] - 1
                    idx_cam2 = frame_rgb["frame_idx_cam2"] - 1
                    idx_cam3 = frame_tm["frame_idx_cam2"] - 1

                    cap_cam1.set(cv2.CAP_PROP_POS_FRAMES, idx_cam1)
                    ret1, f1 = cap_cam1.read()
                    cap_cam2.set(cv2.CAP_PROP_POS_FRAMES, idx_cam2)
                    ret2, f2 = cap_cam2.read()
                    cap_cam3.set(cv2.CAP_PROP_POS_FRAMES, idx_cam3)
                    ret3, f3 = cap_cam3.read()

                    if not (ret1 and ret2 and ret3):
                        raise ValueError("Frame read error")

                    main_img = Image.fromarray(cv2.cvtColor(f2, cv2.COLOR_BGR2RGB)).resize((256, 256))
                    wrist_img= Image.fromarray(cv2.cvtColor(f1, cv2.COLOR_BGR2RGB)).resize((256, 256))
                    right_wrist_img= Image.fromarray(cv2.cvtColor(f3, cv2.COLOR_BGR2RGB)).resize((256, 256))

                    state = frame_rgb["joint_angles_deg"] + [0.0] + [frame_rgb["gripper_position"]]
                    if frame_rgb["action_joint_angles"] is not None:
                        end_state_flag = 1.0 if frame_rgb.get("end_state", False) else 0.0
                        actions = frame_rgb["action_joint_angles"] + [frame_rgb["action_gripper_position"], end_state_flag]
                    else:
                        actions = [0.0] * 8

                    dataset.add_frame({
                        "image": main_img,
                        "wrist_image": wrist_img,
                        "right_wrist_image": right_wrist_img,
                        "state": np.array(state, dtype=np.float32),
                        "actions": np.array(actions, dtype=np.float32),
                        "task":task_prompt,
                    })

                except Exception as e:
                    print(f"Frame {i} skipped in {dataset_dir.name}: {e}")

            print(f"âœ… Loaded {dataset_dir}")
            cap_cam1.release()
            cap_cam2.release()
            cap_cam3.release()

            dataset.save_episode()

        except Exception as e:
            print(f"Error in {dataset_dir}: {e}")
#TODO
#dataset.consolidate()
print(" Dataset saved and consolidated.")