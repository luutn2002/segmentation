#Note: This batch is make for execution with python in anaconda environment, please run this after activate your env.
#Ex: conda activate ${YOUR_ENVIRONMENT_NAME}
#Run this bash file inside "project" file

#set -e

CURRENT_DIR=$(pwd)
EXPORT_FILE="${CURRENT_DIR}/export"
MODEL_EXPORT="${EXPORT_FILE}/trained_model"
MODEL_VIS="${EXPORT_FILE}/model_visualization"
WORK_FILE="${CURRENT_DIR}/train_new_model.py"
CHECKPOINT_DIR="${EXPORT_FILE}/training_checkpoint"

mkdir -p "${EXPORT_FILE}"
mkdir -p "${MODEL_EXPORT}"
mkdir -p "${MODEL_VIS}"
mkdir -p "${CHECKPOINT_DIR}"

python "${WORK_FILE}"

mv model.png "${MODEL_VIS}"
cd "${EXPORT_FILE}"
mv saved_model.pb "${MODEL_EXPORT}"
