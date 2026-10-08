export CUDA_VISIBLE_DEVICES=5,7
# GPU 数量
export MLP_WORKER_GPU=2

# 主节点地址和端口（单机用本机IP和任意空闲端口）
export MLP_WORKER_0_HOST=127.0.0.1
export MLP_WORKER_0_PORT=2950$((RANDOM % 10))

# 当前节点rank（单机就是0）
export MLP_ROLE_INDEX=0

# 节点总数（单机就是1）
export MLP_WORKER_NUM=1

export NCCL_P2P_DISABLE=1

config=$1
mode=${2:-train}
export find_unused_parameters=True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$PYTHONPATH:/data1/qjyuan/OmniNWM
export PYTHONPATH=/data1/qjyuan/OmniNWM/omninwm/models/ops/occ_pooling:$PYTHONPATH

if [ "$mode" = "inference" ]; then
    script="tools/inference_reg.py"
else
    script="tools/train_reg.py"
fi

torchrun --nproc-per-node $MLP_WORKER_GPU \
  --master-addr $MLP_WORKER_0_HOST \
  --master-port $MLP_WORKER_0_PORT \
  --node-rank $MLP_ROLE_INDEX \
  --nnodes $MLP_WORKER_NUM \
  $script \
  $config
