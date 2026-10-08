export PYTHONPATH=$PYTHONPATH:/data1/qjyuan/OmniNWM/src
export PYTHONPATH=/data1/qjyuan/OmniNWM/omninwm/models/ops/occ_pooling:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_VISIBLE_DEVICES=3,4
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=1800

torchrun --nproc-per-node 2 --master_port 29505 tools/inference.py configs/inference/infer.py
# torchrun --nproc-per-node 2 --master_port 29505 tools/inference_reg.py configs/inference/infer_reg.py