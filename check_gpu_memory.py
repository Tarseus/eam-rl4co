#!/usr/bin/env python
"""简单的GPU显存诊断脚本"""
import torch
import sys

print("=" * 60)
print("GPU显存诊断")
print("=" * 60)

if not torch.cuda.is_available():
    print("❌ CUDA不可用")
    sys.exit(1)

print(f"CUDA设备数: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    total_mem = props.total_memory / 1024**3
    print(f"\n设备 {i}: {props.name}")
    print(f"  总显存: {total_mem:.2f} GB")

print("\n" + "=" * 60)
print("当前显存使用:")
print("=" * 60)

torch.cuda.empty_cache()

for i in range(torch.cuda.device_count()):
    allocated = torch.cuda.memory_allocated(i) / 1024**3
    reserved = torch.cuda.memory_reserved(i) / 1024**3
    print(f"设备 {i}:")
    print(f"  已分配: {allocated:.2f} GB")
    print(f"  已缓存: {reserved:.2f} GB")

print("\n" + "=" * 60)
print("测试小批量前向传播:")
print("=" * 60)

try:
    from rl4co.models import MatNet
    from rl4co.envs import FFSPEnv

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建小模型测试
    env = FFSPEnv(generator_params={
        "num_stage": 3,
        "num_machine": 4,
        "num_job": 100,
    })

    print("\n创建模型 (num_encoder_layers=5)...")
    model = MatNet(
        env=env,
        policy_params={
            "embed_dim": 256,
            "num_encoder_layers": 5,
            "num_heads": 16,
        }
    ).to(device)

    allocated = torch.cuda.memory_allocated(0) / 1024**3
    print(f"模型显存占用: {allocated:.2f} GB")

    print("\n测试前向传播 (batch_size=8)...")
    td = env.reset(batch_size=[8])
    with torch.no_grad():
        out = model(td, env, phase="val", decode_type="greedy")

    allocated = torch.cuda.memory_allocated(0) / 1024**3
    print(f"前向后显存占用: {allocated:.2f} GB")
    print(f"奖励: {out['reward'].mean().item():.2f}")

    print("\n✅ 小批量测试成功!")

    # 测试不同batch_size的显存占用
    print("\n" + "=" * 60)
    print("测试不同batch_size的显存占用:")
    print("=" * 60)

    for bs in [4, 8, 16, 24, 32]:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            td = env.reset(batch_size=[bs])
            # 带梯度的前向传播（模拟训练）
            out = model(td, env, phase="train", decode_type="sampling", num_starts=24)

            peak_mem = torch.cuda.max_memory_allocated(0) / 1024**3
            print(f"batch_size={bs:2d}: 峰值显存 = {peak_mem:.2f} GB")

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"batch_size={bs:2d}: ❌ OOM")
                torch.cuda.empty_cache()
            else:
                raise
        finally:
            torch.cuda.empty_cache()

except Exception as e:
    print(f"\n❌ 错误: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
