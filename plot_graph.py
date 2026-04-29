# # # # # import matplotlib.pyplot as plt
# # # # # # import numpy as np

# # # # # # x = np.linspace(0, 2 * np.pi, 100)

# # # # # # x = [256,512,1024,2048,4096,8192,12288]
# # # # # x = [1, 2, 4, 8, 16, 32, 64]

# # # # # y1 = [0.222, 0.253, 0.328, 0.522, 0.803, 1.434, 2.675]
# # # # # y2 = [0.188, 0.216, 0.262, 0.346, 0.373, 0.487, 0.550]

# # # # # # y1 = [4.797, 9.035, 18.542, 37.744, 77.955, 161.335, 321.155]
# # # # # # y2 = [7.948, 14.417, 28.325, 58.282, 122.744, 249.714, 499.385]

# # # # # plt.plot(x, y1, label='Self Attention', color='blue')
# # # # # plt.plot(x, y2, label='MLP', color='red')

# # # # # plt.xlabel('Batch size')
# # # # # plt.ylabel('Time (ms)')

# # # # # plt.title('Q5 Decode')
# # # # # plt.legend()
# # # # # plt.savefig("part2/plots/q5.png")
# # # # # plt.show()


# # # # import matplotlib.pyplot as plt

# # # # # --- Dataset ---
# # # # # Sweep 1: Fixed Batch = 16, Varying Sequence Length
# # # # seq_lens = [256, 512, 1024, 2048]
# # # # ai_seq = [42.9, 51.5, 57.2, 60.6]
# # # # thpt_seq = [0.13, 0.53, 2.01, 6.43]

# # # # # Sweep 2: Fixed Sequence Length = 256, Varying Batch Size
# # # # batches = [1, 4, 8, 16, 32]
# # # # ai_batch = [42.9, 42.9, 42.9, 42.9, 42.9]
# # # # thpt_batch = [0.008, 0.040, 0.066, 0.133, 0.264]

# # # # # --- Plotting ---
# # # # fig, ax = plt.subplots(figsize=(10, 6))

# # # # # Plot 1: Varying Seq Len (The Curve)
# # # # ax.plot(ai_seq, thpt_seq, marker='o', linestyle='-', color='#1f77b4', 
# # # #         linewidth=2, markersize=8, label='Varying Seq Len (Fixed B=16)')

# # # # for i, txt in enumerate(seq_lens):
# # # #     ax.annotate(f"N={txt}", (ai_seq[i], thpt_seq[i]), textcoords="offset points", 
# # # #                 xytext=(0, 10), ha='center', fontsize=9)

# # # # # Plot 2: Varying Batch (The Vertical Line)
# # # # ax.plot(ai_batch, thpt_batch, marker='s', linestyle='--', color='#d62728', 
# # # #         linewidth=2, markersize=8, label='Varying Batch (Fixed N=256)')

# # # # for i, txt in enumerate(batches):
# # # #     # Staggering labels for the vertical line to prevent overlapping text
# # # #     offset_y = -15 if i % 2 == 0 else 10
# # # #     ax.annotate(f"B={txt}", (ai_batch[i], thpt_batch[i]), textcoords="offset points", 
# # # #                 xytext=(-25, offset_y), ha='center', fontsize=9)

# # # # # --- Formatting ---
# # # # ax.set_title('Attention Prefill: Throughput vs Arithmetic Intensity', 
# # # #              fontsize=14, fontweight='bold', pad=15)
# # # # ax.set_xlabel('Arithmetic Intensity (FLOPs/Byte)', fontsize=12, fontweight='bold')
# # # # ax.set_ylabel('Throughput (TFLOPs/s)', fontsize=12, fontweight='bold')

# # # # # Set limits to give the annotations room to breathe
# # # # ax.set_xlim(40, 65)
# # # # ax.set_ylim(-0.5, 7.5)

# # # # ax.grid(True, linestyle='--', alpha=0.6)
# # # # ax.legend(loc='upper left', fontsize=11)

# # # # plt.tight_layout()
# # # # plt.savefig("part2/plots/q7.png")
# # # # plt.show()


# # # import matplotlib.pyplot as plt

# # # # --- Data ---
# # # batches = [1, 4, 8, 16, 32]
# # # speedup_batch = [4014.8, 1019.8, 680.9, 368.8, 191.3]

# # # seq_lens = [256, 512, 1024, 2048] # Stopped before OOM
# # # speedup_seq = [367.2, 117.4, 34.5, 11.6]

# # # # --- Plotting ---
# # # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# # # # Subplot 1: Speedup vs Batch Size
# # # color1 = '#ff7f0e'
# # # ax1.plot(batches, speedup_batch, marker='o', color=color1, linewidth=2.5, markersize=8)
# # # ax1.set_title('Speedup vs Batch Size\n(Fixed Seq_Len = 256)', fontsize=13, fontweight='bold')
# # # ax1.set_xlabel('Batch Size (B)', fontsize=12)
# # # ax1.set_ylabel('Speedup Factor (Eager / Fused)', fontsize=12)
# # # ax1.set_xticks(batches)
# # # ax1.grid(True, linestyle='--', alpha=0.6)
# # # ax1.set_yscale('log')

# # # for i, txt in enumerate(speedup_batch):
# # #     ax1.annotate(f"{txt:.1f}x", (batches[i], speedup_batch[i]), textcoords="offset points", 
# # #                  xytext=(0, 10), ha='center', fontsize=10)

# # # # Subplot 2: Speedup vs Sequence Length
# # # color2 = '#2ca02c'
# # # ax2.plot(seq_lens, speedup_seq, marker='s', color=color2, linewidth=2.5, markersize=8)
# # # ax2.set_title('Speedup vs Sequence Length\n(Fixed Batch = 16)', fontsize=13, fontweight='bold')
# # # ax2.set_xlabel('Sequence Length (N)', fontsize=12)
# # # ax2.set_xticks(seq_lens)
# # # ax2.grid(True, linestyle='--', alpha=0.6)
# # # ax2.set_yscale('log')

# # # for i, txt in enumerate(speedup_seq):
# # #     ax2.annotate(f"{txt:.1f}x", (seq_lens[i], speedup_seq[i]), textcoords="offset points", 
# # #                  xytext=(0, 10), ha='center', fontsize=10)

# # # plt.suptitle('Fused vs Eager Attention Speedup', fontsize=16, fontweight='bold', y=1.05)
# # # plt.tight_layout()
# # # plt.savefig("part2/plots/q8.png")
# # # plt.show()


# # import matplotlib.pyplot as plt
# # import numpy as np

# # # --- Dataset ---
# # seq_lens = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
# # time_ms = [73.525, 71.857, 71.873, 68.303, 70.723, 72.749, 73.172, 74.421]

# # # Calculate AI and Throughput mathematically
# # d = 128
# # compute_gflops = [(16 * 32 * (4*d + 3) * n) / 1e9 for n in seq_lens]
# # ai = [(515 * n) / (520 * n + 512) for n in seq_lens]
# # throughput_tflops = [c / (t / 1000) / 1000 for c, t in zip(compute_gflops, time_ms)]

# # # --- Plotting ---
# # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# # # Subplot 1: Throughput vs Arithmetic Intensity
# # color1 = '#9467bd'
# # ax1.plot(ai, throughput_tflops, marker='o', color=color1, linewidth=2, markersize=8)
# # ax1.set_title('Decode: Throughput vs Arithmetic Intensity', fontsize=13, fontweight='bold')
# # ax1.set_xlabel('Arithmetic Intensity (FLOPs/Byte)', fontsize=12)
# # ax1.set_ylabel('Throughput (TFLOPs/s)', fontsize=12)
# # ax1.grid(True, linestyle='--', alpha=0.6)

# # # Annotate N values (staggered to avoid overlapping on the vertical line)
# # for i, txt in enumerate(seq_lens):
# #     offset_x = 10 if i % 2 == 0 else -35
# #     ax1.annotate(f"N={txt}", (ai[i], throughput_tflops[i]), textcoords="offset points", 
# #                  xytext=(offset_x, 0), va='center', fontsize=9)

# # # Subplot 2: Throughput & Latency vs Sequence Length
# # color2 = '#1f77b4'
# # color3 = '#d62728'

# # ax2.set_title('Decode: Scaling Behavior vs Sequence Length', fontsize=13, fontweight='bold')
# # ax2.set_xlabel('Sequence Length (N)', fontsize=12)
# # ax2.set_ylabel('Throughput (TFLOPs/s)', color=color2, fontsize=12)
# # line1 = ax2.plot(seq_lens, throughput_tflops, marker='^', color=color2, linewidth=2, label='Throughput')
# # ax2.tick_params(axis='y', labelcolor=color2)
# # ax2.set_xscale('log', base=2) # Log scale for sequence lengths

# # ax3 = ax2.twinx()
# # ax3.set_ylabel('Time (ms)', color=color3, fontsize=12)
# # line2 = ax3.plot(seq_lens, time_ms, marker='s', color=color3, linestyle='--', linewidth=2, label='Latency')
# # ax3.tick_params(axis='y', labelcolor=color3)
# # ax3.set_ylim(0, 100) # Anchor latency to 0 to show how flat it actually is

# # # Combine legends
# # lines = line1 + line2
# # labels = [l.get_label() for l in lines]
# # ax2.legend(lines, labels, loc='upper left')
# # ax2.grid(True, linestyle='--', alpha=0.6)

# # plt.tight_layout()
# # plt.savefig("part2/plots/q10.png")
# # plt.show()


# import matplotlib.pyplot as plt

# # --- Dataset ---
# batches = [16, 32, 64, 128, 256, 512, 1024]
# time_ms = [70.267, 71.926, 69.000, 69.600, 74.852, 77.435, 96.347]
# compute_gflops = [0.540, 1.080, 2.160, 4.320, 8.640, 17.281, 34.561]

# # Constant AI for Decode (N=2048)
# ai = [0.99] * len(batches)
# throughput_tflops = [0.0077, 0.0150, 0.0313, 0.0621, 0.1154, 0.2232, 0.3587]

# # --- Plotting ---
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# # Subplot 1: Throughput vs Arithmetic Intensity (The Vertical Line)
# color1 = '#9467bd'
# ax1.plot(ai, throughput_tflops, marker='o', color=color1, linewidth=2, markersize=8)
# ax1.set_title('Decode: Throughput vs Arithmetic Intensity\n(Varying Batch Size)', fontsize=13, fontweight='bold')
# ax1.set_xlabel('Arithmetic Intensity (FLOPs/Byte)', fontsize=12)
# ax1.set_ylabel('Throughput (TFLOPs/s)', fontsize=12)
# ax1.grid(True, linestyle='--', alpha=0.6)

# # Force x-axis limits to center the vertical line
# ax1.set_xlim(0.5, 1.5)

# # Annotate Batch values
# for i, txt in enumerate(batches):
#     offset_x = 10 if i % 2 == 0 else -40
#     ax1.annotate(f"B={txt}", (ai[i], throughput_tflops[i]), textcoords="offset points", 
#                  xytext=(offset_x, 0), va='center', fontsize=9)

# # Subplot 2: Throughput & Latency vs Batch Size
# color2 = '#1f77b4'
# color3 = '#d62728'

# ax2.set_title('Decode: Scaling Behavior vs Batch Size\n(Fixed Seq_Len = 2048)', fontsize=13, fontweight='bold')
# ax2.set_xlabel('Batch Size (B)', fontsize=12)
# ax2.set_ylabel('Throughput (TFLOPs/s)', color=color2, fontsize=12)
# line1 = ax2.plot(batches, throughput_tflops, marker='^', color=color2, linewidth=2, label='Throughput')
# ax2.tick_params(axis='y', labelcolor=color2)
# ax2.set_xscale('log', base=2) # Log scale nicely spaces out powers of 2
# ax2.set_xticks(batches)
# ax2.set_xticklabels(batches)

# ax3 = ax2.twinx()
# ax3.set_ylabel('Time (ms)', color=color3, fontsize=12)
# line2 = ax3.plot(batches, time_ms, marker='s', color=color3, linestyle='--', linewidth=2, label='Latency')
# ax3.tick_params(axis='y', labelcolor=color3)
# ax3.set_ylim(0, 110) # Anchor to 0 to visualize the "flatness" overhead

# # Combine legends
# lines = line1 + line2
# labels = [l.get_label() for l in lines]
# ax2.legend(lines, labels, loc='upper left')
# ax2.grid(True, linestyle='--', alpha=0.6)

# plt.tight_layout()
# plt.savefig("part2/plots/q11.png")

# plt.show()

import matplotlib.pyplot as plt

# --- Dataset ---
seq_lens = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
speedup_seq = [597.8, 308.4, 158.3, 75.9, 38.0, 19.3, 9.8, 4.9]

batches = [16, 32, 64, 128, 256, 512, 1024]
speedup_batch = [79.0, 38.7, 18.3, 9.2, 4.9, 2.5, 1.5]

# --- Plotting ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Subplot 1: Speedup vs Sequence Length
color1 = '#2ca02c'
ax1.plot(seq_lens, speedup_seq, marker='s', color=color1, linewidth=2.5, markersize=8)
ax1.set_title('Speedup vs Sequence Length\n(Fixed Batch = 16)', fontsize=13, fontweight='bold')
ax1.set_xlabel('Sequence Length (N)', fontsize=12)
ax1.set_ylabel('Speedup Factor (Eager / Fused)', fontsize=12)
ax1.set_xscale('log', base=2)
ax1.set_yscale('log')
ax1.set_xticks(seq_lens)
ax1.set_xticklabels(seq_lens, rotation=45)
ax1.grid(True, linestyle='--', alpha=0.6)

for i, txt in enumerate(speedup_seq):
    ax1.annotate(f"{txt:.1f}x", (seq_lens[i], speedup_seq[i]), textcoords="offset points", 
                 xytext=(0, 10), ha='center', fontsize=10)

# Subplot 2: Speedup vs Batch Size
color2 = '#ff7f0e'
ax2.plot(batches, speedup_batch, marker='o', color=color2, linewidth=2.5, markersize=8)
ax2.set_title('Speedup vs Batch Size\n(Fixed Seq_Len = 2048)', fontsize=13, fontweight='bold')
ax2.set_xlabel('Batch Size (B)', fontsize=12)
ax2.set_xscale('log', base=2)
ax2.set_yscale('log')
ax2.set_xticks(batches)
ax2.set_xticklabels(batches)
ax2.grid(True, linestyle='--', alpha=0.6)

for i, txt in enumerate(speedup_batch):
    ax2.annotate(f"{txt:.1f}x", (batches[i], speedup_batch[i]), textcoords="offset points", 
                 xytext=(0, 10), ha='center', fontsize=10)

plt.suptitle('Decode Stage: The 1/x Speedup Decay', fontsize=16, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig("part2/plots/q12.png")
plt.show()