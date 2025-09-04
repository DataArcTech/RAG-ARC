import numpy as np
import heapq
import time

# 生成大数组进行性能测试
np.random.seed(42)
scores = np.random.rand(1_000_000_000)  # 100万元素，修正数字表示
k = 100

print(f"测试数组大小: {len(scores):,} 元素")
print(f"查找前 {k} 个最大值的索引")
print("-" * 50)

# 方法1: np.argpartition + 局部排序
print("方法1: numpy.argpartition + 局部排序")
start = time.time()
top_k_idx_np = np.argpartition(-scores, k-1)[:k]
top_k_idx_np_sorted = top_k_idx_np[np.argsort(-scores[top_k_idx_np])]
time_np = time.time() - start
print(f"耗时: {time_np:.4f} 秒")

# 方法2: heapq.nlargest
print("方法2: heapq.nlargest")
start = time.time()
top_k_idx_heap = [i for i, _ in heapq.nlargest(k, enumerate(scores), key=lambda x: x[1])]
time_heap = time.time() - start
print(f"耗时: {time_heap:.4f} 秒")

# 方法3: np.argsort 全排序
print("方法3: numpy.argsort 全排序")
start = time.time()
top_k_idx_argsort = np.argsort(scores)[::-1][:k]
time_argsort = time.time() - start
print(f"耗时: {time_argsort:.4f} 秒")

print("-" * 50)
print(f"性能对比:")
print(f"numpy argpartition方法: {time_np:.4f} 秒")
print(f"heapq方法: {time_heap:.4f} 秒")
print(f"numpy argsort方法: {time_argsort:.4f} 秒")

# 找出最快的方法
times = {'argpartition': time_np, 'heapq': time_heap, 'argsort': time_argsort}
fastest = min(times, key=times.get)
print(f"最快方法: {fastest} ({times[fastest]:.4f} 秒)")

# 验证结果一致性
print(f"\n结果验证:")
print(f"argpartition结果前5个索引: {top_k_idx_np_sorted[:5]}")
print(f"heapq结果前5个索引: {top_k_idx_heap[:5]}")
print(f"argsort结果前5个索引: {top_k_idx_argsort[:5]}")
print(f"argpartition与heapq结果一致: {set(top_k_idx_np_sorted) == set(top_k_idx_heap)}")
print(f"argpartition与argsort结果一致: {set(top_k_idx_np_sorted) == set(top_k_idx_argsort)}")
print(f"heapq与argsort结果一致: {set(top_k_idx_heap) == set(top_k_idx_argsort)}")