import numpy as np
import time

nums = np.random.randint(1000000, size=(1000000))

# print(nums)

target = np.random.choice(nums)

nums = sorted(nums)

def binary_search(lower, upper):
  midpoint = np.floor((lower + upper) / 2)
  if target == nums[int(midpoint)]:
    return midpoint
  elif nums[int(midpoint)] > target:
    return binary_search(lower, midpoint - 1)
  else:
    return binary_search(midpoint + 1, upper)

start = time.time()
index = binary_search(0, 1000000)
print(time.time() - start)

start = time.time()
for i in range(1000000):
  if nums[i] == target:
    break
print(time.time() - start)

