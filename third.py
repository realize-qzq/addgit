# import sys
# text=sys.stdin.readline().strip().split()
# n,k=int(text[0]),int(text[1])
# text=sys.stdin.readline().strip().split()
# nums=[int(x) for x in text]
# nums.sort()
# ans=0
# i=0
# ind=0
# res=[]
# while ind<k:
#     ind+=1
#     if i<n:
#         y=nums[i]
#     else:
#         res.append(0)
#         # print(0)
#         continue
#     x=y-ans
#     if x==0:
#         i+=1
#     else:
#         res.append(x)
#         ans+=y
#         i+=1
# if len(res)==k:
#     for x in res:
#         print(x)
# else:
#     res=res+[0]*(k-len(res))
#     for x in res:
#         print(x)

import sys
text = sys.stdin.readline().strip().split()
n, k = int(text[0]), int(text[1])
text = sys.stdin.readline().strip().split()
nums = sorted([int(x) for x in text])
ans = 0
xans = 0
for i in range(k):
    y = nums[i]
    if i == 0:
        ans = 0
    else:
        ans = nums[i - 1]
    print(y - ans)
