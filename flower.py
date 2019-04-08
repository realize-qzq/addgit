class flower:
    def __init__(self,zero_nums,one_nums):
        self.zero_nums=zero_nums
        self.one_nums=one_nums
        self.permuation=[]
    def dfs(self,prev,ind,permuation):
        if ind==self.one_nums:
            self.permuation.append(''.join(permuation))
            return
        for i in range(ind,self.zero_nums+1):
            tmp=permuation+["0"]*(ind-prev)+["1"]
            self.dfs(ind,i+1,tmp)
    def gouzao(self):
        self.dfs(0,0,[])
        ans=self.permuation
        print("total ans nums: %s"%len(ans))
        for x in ans:
            print(x)
f=flower(5,3)
f.gouzao()