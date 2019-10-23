import torch

def Poission_generate(float_tensor,timesteps):
    raise NotImplementedError
    return SpikeTensor()

class SpikeTensor():
    def __init__(self,data,timesteps,scale_factor=None,is_spike=True):
        """
        data shape: [batch*t,...]
        """
        self.data=data
        self.timesteps=timesteps
        self.is_spike=is_spike
        self.scale_factor=scale_factor


    def firing_ratio(self):
        chw = self.data.size()[1:]
        firing_ratio = torch.mean(self.data.view(-1, self.timesteps, *chw), 1)
        return firing_ratio

    def timestep_dim_tensor(self):
        chw = self.data.size()[1:]
        return self.data.view(-1,self.timesteps,*chw)

    def size(self,*args):
        return self.data.size(*args)

    def view(self,*args):
        return SpikeTensor(self.data.view(*args),self.timesteps,self.scale_factor,self.is_spike)

    def to_float(self):
        assert self.is_spike and self.scale_factor is not None
        chw=self.data.size()[1:]
        float_tensor=self.firing_ratio()
        float_tensor*=self.scale_factor.view(1,-1,*([1]*(len(chw)-1)))
        return float_tensor

    def input_replica(self):
        self.data=torch.repeat_interleave(self.data.unsqueeze(1),self.timesteps,1).view(-1,*self.data.size()[1:])