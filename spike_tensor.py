import torch

def Poission_generate(float_tensor,timesteps):
    raise NotImplementedError
    return SpikeTensor()

class SpikeTensor():
    def __init__(self,data,timesteps,scale_factor):
        """
        data shape: [t*batch,...]
        """
        self.data=data
        self.timesteps=timesteps
        self.b = self.data.size(0) // timesteps
        self.chw = self.data.size()[1:]
        if isinstance(scale_factor, torch.Tensor):
            self.scale_factor=scale_factor.view( 1,-1,*([1]*(len(self.chw)-1)) )
        else:
            self.scale_factor=scale_factor


    def firing_ratio(self):
        chw = self.data.size()[1:]
        firing_ratio = torch.mean(self.data.view(self.timesteps, -1, *chw), 0)
        return firing_ratio

    def timestep_dim_tensor(self):
        return self.data.view(self.timesteps,-1,*self.chw)

    def size(self,*args):
        return self.data.size(*args)

    def view(self,*args):
        return SpikeTensor(self.data.view(*args),self.timesteps,self.scale_factor)

    def to_float(self):
        assert self.scale_factor is not None
        float_tensor=self.firing_ratio()
        scaled_float_tensor=float_tensor*self.scale_factor
        return scaled_float_tensor
