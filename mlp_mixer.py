import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
import torchsummary
from monarch_lconv_standalone import MonarchConv
from einops import rearrange

class MLPMixer(nn.Module):
    def __init__(self,in_channels=3,img_size=32, patch_size=4, hidden_size=512, hidden_s=256, hidden_c=2048, num_layers=8, num_classes=10, drop_p=0., 
    off_act=False, is_cls_token=False,use_monarch=False,**kwargs):
        super(MLPMixer, self).__init__()
        num_patches = img_size // patch_size * img_size // patch_size
        # (b, c, h, w) -> (b, d, h//p, w//p) -> (b, h//p*w//p, d)
        self.is_cls_token = is_cls_token

        self.patch_emb = nn.Sequential(
            nn.Conv2d(in_channels, hidden_size ,kernel_size=patch_size, stride=patch_size),
            Rearrange('b d h w -> b (h w) d')
        )

        if self.is_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
            num_patches += 1


        self.mixer_layers = nn.Sequential(
            *[
                MixerLayer(num_patches, hidden_size, hidden_s, hidden_c, drop_p, off_act,use_monarch=use_monarch,**kwargs) 
            for _ in range(num_layers)
            ]
        )
        self.ln = nn.LayerNorm(hidden_size)

        self.clf = nn.Linear(hidden_size, num_classes)


    def forward(self, x):
        out = self.patch_emb(x)
        if self.is_cls_token:
            out = torch.cat([self.cls_token.repeat(out.size(0),1,1), out], dim=1)
        out = self.mixer_layers(out)
        out = self.ln(out)
        out = out[:, 0] if self.is_cls_token else out.mean(dim=1)
        out = self.clf(out)
        return out


class MixerLayer(nn.Module):
    def __init__(self, num_patches, hidden_size, hidden_s, hidden_c, drop_p, off_act,use_monarch=False,**kwargs):
        super(MixerLayer, self).__init__()
        #breakpoint()
        self.use_monarch = use_monarch
        self.num_patches = num_patches
        if self.use_monarch:
            self.ln1 = nn.LayerNorm(hidden_size)
            self.mc1 = MonarchConv(hidden_size,num_patches,**kwargs)  #First argument is hidden dim, second argument is sequence dim
            self.mc2 = MonarchConv(num_patches,hidden_size,**kwargs)
            self.ln2 = nn.LayerNorm(hidden_size)
        else:
            self.mlp1 = MLP1(num_patches, hidden_s, hidden_size, drop_p, off_act)
            self.mlp2 = MLP2(hidden_size, hidden_c, drop_p, off_act)
    def forward(self, x):
         #x = (128, 64, 128) (batch_sz, N_patches, Hidden_dim)
        if not self.use_monarch:
            out = self.mlp1(x)
            out = self.mlp2(out)
            #The first mlp is over the N_patches dimension (i.e. the sequence length) and the second mlp is over Hidden_dim,
            #We first mix over sequence dim (like ViT multihead attention) we then mix features
        else:
            x = rearrange(x,"b n h -> b h n")
            out = self.mc1(torch.transpose(self.ln1(torch.transpose(x,-2,-1)),-2,-1))+x
            out = rearrange(out,"b h n -> b n h")
            #breakpoint()
            out = self.mc2(self.ln2(out))+out
        
        return out

class MLP1(nn.Module):
    def __init__(self, num_patches, hidden_s, hidden_size, drop_p, off_act):
        super(MLP1, self).__init__()
        self.ln = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Conv1d(num_patches, hidden_s, kernel_size=1)
        self.do1 = nn.Dropout(p=drop_p)
        self.fc2 = nn.Conv1d(hidden_s, num_patches, kernel_size=1)
        self.do2 = nn.Dropout(p=drop_p)
        self.act = F.gelu if not off_act else lambda x:x
    def forward(self, x):
        out = self.do1(self.act(self.fc1(self.ln(x))))
        out = self.do2(self.fc2(out))
        return out+x

class MLP2(nn.Module):
    def __init__(self, hidden_size, hidden_c, drop_p, off_act):
        super(MLP2, self).__init__()
        self.ln = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_c)
        self.do1 = nn.Dropout(p=drop_p)
        self.fc2 = nn.Linear(hidden_c, hidden_size)
        self.do2 = nn.Dropout(p=drop_p)
        self.act = F.gelu if not off_act else lambda x:x
    def forward(self, x):
        out = self.do1(self.act(self.fc1(self.ln(x))))
        out = self.do2(self.fc2(out))
        return out+x

if __name__ == '__main__':
    net = MLPMixer(
        in_channels=3,
        img_size=32, 
        patch_size=4, 
        hidden_size=128, 
        hidden_s=512, 
        hidden_c=64, 
        num_layers=8, 
        num_classes=10, 
        drop_p=0.,
        off_act=False,
        is_cls_token=True
        )
    torchsummary.summary(net, (3,32,32))
