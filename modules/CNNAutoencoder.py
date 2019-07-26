import torch
from .Flatten import Flatten

class CNNAutoencoder(torch.nn.Module):
    def __init__(self,input_size,hidden_size,conv_kernel,pool_kernel ,padding,
                 stride=1,dilation=1, dropout=0.0,input_noise=0.0):
        super(CNNAutoencoder, self).__init__()
        self.num_layers=len(hidden_size)

        self.noise=torch.nn.Dropout(input_noise)
        layers=[]
        for i, (h,c,p,pad) in enumerate(list(zip(hidden_size,conv_kernel,pool_kernel,padding))):
            s = input_size if i ==0 else hidden_size[i-1]
            layers+=[
                torch.nn.Conv1d(s,h,c,stride=1,padding=pad,dilation=1),
                torch.nn.MaxPool1d(p),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout)
            ]
        self.encoder=torch.nn.Sequential(*layers)

        layers=[]
        for i, (h,c,p,pad) in enumerate(list(zip(hidden_size,conv_kernel,pool_kernel,padding))[::-1]):
            if i ==self.num_layers-1:
                s= input_size
                layers+=[
                    torch.nn.ConvTranspose1d(h,s,c,stride=p,padding=0,dilation=1)
                ]
            else :
                s = hidden_size[::-1][i+1]
                layers+=[
                    torch.nn.ConvTranspose1d(h,s,c,stride=p,padding=0,dilation=1),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(dropout)
                ]
        self.decoder=torch.nn.Sequential(*layers)

    def forward(self,subject):
        noisy=self.noise(subject)
        code=self.encoder(noisy)
        decode=self.decoder(code)
        return code,decode#self.sigmoid(d_c1)
