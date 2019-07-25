import torch
from modules.Flatten import Flatten

class CNN1d(torch.nn.Module):
    def __init__(self,input_size,conv_seq_len,hidden_size,conv_kernel,pool_kernel ,padding,
                 stride=1,dilation=1, dropout=0.0,output_size=1):
        super(CNN1d, self).__init__()
        self.num_layers=len(hidden_size)

        layers=[]
        for i, (h,c,p,pad,d) in enumerate(list(zip(hidden_size,conv_kernel,pool_kernel,padding,dilation))):
            s = input_size if i ==0 else hidden_size[i-1]
            layers+=[
                torch.nn.utils.weight_norm(
                    torch.nn.Conv1d(s,h,c,stride=1,padding=pad,dilation=d)),
                torch.nn.ReLU(),
                torch.nn.MaxPool1d(p,p,padding=pad,dilation=1),
                torch.nn.Dropout(dropout)
            ]
        layers+=[Flatten(),
                torch.nn.Linear(conv_seq_len,output_size),
                torch.nn.Sigmoid()]
        self.model=torch.nn.Sequential(*layers)

    def forward(self,subject):
        return self.model(subject)
