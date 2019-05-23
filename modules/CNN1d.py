import torch

class CNN1d(torch.nn.Module):
    def __init__(self,in_channels, out_channels,kernel_size=1 , dropout=0.0,output_size=1):
        super(CNN1d, self).__init__()
        self.conv=torch.nn.Conv1d(in_channels, out_channels, kernel_size)
        self.relu=torch.nn.ReLU()
        self.pool=torch.nn.MaxPool1d(16317)
        self.drop=torch.nn.Dropout(dropout)
        self.linear=torch.nn.Linear(out_channels,output_size)
        self.sigmoid=torch.nn.Sigmoid()

    def forward(self,subject):
        c=self.conv(subject)
        relu=self.relu(c)
        #max pool
        m=self.pool(relu).squeeze(2)
        drop=self.drop(m)
        out=self.linear(drop)
        return self.sigmoid(out)

    
