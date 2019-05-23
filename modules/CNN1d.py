class CNN1d(torch.nn.Module):
    def __init__(self,in_channels, out_channels, dropout=0.0,kernel_size=1,output_size=1,max_len=16071):
        super(CNN1d, self).__init__()
        self.conv=torch.nn.Conv1d(in_channels, out_channels, kernel_size)
        self.pool=torch.nn.MaxPool1d(max_len)
        self.drop=torch.nn.Dropout(dropout)
        self.relu=torch.nn.ReLU()
        self.linear=torch.nn.Linear(out_channels,output_size)
        self.sigmoid=torch.nn.Sigmoid()

    def forward(self,subject):
        c=self.conv(subject)
        #max pool
        m=self.pool(c).squeeze(2)
        drop=self.drop(m)
        relu=self.relu(drop)
        out=self.linear(relu)
        return self.sigmoid(out)

    def count_params(self):
        """returns (total n° of parameters, n° of trainable parameters)"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params
