import torch

class HierarchicalCNN1d(torch.nn.Module):
    """cf. report on the CNN code"""
    def __init__(self,input_size,conv_seq_len,hidden_size,conv_kernel,pool_kernel ,padding=0,
                 stride=1,dilation=1, dropout=0.0,output_size=1):
        super(HierarchicalCNN1d, self).__init__()
        self.conv1=torch.nn.utils.weight_norm(
            torch.nn.Conv1d(input_size,hidden_size[0],conv_kernel[0],stride=1,padding=padding,dilation=dilation[0]))
        self.relu1=torch.nn.ReLU()
        self.pool1=torch.nn.MaxPool1d(pool_kernel[0],pool_kernel[0],padding,dilation=1)
        self.drop1=torch.nn.Dropout(dropout)
        self.conv2=torch.nn.utils.weight_norm(
            torch.nn.Conv1d(hidden_size[0],hidden_size[1],conv_kernel[1],stride=1,padding=padding,dilation=dilation[1]))
        self.relu2=torch.nn.ReLU()
        self.pool2=torch.nn.MaxPool1d(pool_kernel[1],pool_kernel[1],padding,dilation=1)
        self.drop2=torch.nn.Dropout(dropout)
        self.linear1=torch.nn.Linear(conv_seq_len,output_size)
        self.sigmoid=torch.nn.Sigmoid()
    def forward(self,subject):
        save_feats=[]
        for seq in subject:
            c1=self.conv1(seq)
            r1=self.relu1(c1)
            p1=self.pool1(r1)
            save_feats.append(p1)
        cat=torch.cat(save_feats,dim=2)
        drop1=self.drop1(cat)
        c2=self.conv2(drop1)
        r2=self.relu2(c2)
        p2=self.pool2(r2)
        #flatten
        flat=p2.view(p2.size(0), -1)
        drop2=self.drop2(flat)#cat)
        l1=self.linear1(drop2)
        return self.sigmoid(l1)
