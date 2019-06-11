import torch

class HierarchicalCNN1d(torch.nn.Module):
    """cf. report on the CNN code"""
    def __init__(self,input_size,seq_len,hidden_size,conv_kernel,pool_kernel ,padding=0,
                 stride=1,dilation=1, dropout=0.0,output_size=1,fold=0):
        super(HierarchicalCNN1d, self).__init__()

        out_size=seq_len
        if fold ==0:
            print("seq_len :",out_size)
        #dilation=dilation_factor**0
        self.conv1=torch.nn.utils.weight_norm(
            torch.nn.Conv1d(input_size,hidden_size[0],conv_kernel[0],stride=1,padding=padding,dilation=dilation[0]))
        self.relu1=torch.nn.ReLU()
        out_size=get_out_size(out_size,padding,dilation[0],conv_kernel[0],stride=1)
        if fold ==0:
            print("after conv1 :",out_size)
        self.pool1=torch.nn.MaxPool1d(pool_kernel[0],pool_kernel[0],padding,dilation=1)
        out_size=get_out_size(out_size,padding,dilation=1,kernel_size=pool_kernel[0],stride=pool_kernel[0])
        if fold ==0:
            print("after pool1 :",out_size)
        out_size*=5#5 because we concat 5 ls
        if fold ==0:
            print("after concat :",out_size)
        self.drop1=torch.nn.Dropout(dropout)
        #dilation=dilation_factor**1

        out_size=get_out_size(out_size,padding,dilation[1],conv_kernel[1],stride=1)
        while out_size <1:
            conv_kernel[1]-=1
            out_size=get_out_size(out_size,padding,dilation[1],conv_kernel[1],stride=1)
        self.conv2=torch.nn.utils.weight_norm(
            torch.nn.Conv1d(hidden_size[0],hidden_size[1],conv_kernel[1],stride=1,padding=0,dilation=dilation[1]))
        self.relu2=torch.nn.ReLU()
        if fold ==0:
            print("after conv2 :",out_size)
        if pool_kernel[1]>out_size:
            pool_kernel[1]=int(out_size)
        self.pool2=torch.nn.MaxPool1d(pool_kernel[1],pool_kernel[1],padding,dilation=1)
        out_size=get_out_size(out_size,padding,dilation=1,kernel_size=pool_kernel[1],stride=pool_kernel[1])
        if fold ==0:
            print("after pool2 :",out_size)

        self.drop2=torch.nn.Dropout(dropout)
        self.linear1=torch.nn.Linear(out_size*hidden_size[-1],output_size)#5 because 5 ls

        """self.relu2=torch.nn.ReLU()
        self.drop3=torch.nn.Dropout(dropout)
        self.linear2=torch.nn.Linear(10,output_size)    """

        self.sigmoid=torch.nn.Sigmoid()
    def forward(self,subject):
        #save_flat=[]
        save_feats=[]
        for seq in subject:
            c1=self.conv1(seq)
            r1=self.relu1(c1)
            p1=self.pool1(r1)

            save_feats.append(p1)
            """drop1=self.drop1(p1)
            c2=self.conv2(drop1)
            r2=self.relu2(c2)
            p2=self.pool2(r2)
            #flatten
            flat=p2.view(p2.size(0), -1)
            save_flat.append(flat)"""
        #cat=torch.cat(save_flat,dim=1)
        cat=torch.cat(save_feats,dim=2)
        drop1=self.drop1(cat)
        c2=self.conv2(drop1)
        r2=self.relu2(c2)
        p2=self.pool2(r2)
        #flatten
        flat=p2.view(p2.size(0), -1)
        drop2=self.drop2(flat)#cat)
        l1=self.linear1(drop2)

        """r2=self.relu2(l1)
        drop3=self.drop3(r2)
        l2=self.linear2(drop3)"""
        #l3=self.linear3(drop2)
        return self.sigmoid(l1)