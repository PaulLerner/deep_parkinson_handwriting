import torch
from .utils import on_paper_value, measure2index

class HierarchicalStrokeCRNN(torch.nn.Module):
    """similar to HierarchicalCNN1d except that there's two separate convolutional layers
    for on_paper and in_air strokes
    The on_paper and in_air convs share the same hyperparameters (e.g. kernel size)"""
    def __init__(self,input_size,conv_seq_len,hidden_size,conv_kernel,pool_kernel ,padding=0,
                 stride=1,dilation=1, dropout=0.0,bidirectional=False,output_size=1,is_lstm=True):
        super(HierarchicalStrokeCRNN, self).__init__()
        self.hidden_size=hidden_size
        self.bidirectional=bidirectional
        self.is_lstm=is_lstm

        self.conv1=torch.nn.utils.weight_norm(
            torch.nn.Conv1d(input_size,hidden_size[0],conv_kernel[0],stride=1,padding=padding,dilation=dilation[0]))
        self.conv1_air=torch.nn.utils.weight_norm(
            torch.nn.Conv1d(input_size,hidden_size[0],conv_kernel[0],stride=1,padding=padding,dilation=dilation[0]))
        self.relu1=torch.nn.ReLU()
        self.pool1=torch.nn.MaxPool1d(pool_kernel[0],pool_kernel[0],padding=0,dilation=1)
        self.drop1=torch.nn.Dropout(dropout)

        if self.is_lstm:
            self.rnn=torch.nn.LSTM(hidden_size[0],hidden_size[1],1,batch_first=True,bidirectional=self.bidirectional)
        else:
            self.rnn=torch.nn.GRU(hidden_size[0],hidden_size[1],1,batch_first=True,bidirectional=self.bidirectional)
        self.drop2=torch.nn.Dropout(dropout)
        self.linear1=torch.nn.Linear(hidden_size[-1],output_size)
        self.sigmoid=torch.nn.Sigmoid()
        #self.init_forget_bias()#gives poor results on the l task with baseline
    def forward(self,subject):
        save_feats=[]
        for seq in subject:
            if seq[0][measure2index["button_status"]][0]==on_paper_value:
                c1=self.conv1(seq)
            else:
                c1=self.conv1_air(seq)
            r1=self.relu1(c1)
            p1=self.pool1(r1)
            save_feats.append(p1)
        cat=torch.cat(save_feats,dim=2)
        drop1=self.drop1(cat)
        #batch_first so input shape should be (batch, seq, feature)
        #cat shape is (batch, feature, seq) so we have to transpose (1,2)
        drop1=drop1.transpose(1,2)
        if self.is_lstm:
            rnn_out, (hidden_state, cell_state) = self.rnn(drop1)
        else:#if GRU
            rnn_out, hidden_state= self.rnn(drop1)
        #hidden_state has shape (num_layers * num_directions,batch, hidden_size)
        if self.bidirectional:
            #sums the outputs : direction left-right and direction right-left
            hidden_state=hidden_state[0]+hidden_state[1]

        hidden_state=hidden_state.squeeze(1)#remove layer dim
        drop2=self.drop2(hidden_state)
        l1=self.linear1(drop2)
        return self.sigmoid(l1)
    def init_forget_bias(self):
        """Following advices of Jozefowicz et al. 2015,
        we initialize the bias of the forget gate to a large value such as 1
        In PyTorch, the forget gate bias is stored as b_hf in bias_hh_l[k] :
        the learnable hidden-hidden bias of the kth layer (b_hi|b_hf|b_hg|b_ho), of shape (4*hidden_size).
        So b_hf == bias_hh_lk[hidden_size:2*hidden_size]

        The weights are modified in-place.
        """
        with torch.no_grad():#so the optimizer doesn't know about this ;)
            self.rnn.bias_hh_l0[self.hidden_size[1]:2*self.hidden_size[1]]=torch.ones(self.hidden_size[1])
