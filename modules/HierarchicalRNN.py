import torch
from .utils import on_paper_value, measure2index

class HierarchicalRNN(torch.nn.Module):
    """
    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
            Default: ``True``
        dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
        LSTM layer except the last layer, with dropout probability equal to
        :attr:`dropout`. Default: 0
        batch_size : default : 1
        output_size : default : 1
        is_lstm : default : True
    """
    def __init__(self, input_size, hidden_size, bias=True,dropout=0.0, batch_size=1,output_size=1,is_lstm=True):
        super(HierarchicalRNN, self).__init__()
        #Vanilla LSTM
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias=bias
        self.dropout=dropout
        self.batch_size=batch_size
        self.output_size = output_size
        self.is_lstm=is_lstm

        # Define the LSTM layers
        if self.is_lstm:
            self.layer1=torch.nn.LSTMCell(self.input_size,self.hidden_size,self.bias)
            self.layer1_air=torch.nn.LSTMCell(self.input_size,self.hidden_size,self.bias)
            self.layer2=torch.nn.LSTMCell(self.hidden_size,self.hidden_size,self.bias)
        else:
            self.layer1=torch.nn.GRUCell(self.input_size,self.hidden_size,self.bias)
            self.layer1_air=torch.nn.GRUCell(self.input_size,self.hidden_size,self.bias)
            self.layer2=torch.nn.GRUCell(self.hidden_size,self.hidden_size,self.bias)

        #define the dropout layer
        self.dropout_layer=torch.nn.Dropout(self.dropout)

        # Define the decoder layer
        self.linear = torch.nn.Linear(self.hidden_size, self.output_size)
        self.sigmoid = torch.nn.Sigmoid()
        self.reset_hidden()
        #if self.is_lstm:
            #self.init_forget_bias()
            #gives poor result on hierarchical CRNN on l task

    def forward(self, input):
        """should take a subject as input
        with subject = [seq1, seq2, ..., seqn]"""
        layer1_hidden=[]
        for seq in input:
            if seq[0][0][measure2index["button_status"]]==on_paper_value:
                for timestep in seq:#forward pass through the seq
                    if self.is_lstm:
                        self.h_0,self.c_0=self.layer1(timestep,(self.h_0,self.c_0))
                    else:
                        self.h_0=self.layer1(timestep,self.h_0)
            else:
                for timestep in seq:#forward pass through the seq
                    if self.is_lstm:
                        self.h_0,self.c_0=self.layer1_air(timestep,(self.h_0,self.c_0))
                    else:
                        self.h_0=self.layer1_air(timestep,self.h_0)
            layer1_hidden.append(self.h_0)#take only the last hidden state of the letter (i.e. the letter encoding)
            self.reset_hidden()#reset hidden state after each subsequence
        for hidden in layer1_hidden:#feed all the letters encoding to the 2nd layer
            self.dropout_layer(hidden)#we could either apply the feedforward dropout here or on : `layer1_hidden.append(h_0) `
            if self.is_lstm:
                self.h_1,self.c_1=self.layer2(hidden,(self.h_1,self.c_1))
            else:
                self.h_1=self.layer2(hidden,self.h_1)

        # Only take the output from the final timestep
        drop=self.dropout_layer(self.h_1)
        self.reset_hidden()#reset hidden state after each subject
        y_pred = self.linear(drop)
        y_pred = self.sigmoid(y_pred)
        return y_pred

    def reset_hidden(self,device="cuda"):
        """
        tensor containing the initial cell state for each element in the batch.

        The hidden state(s) is (are) modified in place."""

        self.h_0=torch.zeros((self.batch_size, self.hidden_size),device=device)
        self.h_1=torch.zeros((self.batch_size, self.hidden_size),device=device)
        if self.is_lstm:
            self.c_0=torch.zeros((self.batch_size, self.hidden_size),device=device)
            self.c_1=torch.zeros((self.batch_size, self.hidden_size),device=device)

    def init_forget_bias(self):
        """Following advices of Jozefowicz et al. 2015,
        we initialize the bias of the forget gate to a large value such as 1
        In PyTorch, the forget gate bias is stored as b_hf in bias_hh_l[k] :
        the learnable hidden-hidden bias of the kth layer (b_hi|b_hf|b_hg|b_ho), of shape (4*hidden_size).
        So b_hf == bias_hh_lk[hidden_size:2*hidden_size]

        The weights are modified in-place, like reset_hidden(self).
        """
        with torch.no_grad():#so the optimizer doesn't know about this ;)
            self.layer1.bias_hh[self.hidden_size:2*self.hidden_size]=torch.ones(self.hidden_size)
            self.layer1_air.bias_hh[self.hidden_size:2*self.hidden_size]=torch.ones(self.hidden_size)
            self.layer2.bias_hh[self.hidden_size:2*self.hidden_size]=torch.ones(self.hidden_size)
