#machine learning
import torch
# Model
"""Cf. Report on the code for details about the architecture of the model

- [**Pytorch LSTM doc**](https://pytorch.org/docs/stable/nn.html#torch.nn.LSTM)
- [**Pytorch GRU doc**](https://pytorch.org/docs/stable/nn.html#torch.nn.GRU)
- [**Pytorch Linear doc**](https://pytorch.org/docs/stable/nn.html#torch.nn.Linear)
- [**Pytorch Binary Cross Entropy loss (BCELoss) doc**](https://pytorch.org/docs/stable/nn.html#torch.nn.BCELoss)"""
class Attn(torch.nn.Module):
    def __init__(self, input_size):

        super(Attn, self).__init__()
        self.input_size = input_size
        self.weight=torch.nn.Parameter(torch.empty(input_size,1,1))
        torch.nn.init.xavier_uniform_(self.weight)
    def forward(self,encoder_out):
        attn=self.weight[:len(encoder_out)]*encoder_out
        return torch.sum(attn,dim=0)

class Model(torch.nn.Module):
    """
    Args:
        #Vanilla LSTM/GRU
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
            would mean stacking two LSTMs together to form a `stacked LSTM`,
            with the second LSTM taking in outputs of the first LSTM and
            computing the final results. Default: 1
        bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
            Default: ``True``
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False``
        dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
            LSTM layer except the last layer, with dropout probability equal to
            :attr:`dropout`. Default: 0
        bidirectional: If ``True``, becomes a bidirectional LSTM. Default: ``False``

        #our model
        batch_size : default : 1
        output_size : default : 1
        is_lstm : default : True
    """
    def __init__(self, input_size, hidden_size,num_layers=1, bias=True,batch_first=False,
                 dropout=0,bidirectional=False, batch_size=1, output_size=1,is_lstm=True):
        super(Model, self).__init__()
        #Vanilla LSTM/GRU
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias=bias
        self.batch_first=batch_first
        self.dropout=dropout
        self.bidirectional=bidirectional
        #our model
        self.batch_size = batch_size
        self.output_size = output_size
        self.is_lstm=is_lstm
        self.reset_hidden()
        # Define the encoder (i.e. GRU or LSTM) layer
        if self.is_lstm:
            self.encoder = torch.nn.LSTM(self.input_size, self.hidden_size, self.num_layers,self.bias,self.batch_first,
                            self.dropout,self.bidirectional)
        else:
            self.encoder = torch.nn.GRU(self.input_size, self.hidden_size, self.num_layers,self.bias,self.batch_first,
                            self.dropout,self.bidirectional)

        #define the dropout layer
        self.dropout_layer=torch.nn.Dropout(self.dropout)

        # Define the decoder layer
        self.linear = torch.nn.Linear(self.hidden_size, self.output_size)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input):
        # Forward pass through encoder layer
        # shape of encoder_out: (seq_len, batch, num_directions * hidden_size)
        # shape of self.hidden: (h_n, c_n), where hidden state h_n and cell state c_n both
        # have shape (num_layers * num_directions, batch, hidden_size).
        if self.is_lstm:
            encoder_out, (self.hidden_state, self.cell_state) = self.encoder(input,(self.hidden_state, self.cell_state))
        else:#if GRU
            encoder_out, self.hidden_state= self.encoder(input,self.hidden_state)

        if self.bidirectional:
            #sums the outputs : direction left-right and direction right-left
            # encoder_out shape should now be (seq_len, batch,hidden_size)
            encoder_out = encoder_out[: ,: ,: self.hidden_size] + encoder_out[: , :, self.hidden_size: ]

        # Only take the output from the final timestep
        encoding=encoder_out[-1]
        drop=self.dropout_layer(encoding)
        y_pred = self.linear(drop)
        y_pred = self.sigmoid(y_pred)
        return y_pred

    def reset_hidden(self,device="cuda"):
        """
        For both GRU and LSTM :
        hidden_state of shape (num_layers * num_directions, batch, hidden_size):
        tensor containing the initial hidden state for each element in the batch.
        If the RNN is bidirectional, num_directions should be 2, else it should be 1.

        For LSTM :
        cell_state of shape (num_layers * num_directions, batch, hidden_size):
        tensor containing the initial cell state for each element in the batch.

        The hidden state(s) is (are) modified in place."""

        num_directions=1
        if self.bidirectional:
            num_directions=2
        self.hidden_state=torch.zeros(self.num_layers*num_directions, self.batch_size, self.hidden_size,device=device)
        if self.is_lstm:
            self.cell_state=torch.zeros(self.num_layers*num_directions, self.batch_size, self.hidden_size,device=device)

    def init_forget_bias(self):
        """Following advices of Jozefowicz et al. 2015,
        we initialize the bias of the forget gate to a large value such as 1
        In PyTorch, the forget gate bias is stored as b_hf in bias_hh_l[k] :
        the learnable hidden-hidden bias of the kth layer (b_hi|b_hf|b_hg|b_ho), of shape (4*hidden_size).
        So b_hf == bias_hh_lk[hidden_size:2*hidden_size]

        The weights are modified in-place, like reset_hidden(self).
        """
        gen=self.modules()
        _=next(gen)#model summary : don't care about it
        lstm=next(gen)
        if not isinstance(lstm,torch.nn.LSTM):
            raise NotImplementedError("the encoder should be an LSTM and should be the first module of the model")
        with torch.no_grad():#so the optimizer doesn't know about this ;)
            lstm.bias_hh_l0[self.hidden_size:2*self.hidden_size]=torch.ones(lstm.hidden_size)
            if lstm.bidirectional:
                lstm.bias_hh_l0_reverse[self.hidden_size:2*self.hidden_size]=torch.ones(lstm.hidden_size)
            if lstm.num_layers > 1:
                lstm.bias_hh_l1[self.hidden_size:2*self.hidden_size]=torch.ones(lstm.hidden_size)
                if lstm.bidirectional:
                    lstm.bias_hh_l1_reverse[self.hidden_size:2*self.hidden_size]=torch.ones(lstm.hidden_size)
            if lstm.num_layers > 2:
                lstm.bias_hh_l2[self.hidden_size:2*self.hidden_size]=torch.ones(lstm.hidden_size)
                if lstm.bidirectional:
                    lstm.bias_hh_l2_reverse[self.hidden_size:2*self.hidden_size]=torch.ones(lstm.hidden_size)
            if lstm.num_layers > 3:
                lstm.bias_hh_l3[self.hidden_size:2*self.hidden_size]=torch.ones(lstm.hidden_size)
                if lstm.bidirectional:
                    lstm.bias_hh_l3_reverse[self.hidden_size:2*self.hidden_size]=torch.ones(lstm.hidden_size)
            if lstm.num_layers > 4:
                lstm.bias_hh_l4[self.hidden_size:2*self.hidden_size]=torch.ones(lstm.hidden_size)
                if lstm.bidirectional:
                    lstm.bias_hh_l4_reverse[self.hidden_size:2*self.hidden_size]=torch.ones(lstm.hidden_size)
            if lstm.num_layers>5:
                raise NotImplementedError("you can only have max 5 layers for now")
