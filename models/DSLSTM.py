import random
import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, bias, kernel_size):
        """
        Initialize TransConvLSTM cell.
        :param input_dim:int
                Number of channels of input tensor.
        :param hidden_dim: int
                Number of channels of hidden state.
        :param kernel_size: (int, int)
                Size of the convolutional kernel.
        :param padding: (int, int)
                controls the amount of implicit zero-paddings on both sides for padding number of points for each dimension, Default: (0, 0).
        :param stride: (int, int)
                Controls the stride for the cross-correlation, a tuple, Default: (1, 1).
        :param bias: bool
                Whether or not to add the bias.Default: True.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = 1, 0
        self.bias = bias
        # print(self.kernel_size)
        self.input_conv = nn.Conv2d(in_channels=self.input_dim,
                                    out_channels=4 * self.hidden_dim,
                                    kernel_size=self.kernel_size,
                                    padding = (1,0),
                                    bias=self.bias)

        self.hidden_conv = nn.Conv2d(in_channels=self.hidden_dim,
                                     out_channels = 4 * self.hidden_dim,
                                     kernel_size=(3, 1),
                                     padding = (1, 0),
                                     bias = self.bias)

    def forward(self, input_tensor, pre_state):
        """
        The process of forwarding.
        :param input_tensor: tensor
                The input tensor.
        :param pre_state: tuple
                The initial pre_hidden state included memory state (ht-1) and carry state (ct-1).
        :return:Return the hidden state at present, included memory state (ht) and carry state (ct).
        """

        pre_h, pre_c = pre_state
        # 前向计算的公式
        input_tensor = input_tensor.contiguous()
        pre_h = pre_h.contiguous()
        # print(input_tensor.size())
        input_conv = torch.relu(self.input_conv(input_tensor))
        hidden_conv = torch.relu(self.hidden_conv(pre_h))
        # 这里需要进行一个分割， 因为一共有四个公式涉及到了卷积
        xi, xf, xo, xc = torch.split(input_conv, self.hidden_dim, dim=1)
        hi, hf, ho, hc = torch.split(hidden_conv, self.hidden_dim, dim=1)

        input_gate = torch.sigmoid(xi + hi)
        forget_gate = torch.sigmoid(xf + hf)
        output_gate = torch.sigmoid(xo + ho)
        new_c = torch.tanh(xc + hc)

        c_next = forget_gate * pre_c + input_gate * new_c
        h_next = output_gate * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        """
        Caculate the Output'size of input tensor's convolution and initialize the hidden state.
        :param batch_size: int
                Size of mini-batch.
        :param image_size: tuple
                Size of the input_tensor.
        :return:Return the initial pre_hidden state included memory state (ht-1) and carry state (ct-1).
        """
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.hidden_conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.hidden_conv.weight.device))

class ConvLSTM(nn.Module):
    """
    Parameters:
        input_dim: Number of channels in input.
        hidden_dim: Number of hidden channels.
        kernel_size: Size of kernel in convolutions.
        num_layers: Number of LSTM layers stacked on each other.
        batch_first: Whether or not dimension 0 is the batch or not.
        bias: Bias or no bias in Convolution.
        return_sequences: Return the last cells's output (False, 4D Tensor) or the whole cell's output (True, 5D Tensor) in the final layer. Default: True
    Input:
        A tensor of size Batch, Time, Channel, Height, Width or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    """

    def __init__(self, input_dim, hidden_dim, kernel_size,  num_layers=1,
                 batch_first=True, bias=True, return_sequences=True):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)


        # Make sure that both 'kernel_size' and 'hidden_dim' are lists having len == num_layers
        # 假如整个convlstm模型的层数大小为3，那么我们应该确保存放卷积核大小的列表为3
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)

        if not len(kernel_size) == len(hidden_dim) == num_layers:  # 长度不匹配肯定要报错啦----
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_sequences = return_sequences

        cell_list = []
        for i in range(0, self.num_layers):  # According the num_layers , stack the ConvLSTM layers.
            # If current layer is the first layer,  cur_input_dim = input_dim. If not, cur_input_dim = last layers' hidden_dim
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          bias=self.bias,
                                          kernel_size=kernel_size[i]))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor:
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state:
        Returns
        -------
        last_state_list, layer_output
        """

        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            # If batch is not first ,we need permute to make sure that the batch is first.
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)


        b, _, _, h, w = input_tensor.size()

        layer_output_list = []
        seq_len = input_tensor.size(1) # In fact, the seq_len is Size of time window.
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            # Implement stateful ConvLSTM
            if hidden_state is not None:
                raise NotImplementedError()
            else:
                # Initialize the current layer's hidden state.
                # Every layer's parameters are independent.
                h, c = self._init_hidden(number_layer=layer_idx,
                                         batch_size=b,
                                         image_size= (h, w)
                                         )
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 pre_state=[h, c])  # Temporal direction's caculating and propagating.
                output_inner.append(h)
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            if layer_idx == self.num_layers - 1:
                layer_output_list.append(layer_output)
        if self.return_sequences:
            # return the final layer's all cell's output.
            return layer_output_list[0]
        else:
            # Return the final layer's final cell's output.
            return torch.unsqueeze(layer_output_list[0][:, -1, :], dim=1)

    def _init_hidden(self, number_layer, batch_size, image_size):
        """
        Initialize the current layer's initial hidden state.
        :param layer_index: The index of current layer.
        :param batch_size: The size of mini-batch.
        :return: Return the initialized hidden state.
        """
        h, c = self.cell_list[number_layer].init_hidden(batch_size=batch_size, image_size=image_size)
        return h, c

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        """
        Check the type of 'kernel_size'.
        :param kernel_size: The input tensor's cross convolution's kernel_size.
        :return: None
        """
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            # Format one: kernel_size = (1,3)  Later, we will use function '_extend_for_multilayer'to extend and copy the tuple to the list. i.e. Every layer has same kernel_size.
            # Fornat two: kernel_size = [(1,3), (1, 2)]. Different layer has different kernel_size. Nothing needs to be done
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        """
        Extend and copy the tuple to the list. i.e. Every layer has same kernel_size.
        :param param: kernel_size, hidden_dim, padding, stride, output_padding
        :param num_layers: The size of num_layer.s
        :return: Return the extended num_layers.
        """
        # param: kernel_size = (1, 3), num_layers = 3
        # return: [(1, 3), (1, 3), (1, 3)]
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

class Temporal_Attention(nn.Module):
    "Temporal attention in encoder"
    def __init__(self, hidden_size):
        super(Temporal_Attention, self).__init__()
        self.atten_linear = nn.Linear(in_features=hidden_size, out_features=hidden_size)

    def forward(self, x):

        query = torch.unsqueeze(x[-1], dim=-1)
        value = x[0]
        self.batch_size = value.size()[0]

        key = self.atten_linear(value)
        score = torch.bmm(key, query)
        score = nn.functional.softmax(score, dim=1)
        context = torch.matmul(score.permute(0, 2, 1), value)
        return context

class Fusion_Attention(nn.Module):
    "Fusion attention in decoder"
    def __init__(self, hidden_size):
        super(Fusion_Attention, self).__init__()
        # self.hidden_size = hidden_size
        self.atten_linear = nn.Linear(in_features=hidden_size,
                                      out_features=hidden_size)

    def forward(self, x):
        lstm_out = x[0]
        lstm_hidden = x[1]
        lstm_tmp_out = torch.chunk(lstm_out, 2, -1)
        h = lstm_tmp_out[0] + lstm_tmp_out[1]
        lstm_hidden = torch.sum(lstm_hidden, dim=1).unsqueeze(1)
        atten_w = self.atten_linear(lstm_hidden)
        m = torch.tanh(h)
        atten_context = torch.bmm(atten_w, m.transpose(1, 2))
        softmax_w = torch.softmax(atten_context, dim=-1)
        context = torch.bmm(softmax_w, h)
        result = context.squeeze(1)
        return result

class Model(nn.Module):
    """
    Build the conv_lstm_res network;
    """

    def __init__(self, input_dim, hidden_dim, output_dim, kernel_size, lstmhidden_dim, horizon, dropout):
        super(Model, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.lstmhidden_dim = lstmhidden_dim
        self.horizon = horizon

        # encoder--convlstm
        self.convlstm = ConvLSTM(input_dim=self.input_dim,
                                 hidden_dim=self.hidden_dim,
                                 kernel_size=(3, 1),
                                 num_layers=1,
                                 return_sequences=False)

        self.conv_tail = ConvLSTM(input_dim=self.input_dim,
                                 hidden_dim=self.hidden_dim,
                                 kernel_size=(3, 1),
                                 num_layers=1,
                                 return_sequences=False)

        # decoder--bilstm
        self.bilstm = nn.Sequential(nn.LSTM(input_size=self.hidden_dim,
                                            hidden_size=self.lstmhidden_dim,
                                            batch_first=True,
                                            bidirectional=True,
                                            num_layers=1))    #双向LSTM

        self.dropout = nn.Dropout(p=dropout)

        self.tail_linear = nn.Linear(in_features=self.hidden_dim,
                                     out_features=self.input_dim)

        self.linear_1 = nn.Linear(in_features=self.lstmhidden_dim,
                                  out_features=output_dim)

        self.res_linear = nn.Linear(in_features=7,
                                    out_features=1)

        self.temporal_attention_tail = Temporal_Attention(hidden_size=self.hidden_dim)
        self.fusion_attention = Fusion_Attention(hidden_size=self.lstmhidden_dim)

    def forward(self, x):
        x = x.reshape(x.shape[0], 7, 1, 24, x.shape[-1])

        x = x.permute(0, 1, 4, 3, 2)


        # 对长期的输入进行一个编码
        encoder_out = self.convlstm(x)
        short_x_tail = torch.cat([x[:, :, :, :3, :], self.tail_linear(encoder_out.permute(0, 1, 4, 3, 2)).permute(0, 1, 4, 3, 2)[:, :, :, :3, :]], dim = 1)
        encoder_out= encoder_out.squeeze(1).squeeze(-1).permute(0, 2, 1)

        short_encoder_out_tail = self.conv_tail(short_x_tail).squeeze(-1).squeeze(1).permute(0, 2, 1)
        short_encoder_out_tail = self.temporal_attention_tail([short_encoder_out_tail, encoder_out[:, self.horizon-1, :]])

        # 解码
        decoder_out, _ = self.bilstm(torch.cat([encoder_out, short_encoder_out_tail], dim=1))
        last_hidden_state = _[0].permute(1, 0, 2)
        result = self.fusion_attention([decoder_out[:, self.horizon-1: self.horizon-1 + 2, :], last_hidden_state])
        result = self.dropout(result)

        # 残差连接
        res = torch.zeros([x.shape[0], 7, self.input_dim]).cuda()
        for i in range(7):
            res[:, i, :] = x[:, i, :, self.horizon - 1, :].squeeze(-1)
        res = res.permute(0, 2, 1)
        res = self.res_linear(res)
        res = res.squeeze(-1)


        final_out = self.linear_1(result) + res

        return torch.sigmoid(final_out)

