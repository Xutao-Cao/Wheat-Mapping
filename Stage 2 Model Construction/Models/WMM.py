"""ConvLSTM was implemented by https://github.com/ndrplz/ConvLSTM_pytorch"""
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from torch.nn.init import xavier_normal_

class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):

    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.
    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=True, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

class BiConvLSTM(nn.Module):
    """output dim will be hidden_dim*2"""
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers) -> None:
        super().__init__()
        self.forward_convlstm = ConvLSTM(input_dim, hidden_dim, kernel_size, num_layers)
        self.backward_convlstm = ConvLSTM(input_dim, hidden_dim, kernel_size, num_layers)
    
    def forward(self, x):
        out0 = self.forward_convlstm(x)[0][0]
        reversed_index = list(reversed(range(x.shape[1])))
        out1 = self.backward_convlstm(x[:,reversed_index,...])[0][0]
        out = torch.cat([out0, out1], 2)
        return out


class ResDoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, timesteps) -> None:
        super().__init__()
        if timesteps == 0:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3,  padding = 1, bias = False)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3,  padding = 1, bias = False)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, groups= timesteps , padding = 1, bias = False)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, groups= timesteps , padding = 1, bias = False)
        if in_channels != out_channels:
            if timesteps == 0:
                self.res_connection = nn.Sequential(nn.Conv2d(in_channels,out_channels, kernel_size = 1, padding = 0, bias = False), 
                                                nn.BatchNorm2d(out_channels))
            else:
                self.res_connection = nn.Sequential(nn.Conv2d(in_channels,out_channels, kernel_size = 1, groups= timesteps , padding = 0, bias = False), 
                                                nn.BatchNorm2d(out_channels))
            
        else:
            self.res_connection = nn.Sequential()
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.res_connection(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        out = self.relu(x + identity)
        out = self.relu(x)
        return out

class DownSampleBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, timesteps) -> None:
        super().__init__()
        self.down_sample = nn.Sequential(
            nn.MaxPool2d(2),
            ResDoubleConv(in_channels, out_channels, timesteps)
        )

    def forward(self, x):
        return self.down_sample(x)

class UpSampleBlock(nn.Module):
    """Transposedconv -> concatenate -> double conv"""

    def __init__(self, in_channels, out_channels, timesteps):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2, groups=timesteps)
        self.double_conv = ResDoubleConv(3*in_channels//2, out_channels, timesteps)

    def forward(self, x, down_sample_x):
        x = self.up(x)
        x = torch.cat([down_sample_x, x], dim=1)
        x = self.double_conv(x)
        return x

class _DenseLayer(nn.Module):
    def __init__(self,in_channels, out_channels, dropout) -> None:
        super().__init__()
        self.forward_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
    
    def forward(self, x,):
        return self.forward_op(x)
        
        
            
class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, block_len, dropout = 0.5) -> None:
        super().__init__()


class WMM(nn.Module):

    def __init__(self, n_channels, n_classes, timesteps,dropout = 0.2, n_feature_maps=120, n_convlstm = 1) -> None:
        super().__init__()
        self.channels = n_channels
        self.classes = n_classes
        self.timesteps = timesteps
        self.n_convlstm = n_convlstm
        self.doubleconv_0 = ResDoubleConv(n_channels, n_feature_maps, timesteps)
        self.down_0 = DownSampleBlock(n_feature_maps, 2 * n_feature_maps, timesteps)
        self.down_1 = DownSampleBlock(2 * n_feature_maps, 4 * n_feature_maps, timesteps)
        self.down_2 = DownSampleBlock(4 * n_feature_maps, 8 * n_feature_maps, timesteps)
        self.down_3 = DownSampleBlock(8 * n_feature_maps, 16 * n_feature_maps, timesteps)
        self.dense0 = _DenseLayer(16 * n_feature_maps, 16 * n_feature_maps, dropout)
        self.dense1 = _DenseLayer(32 * n_feature_maps, 16 * n_feature_maps, dropout)
        self.dense2 = _DenseLayer(48 * n_feature_maps, 16 * n_feature_maps, dropout)
        # self.dense3 = _DenseLayer(64* n_feature_maps, 16 * n_feature_maps, dropout)
        self.up_0 = UpSampleBlock(16 * n_feature_maps, 8 * n_feature_maps, timesteps)
        self.up_1 = UpSampleBlock(8 * n_feature_maps, 4 * n_feature_maps, timesteps)
        self.up_2 = UpSampleBlock(4 * n_feature_maps, 2 * n_feature_maps, timesteps)
        self.up_3 = UpSampleBlock(2 * n_feature_maps, n_feature_maps, timesteps)
        self.out = nn.Conv2d(n_feature_maps, n_classes, kernel_size=1,)
        self.convlstm0 = ConvLSTM(n_feature_maps // timesteps, n_feature_maps // timesteps, (3,3), n_convlstm, batch_first=True, bias = False)
        self.convlstm1 = ConvLSTM(2 * n_feature_maps // timesteps, 2 * n_feature_maps // timesteps, (3,3), n_convlstm, batch_first=True, bias = False)
        self.convlstm2 = ConvLSTM(4 * n_feature_maps // timesteps, 4 * n_feature_maps // timesteps, (3,3), n_convlstm, batch_first=True, bias = False)
        self.convlstm3 = ConvLSTM(8 * n_feature_maps // timesteps, 8 * n_feature_maps // timesteps, (3,3), n_convlstm, batch_first=True, bias = False)
        self.biconvlstm0 = BiConvLSTM(n_feature_maps // timesteps, n_feature_maps // timesteps, (3,3), n_convlstm)
        self.biconvlstm1 = BiConvLSTM(2 * n_feature_maps // timesteps, 2* n_feature_maps // timesteps, (3,3), n_convlstm)
        self.biconvlstm2 = BiConvLSTM(4 * n_feature_maps // timesteps, 4 * n_feature_maps // timesteps, (3,3), n_convlstm)
        self.biconvlstm3 = BiConvLSTM(8 * n_feature_maps // timesteps, 8 * n_feature_maps // timesteps, (3,3), n_convlstm)
    
    def forward(self, x):
        x = self._timeseq2img(x)
        x0 = self.doubleconv_0(x)
        x0_connect = self.biconvlstm0(self._img2timeseq(x0))
        x1 = self.down_0(x0)
        x1_connect = self.biconvlstm1(self._img2timeseq(x1))
        x2 = self.down_1(x1)
        x2_connect = self.biconvlstm2(self._img2timeseq(x2))
        x3 = self.down_2(x2)
        x3_connect= self.biconvlstm3(self._img2timeseq(x3))
        x = self.down_3(x3)
        dense0 = self.dense0(x)
        dense1 = self.dense1(torch.cat([x, dense0], 1))
        x = self.dense2(torch.cat([x, dense0, dense1], 1))
        # x = self.dense3(torch.cat([x, dense0, dense1, dense2], 1))   
        x = self.up_0(x, self._timeseq2img(x3_connect))
        x = self.up_1(x, self._timeseq2img(x2_connect))
        x = self.up_2(x, self._timeseq2img(x1_connect))
        x = self.up_3(x, self._timeseq2img(x0_connect))
        x = self.out(x)
        return x.squeeze()

    def _timeseq2img(self, x:torch.tensor):
        """
        input: (n, t, c, h, w)
        output: (n, t*c, h, w)
        """
        return x.view((x.size(0), -1, x.size(3), x.size(4)))

    def _img2timeseq(self, x:torch.tensor):
        """
        input: (n, t*c, h, w)
        output: (n, t, c, h, w)
        """
        return x.view(x.size(0), self.timesteps, -1, x.size(2), x.size(3))
    
    def extract_hidden(self, x):
        x = self._timeseq2img(x)
        x0 = self.doubleconv_0(x)
        x0_connect = self.biconvlstm0(self._img2timeseq(x0))
        x1 = self.down_0(x0)
        x1_connect = self.biconvlstm1(self._img2timeseq(x1))
        x2 = self.down_1(x1)
        x2_connect = self.biconvlstm2(self._img2timeseq(x2))
        x3 = self.down_2(x2)
        x3_connect= self.biconvlstm3(self._img2timeseq(x3))
        x = self.down_3(x3)
        dense0 = self.dense0(x)
        dense1 = self.dense1(torch.cat([x, dense0], 1))
        x = self.dense2(torch.cat([x, dense0, dense1], 1))
        # x = self.dense3(torch.cat([x, dense0, dense1, dense2], 1))   
        x = self.up_0(x, self._timeseq2img(x3_connect))
        x = self.up_1(x, self._timeseq2img(x2_connect))
        x = self.up_2(x, self._timeseq2img(x1_connect))
        x = self.up_3(x, self._timeseq2img(x0_connect))
        return x

