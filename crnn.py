import torch.nn as nn
import pdb

class BidirectionalLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True)
        self.embedding = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size() # 26, 1, 512
        t_rec = recurrent.view(T * b, h)
        output = self.embedding(t_rec)  # [T * b, num_out]
        output = output.view(T, b, -1)
        return output

class CRNN(nn.Module):

    def __init__(self, image_height, column_size, class_size, hidden_size, leakyRelu=False):
        super(CRNN, self).__init__()
        assert image_height % 16 == 0, 'Image height has to be a multiple of 16'

        kernal_size = [3, 3, 3, 3, 3, 3, 2]
        padding_size = [1, 1, 1, 1, 1, 1, 0]
        stride_size = [1, 1, 1, 1, 1, 1, 1]
        map_size = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            input_size = column_size if i == 0 else map_size[i - 1]
            output_size = map_size[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(input_size, output_size, kernal_size[i], stride_size[i], padding_size[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(output_size))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2), nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3), nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16

        self.cnn = cnn

        # Deep BidirectionalLSTM
        self.rnn = nn.Sequential(BidirectionalLSTM(512, hidden_size, hidden_size)
                                 , BidirectionalLSTM(hidden_size, hidden_size, class_size))

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output = self.rnn(conv)
        return output