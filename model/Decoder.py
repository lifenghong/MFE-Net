import torch
import torch.nn as nn

from model.AGFF import AGFF


class Res_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Res_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out
    
class Res_Decoder(nn.Module):  ###
	def __init__(self, num_classes=1, block=None):
		super(Res_Decoder, self).__init__()
		self.num_class = num_classes
		# resent18输出=[64，128，256，512]
		input_channels = [6, 64, 128, 256, 512, 1024]
		self.layer4 = self._make_layer(block, input_channels[4],
		                               input_channels[3])  # 1/8
		self.layer3 = self._make_layer(block, input_channels[3],
		                               input_channels[2])  # 1/4
		self.layer2 = self._make_layer(block, input_channels[2],
		                               input_channels[1])
		self.layer1 = self._make_layer(block, input_channels[1] + input_channels[0],
		                               input_channels[0])
		self.final = nn.Conv2d(input_channels[0], num_classes, kernel_size=1)
		self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
		self.agff4 = AGFF(1024, 512, is_bottom=False)  # 512
		self.agff3 = AGFF(256, 256, is_bottom=False)  # 256
		self.agff2 = AGFF(128, 128, is_bottom=False)  # 256
	
	def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
		layers = []
		layers.append(block(input_channels, output_channels))
		for i in range(num_blocks - 1):
			layers.append(block(output_channels, output_channels))
		return nn.Sequential(*layers)
	
	def forward(self, Features):
		x0, x1, x2, x3, x4 = Features[0], Features[1], Features[2], Features[3], Features[4]
		x4_3 = self.agff4(self.up(x4), x3)
		x4_3 = self.layer4(x4_3)  ##16->8
		x3_2 = self.agff3(self.up(x4_3), x2)
		x3_2 = self.layer3(x3_2)  ##8->4
		x2_1 = self.agff2(self.up(x3_2), x1)
		x2_1 = self.layer2(x2_1)  ##4->2
		x1_0 = self.layer1(torch.cat([self.up(x2_1), x0], dim=1))  ##2->1
		out = self.final(x1_0)  ##out
		return out
	

