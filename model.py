from PIL import Image
import torch 
from torchvision.transforms import Resize, ToPILImage, ToTensor
from einops import rearrange
from torch.nn import Sigmoid, functional as F
from torch import nn, sin
import pennylane as qml



class Quanv(nn.Module):
    def __init__(self, n_qubits, out):
        
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev)
        def q_all_kern(inputs, weights_0, weights_1, weights_2, weight_3, weight_4, weights_5):
            qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.templates.StronglyEntanglingLayers(weights_0, wires=range(n_qubits))
            qml.templates.BasicEntanglerLayers(weights_1, wires=range(n_qubits))
            qml.Rot(*weights_2, wires=0)
            qml.RY(weight_3, wires=1)
            qml.RZ(weight_4, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.Rot(*weights_5, wires=0)
            return [qml.expval(qml.Z(i)) for i in range(n_qubits)[-out:]]

        weight_shapes = {
            "weights_0": (3, n_qubits, 3),
            "weights_1": (3, n_qubits),
            "weights_2": 3,
            "weight_3": 1,
            "weight_4": (1,),
            "weights_5": 3,
        }

        init_method = {
            "weights_0": torch.nn.init.normal_,
            "weights_1": torch.nn.init.uniform_,
            "weights_2": torch.tensor([1., 2., 3.]),
            "weight_3": torch.tensor(1.),  # scalar when shape is not an iterable and is <= 1
            "weight_4": torch.tensor([1.]),
            "weights_5": torch.tensor([1., 2., 3.]),
        }
        super().__init__()
        self.fc1 = qml.qnn.TorchLayer(q_all_kern, weight_shapes=weight_shapes, init_method=init_method)
        self.fc3 = nn.Softmax(dim=-1)

        # fig, ax = qml.draw_mpl(q_all_kern ,expansion_strategy='device')(torch.randn(n_qubits),
        #                     torch.randn(3, n_qubits, 3), torch.randn(3, n_qubits), torch.randn(3,),
        #                     torch.randn(1,), torch.randn(1, ), torch.randn(3,))
        # fig.savefig('./model.png')

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc3(x)
        return x

class Q_Conv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int=0, num_layers: int=0, num_qubits: int=0):
        super().__init__()
        if num_qubits == 0:
            num_qubits = kernel_size**2 * in_channels
        if num_layers == 0:
            num_layers = kernel_size**2
        assert num_qubits == kernel_size**2 * in_channels, "The kernel size must be a square of the number of qubits"
        dev = qml.device("default.qubit", wires=num_qubits)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.in_channels = in_channels
        self.out_channels = out_channels

        @qml.qnode(dev)
        def qnode(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(num_qubits))
            qml.BasicEntanglerLayers(weights, wires=range(num_qubits))
            return [qml.expval(qml.PauliZ(wires=list(range(num_qubits))[-1]))]

        weight_shapes = {"weights": (num_layers, num_qubits)}

        self.qlayer_list = nn.ModuleList([qml.qnn.TorchLayer(qnode, weight_shapes) for _ in range(self.out_channels)])

    def forward(self, x):
        assert len(x.shape) == 4, "The input tensor must be 4D"
        assert x.shape[1] == self.in_channels, "The number of input channels must be equal to the in_channels"
        res = list()
        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding))
        x = x.unfold(2, self.kernel_size, self.stride)
        x = x.unfold(3, self.kernel_size, self.stride)
        x = rearrange(x, 'b c h w i j -> b h w (c i j)')    
        bs, h, w, _ = x.shape
        for i in range(self.out_channels):
            res.append(self.qlayer_list[i](x).view(bs, h, w))        
        x = torch.stack(res, dim=1)
        return x

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__() 
        mid_ch = out_ch if not mid_ch else mid_ch
        self.net = nn.Sequential(nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(mid_ch),
                                 nn.ReLU(inplace=True),

                                 nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(out_ch),
                                 nn.ReLU(inplace=True))


    def forward(self, x):
        x = x 
        return self.net(x)

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__() 
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch, in_ch//2)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch//2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        ypad = x2.shape[2] - x1.shape[2]
        xpad = x2.shape[3] - x1.shape[3]
        x1 = F.pad(x1, [xpad//2, xpad-xpad//2, ypad//2, ypad-ypad//2]) 

        x = torch.cat([x2, x1], dim =1)
        return self.conv(x)

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__() 
        self.net = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2),
                                 DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.net(x)

class Encoder(nn.Module):
    def __init__(self, in_ch=1, bilinear=True):
        super().__init__() 
        self.ch = DoubleConv(in_ch, 64)
        self.down1 = DownBlock(64, 128)
        self.down2 = DownBlock(128, 256)
        self.down3 = DownBlock(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = DownBlock(512, 1024//factor)

    def forward(self, x):
        x0 = self.ch(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4]

class Decoder(nn.Module): 
    def __init__(self, out_ch=1, bilinear=True): 
        super().__init__() 
        factor = 2 if bilinear else 1 
        self.up1 = UpBlock(1024, 512//factor, bilinear)
        self.up2 = UpBlock(512, 256//factor, bilinear)
        self.up3 = UpBlock(256, 128//factor, bilinear)
        self.up4 = UpBlock(128, 64, bilinear)
        self.out = nn.Sequential(nn.Conv2d(64, out_ch, kernel_size=1, stride=1),
                                 nn.Sigmoid())

    def forward(self, x0, x1, x2, x3, x):

        x = self.up1(x, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)

        return self.out(x)



class Discriminator(nn.Module):
    def __init__(self, num_filters=256):
        super().__init__() 

        self.down0 = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1), 
                                   nn.ReLU(), 
                                   nn.MaxPool2d(2, 2), 
                                   nn.Conv2d(128, num_filters, 3, 1, 1), 
                                   nn.MaxPool2d(2,2),
                                   nn.ReLU(), 
                                   nn.Conv2d(num_filters, num_filters, 3, 1, 1),
                                   nn.MaxPool2d(2,2),
                                   nn.ReLU(),
                                   nn.Conv2d(num_filters, num_filters//2, 3, 1, 1),
                                   nn.MaxPool2d(2,2),
                                   nn.ReLU(),)




        self.down1 = nn.Sequential(nn.Conv2d(128, num_filters, 3, 1, 1), 
                                   nn.MaxPool2d(2,2),
                                   nn.ReLU(), 
                                   nn.Conv2d(num_filters, num_filters, 3, 1, 1),
                                   nn.MaxPool2d(2,2),
                                   nn.ReLU(),
                                   nn.Conv2d(num_filters, num_filters//2, 3, 1, 1),
                                   nn.MaxPool2d(2,2),
                                   nn.ReLU(),)

        self.down2 = nn.Sequential(nn.Conv2d(256, num_filters, 3, 1, 1), 
                                   nn.MaxPool2d(2,2),
                                   nn.ReLU(), 
                                   nn.Conv2d(num_filters, num_filters//2, 3, 1, 1),
                                   nn.MaxPool2d(2,2),
                                   nn.ReLU(),)

        self.down3 = nn.Sequential(nn.Conv2d(512, num_filters//2, 3, 1, 1), 
                                   nn.MaxPool2d(2,2),
                                   nn.ReLU(),)

        self.down4 = nn.Sequential(nn.Conv2d(512, num_filters//2, 3, 1, 1), 
                                   nn.ReLU(),)

        self.disc= nn.Sequential(nn.Conv2d(int(2.5 * num_filters), 2*num_filters, 3, 1, 1), 
                                 nn.MaxPool2d(2,2),
                                 nn.ReLU(),
                                 nn.Conv2d(2*num_filters, 64, 3, 1, 1),
                                 nn.MaxPool2d(2,2),
                                 nn.ReLU(),
                                 nn.Flatten(),
                                 nn.Linear(2240, 1024),
                                 nn.ReLU(),
                                 nn.Linear(1024, 128),
                                 nn.ReLU(),
                                 nn.Linear(128, 10),
                                 nn.ReLU(),
                                 nn.Linear(10, 1),
                                 )

    def forward(self, x0, x1, x2, x3, x4, detach=False):
        if detach:
            x0 = x0.detach()
            x1 = x1.detach() 
            x2 = x2.detach() 
            x3 = x3.detach() 
            x4 = x4.detach()
        x0 = self.down0(x0)
        x1 = self.down1(x1)
        x2 = self.down2(x2)
        x3 = self.down3(x3)
        x4 = self.down4(x4)
        x = torch.cat([x0, x1, x2, x3, x4], dim=1)
        x = self.disc(x)
        return x


if __name__ == '__main__': 
    torch.manual_seed(0)
    dev = 'cuda:1'
    img = Image.open('/home/shivac/qml-data/MEDVID0001_M_20210908_130347_0001_IMAGES/10/img.png')
    mask = Image.open('/home/shivac/qml-data/MEDVID0001_M_20210908_130347_0001_IMAGES/10/mask.png')
    import argparse 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--num_filters', type=int, default=256)
    args = parser.parse_args()
    img = ToTensor()(img).to(dev)
    img = img.unsqueeze(0)
    enc = Encoder(3).to(dev)
    dec = Decoder().to(dev)
    disc = Discriminator(args.num_filters).to(dev)
    logits = enc(img)
    disc_logits = disc(*logits)
    dec_logits = dec(*logits)
    print(f'{disc_logits.shape = }')
    print(f'{dec_logits.shape = }')
    print(f'{sum([p.numel() for p in disc.parameters()])/10**6 = }')
    print(f'{sum([p.numel() for p in dec.parameters()])/10**6 = }')
    print(f'{sum([p.numel() for p in enc.parameters()])/10**6 = }')


