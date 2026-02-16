import torch 
import torch.nn as nn

def dice_loss(pred,target):
    numerator = 2 * torch.sum(pred * target)
    denominator = torch.sum(pred + target)
    return 1 - (numerator + 1) / (denominator + 1)

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.conv3 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_c)
        self.conv4 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        return x

class Encoder(nn.Module):
    def __init__(self, in_c, out_c, ds=(2,2)):
        super().__init__()
        self.conv = ConvBlock(in_c, out_c)
        self.pool = nn.MaxPool2d(ds)

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p

class Decoder(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = ConvBlock(out_c+out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x

class UNETDD(nn.Module):
    def __init__(self, out_c=1):
        super().__init__()
        #Parameters
        inch = 8
        nfilter = 16
        out_channels = out_c

        #Encoder
        self.e1 = Encoder(inch, nfilter, ds=(2,1))
        self.e2 = Encoder(nfilter, nfilter*2)
        self.e3 = Encoder(nfilter*2, nfilter*4)
        self.e4 = Encoder(nfilter*4, nfilter*8, ds=(4,2))
        self.e5 = Encoder(nfilter*8, nfilter*16, ds=(4,2))
        
        self.b = ConvBlock(nfilter*16, nfilter*32) 

        #Decoder1
        self.pool0 = nn.MaxPool2d(kernel_size=(2,1))
        self.d0 = Decoder(nfilter*32, nfilter*16)
        self.pool1 = nn.MaxPool2d(kernel_size=(4,1))
        self.d1 = Decoder(nfilter*16, nfilter*8)
        self.pool2 = nn.MaxPool2d(kernel_size=(4,1))
        self.d2 = Decoder(nfilter*8, nfilter*4)
        self.pool3 = nn.MaxPool2d(kernel_size=(4,1))
        self.d3 = Decoder(nfilter*4, nfilter*2)
        self.pool4 = nn.MaxPool2d(kernel_size=(4,1))
        self.up4 = nn.Upsample(scale_factor=(1, 2),mode='bicubic')
        self.d4 = Decoder(nfilter*2, nfilter)
        
        self.output1 = nn.Conv2d(nfilter, out_channels, kernel_size=1, padding=0) #1x1 convolution
        self.sigmoid = nn.Sigmoid()

        #Decoder2
        self.dpool0 = nn.MaxPool2d(kernel_size=(2,1))
        self.dd0 = Decoder(nfilter*32, nfilter*16)
        self.dpool1 = nn.MaxPool2d(kernel_size=(4,1))
        self.dd1 = Decoder(nfilter*16, nfilter*8)
        self.dpool2 = nn.MaxPool2d(kernel_size=(4,1))
        self.dd2 = Decoder(nfilter*8, nfilter*4)
        self.dpool3 = nn.MaxPool2d(kernel_size=(4,1))
        self.dd3 = Decoder(nfilter*4, nfilter*2)
        self.dpool4 = nn.MaxPool2d(kernel_size=(4,1))
        self.dup4 = nn.Upsample(scale_factor=(1, 2),mode='bicubic')
        self.dd4 = Decoder(nfilter*2, nfilter)
        
        self.output2 = nn.Conv2d(nfilter, out_channels, kernel_size=1, padding=0) #1x1 convolution
        self.relu = nn.ReLU()
        

    def forward(self, inputs):
        #Encoder
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        s5, p5 = self.e5(p4)

        b = self.b(p5)

        #Decoder1     
        s5up = self.pool0(s5)  
        d0 = self.d0(b, s5up)
        s4up = self.pool1(s4)
        d1 = self.d1(d0, s4up)
        s3up = self.pool2(s3)
        d2 = self.d2(d1, s3up)
        s2up = self.pool3(s2)
        d3 = self.d3(d2, s2up)
        s1up = self.pool4(s1)
        s1up = self.up4(s1up)
        d4 = self.d4(d3, s1up)
        
        outputs1 = self.output1(d4)
        outputs1 = self.sigmoid(outputs1)

        #Decoder2
        s5up2 = self.dpool0(s5)  
        dd0 = self.dd0(b, s5up2)
        s4up2 = self.dpool1(s4)
        dd1 = self.dd1(dd0, s4up2)
        s3up2 = self.dpool2(s3)
        dd2 = self.dd2(dd1, s3up2)
        s2up2 = self.dpool3(s2)
        dd3 = self.dd3(dd2, s2up2)
        s1up2 = self.dpool4(s1)
        s1up2 = self.dup4(s1up2)
        dd4 = self.dd4(dd3, s1up2)

        outputs2 = self.output2(dd4)
        outputs2 = self.relu(outputs2)
        
        return outputs1, outputs2

#----------------------------------------------------------

def save_model(filename, model, optimizer,history, suffix=None):
	if suffix is not None: filename = filename.replace('.pth','_{0}.pth'.format(suffix))
	checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'history': history}
	torch.save(checkpoint, filename)

#----------------------------------------------------------

if __name__ == '__main__':
    input_image = torch.rand((1, 8, 1024, 128))
    model = UNETDD()
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
    output1, output2 = model(input_image)
    print(output1.shape)
    print(output2.shape)
