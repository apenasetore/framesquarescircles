import torch 
import torch.nn as nn

def dice_loss(pred,target):
    numerator = 2 * torch.sum(pred * target)
    denominator = torch.sum(pred + target)
    return 1 - (numerator + 1) / (denominator + 1)

class ResidualConv2DBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + identity
        x = self.relu(x)
        return x

    
class UpsampleBlock2D(nn.Module):
    def __init__(self, in_c,out_c):
        super().__init__()
        self.upsamlpe = nn.Upsample(scale_factor=2)
        self.conv = nn.Conv2d(in_c,out_c,kernel_size=3,stride=1, padding=1)
        self.bn = nn.BatchNorm2d(num_features=out_c)
        self.relu = nn.ReLU()

    def forward(self,inputs):
        up = self.upsamlpe(inputs)
        x = self.conv(up)
        x = self.bn(x)
        x = self.relu(x)
        
        return x

class ChannelAttentionBlock(nn.Module):
    def __init__(self, depth, size):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(2 * depth, depth, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(depth, depth, kernel_size=1, padding=0)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.size = size

    def forward(self, low_input, high_input):
        
        x = torch.cat((low_input, high_input), dim=1) 
        x = self.global_pool(x)                        
        x = self.conv1(x)                                
        x = self.relu(x)
        x = self.conv2(x)                                
        x = self.sigmoid(x)
        x = x.repeat(1, 1, self.size, self.size)         

        out1 = low_input * x
        out2 = out1 + high_input

        return out2

class ResidualConv3DBlock(nn.Module):
    def __init__(self, in_c, out_c,kernel_size,stride,padding):
        super().__init__()
        self.conv1 = nn.Conv3d(in_c, out_c, kernel_size,stride,padding)
        self.bn1 = nn.BatchNorm3d(out_c)
        self.conv2 = nn.Conv3d(out_c, out_c, kernel_size,stride,padding)
        self.bn2 = nn.BatchNorm3d(out_c)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        identity = inputs
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + identity
        x = self.relu(x)

        return x

#----------------------------------------------------------
# The output layer of the decoder, i have to change it so the channel dimension is not 8 .
class Conv2DOutput(nn.Module):
    def __init__(self,  channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 1, kernel_size=(1,1), stride=(1,1),padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.sigmoid(x)

        return x

class FeatureFusionOperation(nn.Module):
    def __init__(self,  channels):
        super().__init__()
        self.skip_layer = nn.Conv3d(channels, channels, kernel_size=(4, 1, 1), stride=1)

    def forward(self, inputs, size):
        
        skipconv1 = self.skip_layer(inputs)

        B, C, F, W, H = skipconv1.shape
        x = skipconv1.view(B, C * F, W, H)
        #FAZ NADA ÇA MERDA?
        lower_input = nn.functional.interpolate(
            x, size=size, mode="bilinear", align_corners=False
        )
        return lower_input 
    

class ENCODERBLOCK(nn.Module):
    def __init__(self,in_c,n_filter):
        super().__init__()
        self.conv_layer_first = nn.Conv3d(in_c, n_filter, kernel_size=1, stride=1, padding=0)
        self.conv_layer = nn.Conv3d(in_c, n_filter, kernel_size=2, stride=(1, 2, 2), padding=0)
        self.convblock_layer = ResidualConv3DBlock(n_filter, n_filter, kernel_size=3, stride=1, padding=(1, 1, 1))
        self.ffo = FeatureFusionOperation(n_filter)
        self.dropout_layer = nn.Dropout3d(p=0.5)

    def forward(self, inputs, size, is_first=False, dropout=False):

        if dropout:
            inputs = self.dropout_layer(inputs)

        if is_first:
            conv1 = self.conv_layer_first(inputs)
        else:
            inputs = nn.functional.pad(inputs, (0, 1, 0, 1, 0, 1))
            conv1 = self.conv_layer(inputs)

        out = self.convblock_layer(conv1)
        lower_input = self.ffo(out,size)
        
        return out, lower_input



class DECODERBLOCK(nn.Module):
    def __init__(self,in_c,nfilter,depth,size):
        super().__init__()
        
        #============================================ DECODER ============================================# 
        
        #---------------------------------  Layer ---------------------------------#

        self.upsample = UpsampleBlock2D(in_c=in_c,out_c=nfilter)
        self.cab = ChannelAttentionBlock(depth,size)
        self.conv_block = ResidualConv2DBlock(nfilter)
        self.conv_output = Conv2DOutput(nfilter)

        #=================================================================================================#



    def forward(self, input, lower_input, flag=False):
        
        high_input = self.upsample(input)
        out_cab  = self.cab(lower_input,high_input)
        out_cont = self.conv_block(out_cab)
        
        if flag:
            out = self.conv_output(out_cont)
        else:
            out = out_cont
        
        
        return out




class SQUNET(nn.Module):
    def __init__(self,in_c,nfilter):
        super().__init__()
        
        #============================================ Encoder Layers ============================================# 
        self.encoderblopck1 = ENCODERBLOCK(in_c,nfilter)
        self.encoderblopck2 = ENCODERBLOCK(nfilter,2*nfilter)
        self.encoderblopck3 = ENCODERBLOCK(2*nfilter,4*nfilter)
        self.encoderblopck4 = ENCODERBLOCK(4*nfilter,8*nfilter)
        self.encoderblopck5 = ENCODERBLOCK(8*nfilter,16*nfilter)
        self.encoderblopck6 = ENCODERBLOCK(16*nfilter,32*nfilter)
        self.encoderblopck7 = ENCODERBLOCK(32*nfilter,64*nfilter)
        #========================================================================================================#

        #============================================ Decoder Layers ============================================# 

        self.decoderblopck1 = DECODERBLOCK(64*nfilter,32*nfilter,depth=256,size=16)
        self.decoderblopck2 = DECODERBLOCK(32*nfilter,16*nfilter,depth=128,size=32)
        self.decoderblopck3 = DECODERBLOCK(16*nfilter,8*nfilter,depth=64,size=64)
        self.decoderblopck4 = DECODERBLOCK(8*nfilter,4*nfilter,depth=32,size=128)
        self.decoderblopck5 = DECODERBLOCK(4*nfilter,2*nfilter,depth=16,size=256)
        self.decoderblopck6 = DECODERBLOCK(2*nfilter,nfilter,depth=8,size=512)
        #========================================================================================================#


        


    def forward(self, inputs):
        
        #============================================ Encoder ============================================#
        e_out1, lowerinput1 = self.encoderblopck1(inputs,is_first=True,size=(512,512))
        e_out2, lowerinput2 = self.encoderblopck2(e_out1,size=(256,256))
        e_out3, lowerinput3 = self.encoderblopck3(e_out2,size=(128,128))
        e_out4, lowerinput4 = self.encoderblopck4(e_out3,size=(64,64))
        e_out5, lowerinput5 = self.encoderblopck5(e_out4,size=(32,32))
        e_out6, lowerinput6 = self.encoderblopck6(e_out5,dropout=True,size=(16,16))
        _, e_out7 = self.encoderblopck7(e_out6,dropout=True,size=(8,8))
        #=================================================================================================#
        
        #============================================ Decoder ============================================#
        d_out1 = self.decoderblopck1(e_out7,lowerinput6)
        d_out2 = self.decoderblopck2(d_out1,lowerinput5)
        d_out3 = self.decoderblopck3(d_out2,lowerinput4)
        d_out4 = self.decoderblopck4(d_out3,lowerinput3)
        d_out5 = self.decoderblopck5(d_out4,lowerinput2)
        d_out6 = self.decoderblopck6(d_out5,lowerinput1,flag=True)

        #=================================================================================================#


        return d_out6
        #=================================================================================================#        
#----------------------------------------------------------
def model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement()*param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement()*buffer.element_size()

    size_all_mb = (param_size+buffer_size)/1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))    
    
#----------------------------------------------------------

def save_model(filename, model, optimizer,history, suffix=None):
	if suffix is not None: filename = filename.replace('.pth','_{0}.pth'.format(suffix))
	checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'history': history}
	torch.save(checkpoint, filename)

#----------------------------------------------------------

if __name__ == '__main__':

    input_data = torch.rand((1, 1, 4, 512, 512))
    model = SQUNET(in_c=1,nfilter=8)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
    model_size(model)
  

    output_data = model(input_data)
    print(output_data.shape)
