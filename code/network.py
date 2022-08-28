import torch
import torch.nn as nn
import torch.nn.functional as F


class generator(nn.Module):
    # initializers
    def __init__(self, d=64, batch_size=1):
        super(generator, self).__init__()


        bn = None
        if batch_size == 1:
          bn = False
        else:
          bn = True
          
        
        # Unet encoder
        #######################
        ### Encoding Layers 1-4
        #######################
        self.conv1_1 = nn.Conv2d(3, d, 4, 2, 1) # [3x256x256] -> [64x128x128]
        self.conv2_1 = nn.Conv2d(d, d * 2, 4, 2, 1) # [64x128x128] -> [128x64x64]
        self.conv2_bn_1 = nn.BatchNorm2d(d * 2) if bn else nn.InstanceNorm2d(d * 2)
        self.conv3_1 = nn.Conv2d(d * 2, d * 4, 4, 2, 1) # [128x64x64] -> [256x32x32]
        self.conv3_bn_1 = nn.BatchNorm2d(d * 4) if bn else nn.InstanceNorm2d(d * 4)
        self.conv4_1 = nn.Conv2d(d * 4, d * 8, 4, 2, 1) # [256x32x32] -> [512x16x16]
        self.conv4_bn_1 = nn.BatchNorm2d(d * 8) if bn else nn.InstanceNorm2d(d * 8)

        
        ##########################
        ### Encoding Layers 5-8, 1
        ##########################
        self.conv5_1 = nn.Conv2d(d * 8, d * 8, 4, 2, 1) # [512x16x16] -> [512x8x8]
        self.conv5_bn_1 = nn.BatchNorm2d(d * 8) if bn else nn.InstanceNorm2d(d * 8)
        self.conv6_1 = nn.Conv2d(d * 8, d * 8, 4, 2, 1) # [512x8x8] -> [512x4x4]
        self.conv6_bn_1 = nn.BatchNorm2d(d * 8) if bn else nn.InstanceNorm2d(d * 8)
        self.conv7_1 = nn.Conv2d(d * 8, d * 8, 4, 2, 1) # [512x4x4] -> [512x2x2]
        self.conv7_bn_1 = nn.BatchNorm2d(d * 8) if bn else nn.InstanceNorm2d(d * 8)
        self.conv8_1 = nn.Conv2d(d * 8, d * 8, 4, 2, 1) # [512x2x2] -> [512x1x1]
        #self.conv8_bn_1 = nn.BatchNorm2d(d * 8) if bn else nn.InstanceNorm2d(d * 8)

        
        ##########################
        ### Encoding Layers 5-8, 2
        ##########################
        self.conv5_2 = nn.Conv2d(d * 8, d * 8, 4, 2, 1) # [512x16x16] -> [512x8x8]
        self.conv5_bn_2 = nn.BatchNorm2d(d * 8) if bn else nn.InstanceNorm2d(d * 8)
        self.conv6_2 = nn.Conv2d(d * 8, d * 8, 4, 2, 1) # [512x8x8] -> [512x4x4]
        self.conv6_bn_2 = nn.BatchNorm2d(d * 8) if bn else nn.InstanceNorm2d(d * 8)
        self.conv7_2 = nn.Conv2d(d * 8, d * 8, 4, 2, 1) # [512x4x4] -> [512x2x2]
        self.conv7_bn_2 = nn.BatchNorm2d(d * 8) if bn else nn.InstanceNorm2d(d * 8)
        self.conv8_2 = nn.Conv2d(d * 8, d * 8, 4, 2, 1) # [512x2x2] -> [512x1x1]
        #self.conv8_bn_2 = nn.BatchNorm2d(d * 8) if bn else nn.InstanceNorm2d(d * 8)
        
        ##########################
        ### Encoding Layers 5-8, 3
        ##########################
        self.conv5_3 = nn.Conv2d(d * 8, d * 8, 4, 2, 1) # [512x16x16] -> [512x8x8]
        self.conv5_bn_3 = nn.BatchNorm2d(d * 8) if bn else nn.InstanceNorm2d(d * 8)
        self.conv6_3 = nn.Conv2d(d * 8, d * 8, 4, 2, 1) # [512x8x8] -> [512x4x4]
        self.conv6_bn_3 = nn.BatchNorm2d(d * 8) if bn else nn.InstanceNorm2d(d * 8)
        self.conv7_3 = nn.Conv2d(d * 8, d * 8, 4, 2, 1) # [512x4x4] -> [512x2x2]
        self.conv7_bn_3 = nn.BatchNorm2d(d * 8) if bn else nn.InstanceNorm2d(d * 8)
        self.conv8_3 = nn.Conv2d(d * 8, d * 8, 4, 2, 1) # [512x2x2] -> [512x1x1]
        #self.conv8_bn_3 = nn.BatchNorm2d(d * 8) if bn else nn.InstanceNorm2d(d * 8)
        
        ##########################
        ### Encoding Layers 5-8, 4
        ##########################
        self.conv5_4 = nn.Conv2d(d * 8, d * 8, 4, 2, 1) # [512x16x16] -> [512x8x8]
        self.conv5_bn_4 = nn.BatchNorm2d(d * 8) if bn else nn.InstanceNorm2d(d * 8)
        self.conv6_4 = nn.Conv2d(d * 8, d * 8, 4, 2, 1) # [512x8x8] -> [512x4x4]
        self.conv6_bn_4 = nn.BatchNorm2d(d * 8) if bn else nn.InstanceNorm2d(d * 8)
        self.conv7_4 = nn.Conv2d(d * 8, d * 8, 4, 2, 1) # [512x4x4] -> [512x2x2]
        self.conv7_bn_4 = nn.BatchNorm2d(d * 8) if bn else nn.InstanceNorm2d(d * 8)
        self.conv8_4 = nn.Conv2d(d * 8, d * 8, 4, 2, 1) # [512x2x2] -> [512x1x1]
        #self.conv8_bn_4 = nn.BatchNorm2d(d * 8) if bn else nn.InstanceNorm2d(d * 8)
        
        # Unet decoder
        #############
        ### Decoder 1
        #############
        self.deconv1_1 = nn.ConvTranspose2d(d * 8, d * 8, 4, 2, 1) # [512x1x1] -> [512x2x2]
        self.deconv1_bn_1 = nn.BatchNorm2d(d * 8) if bn else nn.InstanceNorm2d(d * 8)
        self.deconv2_1 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1) # [(512+512)x2x2] -> [512x4x4]
        self.deconv2_bn_1 = nn.BatchNorm2d(d * 8) if bn else nn.InstanceNorm2d(d * 8)
        self.deconv3_1 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1) # [(512+512)x4x4] -> [512x8x8]
        self.deconv3_bn_1 = nn.BatchNorm2d(d * 8) if bn else nn.InstanceNorm2d(d * 8)
        self.deconv4_1 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1) # [(512+512)x8x8] -> [512x16x16]
        self.deconv4_bn_1 = nn.BatchNorm2d(d * 8) if bn else nn.InstanceNorm2d(d * 8)
        self.deconv5_1 = nn.ConvTranspose2d(d * 8 * 2, d * 4, 4, 2, 1) # [(512+512)x16x16] -> [256x32x32]
        self.deconv5_bn_1 = nn.BatchNorm2d(d * 4) if bn else nn.InstanceNorm2d(d * 4)          
        self.deconv6_1 = nn.ConvTranspose2d(d * 4 * 2, d * 2, 4, 2, 1) # [(256+256)x32x32] -> [128x64x64]
        self.deconv6_bn_1 = nn.BatchNorm2d(d * 2) if bn else nn.InstanceNorm2d(d * 2)
        self.deconv7_1 = nn.ConvTranspose2d(d * 2 * 2, d, 4, 2, 1) # [(128+128)x64x64] -> [64x128x128]
        self.deconv7_bn_1 = nn.BatchNorm2d(d) if bn else nn.InstanceNorm2d(d)
        self.deconv8_1 = nn.ConvTranspose2d(d * 2, 1, 4, 2, 1) # [(64+64)x128x128] -> [1x256x256]
        self.deconv9_1 = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=1) # [4x256x256] -> [1x256x256]
        
        #############
        ### Decoder 2
        #############
        self.deconv1_2 = nn.ConvTranspose2d(d * 8, d * 8, 4, 2, 1)
        self.deconv1_bn_2 = nn.BatchNorm2d(d * 8) if bn else nn.InstanceNorm2d(d * 8)
        self.deconv2_2 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1)
        self.deconv2_bn_2 = nn.BatchNorm2d(d * 8) if bn else nn.InstanceNorm2d(d * 8)
        self.deconv3_2 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1)
        self.deconv3_bn_2 = nn.BatchNorm2d(d * 8) if bn else nn.InstanceNorm2d(d * 8)
        self.deconv4_2 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1)
        self.deconv4_bn_2 = nn.BatchNorm2d(d * 8) if bn else nn.InstanceNorm2d(d * 8)
        self.deconv5_2 = nn.ConvTranspose2d(d * 8 * 2, d * 4, 4, 2, 1)
        self.deconv5_bn_2 = nn.BatchNorm2d(d * 4) if bn else nn.InstanceNorm2d(d * 4)
        self.deconv6_2 = nn.ConvTranspose2d(d * 4 * 2, d * 2, 4, 2, 1)
        self.deconv6_bn_2 = nn.BatchNorm2d(d * 2) if bn else nn.InstanceNorm2d(d * 2)
        self.deconv7_2 = nn.ConvTranspose2d(d * 2 * 2, d, 4, 2, 1)
        self.deconv7_bn_2 = nn.BatchNorm2d(d) if bn else nn.InstanceNorm2d(d)
        self.deconv8_2 = nn.ConvTranspose2d(d * 2, 1, 4, 2, 1)
        self.deconv9_2 = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=1)
        
        #############
        ### Decoder 3
        #############
        self.deconv1_3 = nn.ConvTranspose2d(d * 8, d * 8, 4, 2, 1)
        self.deconv1_bn_3 = nn.BatchNorm2d(d * 8) if bn else nn.InstanceNorm2d(d * 8)
        self.deconv2_3 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1)
        self.deconv2_bn_3 = nn.BatchNorm2d(d * 8) if bn else nn.InstanceNorm2d(d * 8)
        self.deconv3_3 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1)
        self.deconv3_bn_3 = nn.BatchNorm2d(d * 8) if bn else nn.InstanceNorm2d(d * 8)
        self.deconv4_3 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1)
        self.deconv4_bn_3 = nn.BatchNorm2d(d * 8) if bn else nn.InstanceNorm2d(d * 8)
        self.deconv5_3 = nn.ConvTranspose2d(d * 8 * 2, d * 4, 4, 2, 1)
        self.deconv5_bn_3 = nn.BatchNorm2d(d * 4) if bn else nn.InstanceNorm2d(d * 4)
        self.deconv6_3 = nn.ConvTranspose2d(d * 4 * 2, d * 2, 4, 2, 1)
        self.deconv6_bn_3 = nn.BatchNorm2d(d * 2) if bn else nn.InstanceNorm2d(d * 2)
        self.deconv7_3 = nn.ConvTranspose2d(d * 2 * 2, d, 4, 2, 1)
        self.deconv7_bn_3 = nn.BatchNorm2d(d) if bn else nn.InstanceNorm2d(d)
        self.deconv8_3 = nn.ConvTranspose2d(d * 2, 1, 4, 2, 1)
        self.deconv9_3 = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=1)
        
        #############
        ### Decoder 4
        #############
        self.deconv1_4 = nn.ConvTranspose2d(d * 8, d * 8, 4, 2, 1)
        self.deconv1_bn_4 = nn.BatchNorm2d(d * 8) if bn else nn.InstanceNorm2d(d * 8)
        self.deconv2_4 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1)
        self.deconv2_bn_4 = nn.BatchNorm2d(d * 8) if bn else nn.InstanceNorm2d(d * 8)
        self.deconv3_4 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1)
        self.deconv3_bn_4 = nn.BatchNorm2d(d * 8) if bn else nn.InstanceNorm2d(d * 8)
        self.deconv4_4 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1)
        self.deconv4_bn_4 = nn.BatchNorm2d(d * 8) if bn else nn.InstanceNorm2d(d * 8)
        self.deconv5_4 = nn.ConvTranspose2d(d * 8 * 2, d * 4, 4, 2, 1)
        self.deconv5_bn_4 = nn.BatchNorm2d(d * 4) if bn else nn.InstanceNorm2d(d * 4)
        self.deconv6_4 = nn.ConvTranspose2d(d * 4 * 2, d * 2, 4, 2, 1)
        self.deconv6_bn_4 = nn.BatchNorm2d(d * 2) if bn else nn.InstanceNorm2d(d * 2)
        self.deconv7_4 = nn.ConvTranspose2d(d * 2 * 2, d, 4, 2, 1)
        self.deconv7_bn_4 = nn.BatchNorm2d(d) if bn else nn.InstanceNorm2d(d)
        self.deconv8_4 = nn.ConvTranspose2d(d * 2, 1, 4, 2, 1)
        self.deconv9_4 = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
                    
          if self._modules[m].weight.requires_grad or self._modules[m].bias.requires_grad:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
    
        e1_1 = self.conv1_1(input)        
        e2_1 = self.conv2_bn_1(self.conv2_1(F.leaky_relu(e1_1, 0.2)))
        e3_1 = self.conv3_bn_1(self.conv3_1(F.leaky_relu(e2_1, 0.2)))
        e4_1 = self.conv4_bn_1(self.conv4_1(F.leaky_relu(e3_1, 0.2)))
                
        e5_1 = self.conv5_bn_1(self.conv5_1(F.leaky_relu(e4_1, 0.2)))
        e6_1 = self.conv6_bn_1(self.conv6_1(F.leaky_relu(e5_1, 0.2)))
        e7_1 = self.conv7_bn_1(self.conv7_1(F.leaky_relu(e6_1, 0.2)))
        e8_1 = self.conv8_1(F.leaky_relu(e7_1, 0.2))
        
        e5_2 = self.conv5_bn_2(self.conv5_2(F.leaky_relu(e4_1, 0.2)))
        e6_2 = self.conv6_bn_2(self.conv6_2(F.leaky_relu(e5_2, 0.2)))
        e7_2 = self.conv7_bn_2(self.conv7_2(F.leaky_relu(e6_2, 0.2)))
        e8_2 = self.conv8_2(F.leaky_relu(e7_2, 0.2))
        
        e5_3 = self.conv5_bn_3(self.conv5_3(F.leaky_relu(e4_1, 0.2)))
        e6_3 = self.conv6_bn_3(self.conv6_3(F.leaky_relu(e5_3, 0.2)))
        e7_3 = self.conv7_bn_3(self.conv7_3(F.leaky_relu(e6_3, 0.2)))
        e8_3 = self.conv8_3(F.leaky_relu(e7_3, 0.2))
        
        e5_4 = self.conv5_bn_4(self.conv5_4(F.leaky_relu(e4_1, 0.2)))
        e6_4 = self.conv6_bn_4(self.conv6_4(F.leaky_relu(e5_4, 0.2)))
        e7_4 = self.conv7_bn_4(self.conv7_4(F.leaky_relu(e6_4, 0.2)))
        e8_4 = self.conv8_4(F.leaky_relu(e7_4, 0.2))
                
        # decoder 1
        d1_1 = F.dropout(self.deconv1_bn_1(self.deconv1_1(F.relu(e8_1))), 0.5, training=True)
        d1_1 = torch.cat([d1_1, e7_1], 1)
        d2_1 = F.dropout(self.deconv2_bn_1(self.deconv2_1(F.relu(d1_1))), 0.5, training=True)
        d2_1 = torch.cat([d2_1, e6_1], 1)
        d3_1 = F.dropout(self.deconv3_bn_1(self.deconv3_1(F.relu(d2_1))), 0.5, training=True)
        d3_1 = torch.cat([d3_1, e5_1], 1)
        d4_1 = self.deconv4_bn_1(self.deconv4_1(F.relu(d3_1)))
        d4_1 = torch.cat([d4_1, e4_1], 1)
        d5_1 = self.deconv5_bn_1(self.deconv5_1(F.relu(d4_1)))
        d5_1 = torch.cat([d5_1, e3_1], 1)
        d6_1 = self.deconv6_bn_1(self.deconv6_1(F.relu(d5_1)))
        d6_1 = torch.cat([d6_1, e2_1], 1)
        d7_1 = self.deconv7_bn_1(self.deconv7_1(F.relu(d6_1)))
        d7_1 = torch.cat([d7_1, e1_1], 1)
        d8_1 = self.deconv8_1(F.relu(d7_1))
        o_1 = torch.tanh(d8_1)
        
        o_1_pitl = torch.cat([o_1, input], 1)
        o_1_final = self.deconv9_1(o_1_pitl)
        o_1_final_img = torch.tanh(o_1_final)
        
        
        # decoder 2
        d1_2 = F.dropout(self.deconv1_bn_2(self.deconv1_2(F.relu(e8_2))), 0.5, training=True)
        d1_2 = torch.cat([d1_2, e7_2], 1)
        d2_2 = F.dropout(self.deconv2_bn_2(self.deconv2_2(F.relu(d1_2))), 0.5, training=True)
        d2_2 = torch.cat([d2_2, e6_2], 1)
        d3_2 = F.dropout(self.deconv3_bn_2(self.deconv3_2(F.relu(d2_2))), 0.5, training=True)
        d3_2 = torch.cat([d3_2, e5_2], 1)
        d4_2 = self.deconv4_bn_2(self.deconv4_2(F.relu(d3_2)))
        d4_2 = torch.cat([d4_2, e4_1], 1)
        d5_2 = self.deconv5_bn_2(self.deconv5_2(F.relu(d4_2)))
        d5_2 = torch.cat([d5_2, e3_1], 1)
        d6_2 = self.deconv6_bn_2(self.deconv6_2(F.relu(d5_2)))
        d6_2 = torch.cat([d6_2, e2_1], 1)
        d7_2 = self.deconv7_bn_2(self.deconv7_2(F.relu(d6_2)))
        d7_2 = torch.cat([d7_2, e1_1], 1)
        d8_2 = self.deconv8_2(F.relu(d7_2))
        o_2 = torch.tanh(d8_2)
        
        o_2_pitl = torch.cat([o_2, input], 1)
        o_2_final = self.deconv9_2(o_2_pitl)
        o_2_final_img = torch.tanh(o_2_final)
        
        # decoder 3
        d1_3 = F.dropout(self.deconv1_bn_3(self.deconv1_1(F.relu(e8_3))), 0.5, training=True)
        d1_3 = torch.cat([d1_3, e7_3], 1)
        d2_3 = F.dropout(self.deconv2_bn_3(self.deconv2_1(F.relu(d1_3))), 0.5, training=True)
        d2_3 = torch.cat([d2_3, e6_3], 1)
        d3_3 = F.dropout(self.deconv3_bn_3(self.deconv3_1(F.relu(d2_3))), 0.5, training=True)
        d3_3 = torch.cat([d3_3, e5_3], 1)
        d4_3 = self.deconv4_bn_3(self.deconv4_3(F.relu(d3_3)))
        d4_3 = torch.cat([d4_3, e4_1], 1)
        d5_3 = self.deconv5_bn_3(self.deconv5_3(F.relu(d4_3)))
        d5_3 = torch.cat([d5_3, e3_1], 1)
        d6_3 = self.deconv6_bn_3(self.deconv6_3(F.relu(d5_3)))
        d6_3 = torch.cat([d6_3, e2_1], 1)
        d7_3 = self.deconv7_bn_3(self.deconv7_3(F.relu(d6_3)))
        d7_3 = torch.cat([d7_3, e1_1], 1)
        d8_3 = self.deconv8_3(F.relu(d7_3))
        o_3 = torch.tanh(d8_3)
        
        o_3_pitl = torch.cat([o_3, input], 1)
        o_3_final = self.deconv9_3(o_3_pitl)
        o_3_final_img = torch.tanh(o_3_final)
        
        # decoder 4
        d1_4 = F.dropout(self.deconv1_bn_4(self.deconv1_4(F.relu(e8_4))), 0.5, training=True)
        d1_4 = torch.cat([d1_4, e7_4], 1)
        d2_4 = F.dropout(self.deconv2_bn_4(self.deconv2_4(F.relu(d1_4))), 0.5, training=True)
        d2_4 = torch.cat([d2_4, e6_4], 1)
        d3_4 = F.dropout(self.deconv3_bn_4(self.deconv3_4(F.relu(d2_4))), 0.5, training=True)
        d3_4 = torch.cat([d3_4, e5_4], 1)
        d4_4 = self.deconv4_bn_4(self.deconv4_4(F.relu(d3_4)))
        d4_4 = torch.cat([d4_4, e4_1], 1)
        d5_4 = self.deconv5_bn_4(self.deconv5_4(F.relu(d4_4)))
        d5_4 = torch.cat([d5_4, e3_1], 1)
        d6_4 = self.deconv6_bn_4(self.deconv6_4(F.relu(d5_4)))
        d6_4 = torch.cat([d6_4, e2_1], 1)
        d7_4 = self.deconv7_bn_4(self.deconv7_4(F.relu(d6_4)))
        d7_4 = torch.cat([d7_4, e1_1], 1)
        d8_4 = self.deconv8_4(F.relu(d7_4))
        o_4 = torch.tanh(d8_4)
        
        o_4_pitl = torch.cat([o_4, input], 1)
        o_4_final = self.deconv9_4(o_4_pitl)
        o_4_final_img = torch.tanh(o_4_final)
                
        
        return torch.cat((o_1_final_img, o_2_final_img, o_3_final_img, o_4_final_img), dim=3)
        #return torch.cat((o_1, o_2, o_3, o_4), dim=3) #use if removing PITL
        

class discriminator(nn.Module):
    # initializers
    def __init__(self, d=64, batch_size=1):
        super(discriminator, self).__init__()
        
        bn = None
        if batch_size == 1:
          bn = False
        else:
          bn = True
          
        self.conv1_1 = nn.Conv2d(4, d, 4, 2, 1) # [(3+1)x256x256] -> [64x128x128] 
        self.conv2_1 = nn.Conv2d(d, d * 2, 4, 2, 1) # [64x128x128] -> [128x64x64]
        self.conv2_bn_1 = nn.BatchNorm2d(d * 2) if bn else nn.InstanceNorm2d(d * 2)
        self.conv3_1 = nn.Conv2d(d * 2, d * 4, 4, 2, 1) # [128x64x64] -> [256x32x32]
        self.conv3_bn_1 = nn.BatchNorm2d(d * 4) if bn else nn.InstanceNorm2d(d * 4)
        self.conv4_1 = nn.Conv2d(d * 4, d * 8, 4, 1, 1) # [256x32x32] -> [512x31x31] 
        self.conv4_bn_1 = nn.BatchNorm2d(d * 8) if bn else nn.InstanceNorm2d(d * 8)
        self.conv5_1 = nn.Conv2d(d * 8, 1, 4, 1, 1) # [512x31x31] -> [1x30x30] (Fully Convolutional, PatchGAN)
        
        self.conv1_2 = nn.Conv2d(4, d, 4, 2, 1)
        self.conv2_2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv2_bn_2 = nn.BatchNorm2d(d * 2) if bn else nn.InstanceNorm2d(d * 2)
        self.conv3_2 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn_2 = nn.BatchNorm2d(d * 4) if bn else nn.InstanceNorm2d(d * 4)
        self.conv4_2 = nn.Conv2d(d * 4, d * 8, 4, 1, 1)
        self.conv4_bn_2 = nn.BatchNorm2d(d * 8) if bn else nn.InstanceNorm2d(d * 8)
        self.conv5_2 = nn.Conv2d(d * 8, 1, 4, 1, 1)
        
        self.conv1_3 = nn.Conv2d(4, d, 4, 2, 1)
        self.conv2_3 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv2_bn_3 = nn.BatchNorm2d(d * 2) if bn else nn.InstanceNorm2d(d * 2)
        self.conv3_3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn_3 = nn.BatchNorm2d(d * 4) if bn else nn.InstanceNorm2d(d * 4)
        self.conv4_3 = nn.Conv2d(d * 4, d * 8, 4, 1, 1)
        self.conv4_bn_3 = nn.BatchNorm2d(d * 8) if bn else nn.InstanceNorm2d(d * 8)
        self.conv5_3 = nn.Conv2d(d * 8, 1, 4, 1, 1)
        
        self.conv1_4 = nn.Conv2d(4, d, 4, 2, 1)
        self.conv2_4 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv2_bn_4 = nn.BatchNorm2d(d * 2) if bn else nn.InstanceNorm2d(d * 2)
        self.conv3_4 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn_4 = nn.BatchNorm2d(d * 4) if bn else nn.InstanceNorm2d(d * 4)
        self.conv4_4 = nn.Conv2d(d * 4, d * 8, 4, 1, 1)
        self.conv4_bn_4 = nn.BatchNorm2d(d * 8) if bn else nn.InstanceNorm2d(d * 8)
        self.conv5_4 = nn.Conv2d(d * 8, 1, 4, 1, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label):

        label_0 = label[:, :, :, 0:256] # [1, 1, 256, 256]
        label_1 = label[:, :, :, 256:512]
        label_2 = label[:, :, :, 512:768]
        label_3 = label[:, :, :, 768:1024]
        
        x_1 = torch.cat([input, label_0], 1) # ([1, 3, 256, 256], [1, 1, 256, 256])  -> [1, 4, 256, 256]
        x_1 = F.leaky_relu(self.conv1_1(x_1), 0.2)
        x_1 = F.leaky_relu(self.conv2_bn_1(self.conv2_1(x_1)), 0.2)
        x_1 = F.leaky_relu(self.conv3_bn_1(self.conv3_1(x_1)), 0.2)
        x_1 = F.leaky_relu(self.conv4_bn_1(self.conv4_1(x_1)), 0.2)
        x_1 = torch.sigmoid(self.conv5_1(x_1))
        
        x_2 = torch.cat([input, label_1], 1)
        x_2 = F.leaky_relu(self.conv1_2(x_2), 0.2)
        x_2 = F.leaky_relu(self.conv2_bn_2(self.conv2_2(x_2)), 0.2)
        x_2 = F.leaky_relu(self.conv3_bn_2(self.conv3_2(x_2)), 0.2)
        x_2 = F.leaky_relu(self.conv4_bn_2(self.conv4_2(x_2)), 0.2)
        x_2 = torch.sigmoid(self.conv5_2(x_2))

        x_3 = torch.cat([input, label_2], 1)
        x_3 = F.leaky_relu(self.conv1_3(x_3), 0.2)
        x_3 = F.leaky_relu(self.conv2_bn_3(self.conv2_3(x_3)), 0.2)
        x_3 = F.leaky_relu(self.conv3_bn_3(self.conv3_3(x_3)), 0.2)
        x_3 = F.leaky_relu(self.conv4_bn_3(self.conv4_3(x_3)), 0.2)
        x_3 = torch.sigmoid(self.conv5_3(x_3))
        
        x_4 = torch.cat([input, label_3], 1)
        x_4 = F.leaky_relu(self.conv1_4(x_4), 0.2)
        x_4 = F.leaky_relu(self.conv2_bn_4(self.conv2_4(x_4)), 0.2)
        x_4 = F.leaky_relu(self.conv3_bn_4(self.conv3_4(x_4)), 0.2)
        x_4 = F.leaky_relu(self.conv4_bn_4(self.conv4_4(x_4)), 0.2)
        x_4 = torch.sigmoid(self.conv5_4(x_4))

        return torch.cat((x_1, x_2, x_3, x_4), 1)
        
def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()