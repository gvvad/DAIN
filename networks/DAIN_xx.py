import torch
import torch.nn as nn
from my_package.FilterInterpolation import FilterInterpolationModule
from my_package.FlowProjection import FlowProjectionModule #,FlowFillholeModule
from my_package.DepthFlowProjection import DepthFlowProjectionModule

from torch.nn.functional import interpolate

from Stack import Stack
import time
import PWCNet
import S2D_models
import Resblock
import MegaDepth
# import time
import numpy as np
# import imageio

class TimeStat:
    def checkpoint(self):
        self.time_point = time.time()
    def getDelta(self) -> float:
        return time.time() - self.time_point
    def getDeltaInSeconds(self):
        return self.getDelta()

class DAIN(torch.nn.Module):
    def __init__(self, frames=1, stat=False):
        # base class initialization
        super(DAIN, self).__init__()
        
        self.device = torch.device("cuda")

        self.height = self.width = None
        self.padding = None
        self.padder = None
        self.frame_a = None
        
        self.stat = stat
        if self.stat:
            self.timestat = TimeStat()

        parts = frames + 1
        delta = 1.0 / parts
        self.time_offsets = [(i+1)*delta for i in range(parts-1)]

        self.filter_size = 4

        self.initScaleNets_filter,self.initScaleNets_filter1,self.initScaleNets_filter2 = \
            self.get_MonoNet5(3, self.filter_size * self.filter_size, "filter")

        self.ctxNet = S2D_models.S2DF_3dense()
        self.ctx_ch = 3 * 64 + 3

        self.rectifyNet = Resblock.MultipleBasicBlock_4(3 + 3 + 3 + 2*1 + 2*2 + 16*2 + 2 * self.ctx_ch, 128)
        
        self.flownets = PWCNet.pwc_dc_net()
        
        self.div_flow = 20.0
        #extract depth information
        self.depthNet=MegaDepth.HourGlass()#.half()

    def statCheckpoint(self):
        if self.stat:
            self.timestat.checkpoint()
    def statPrintMessage(self, msg:str, ms=False):
        if self.stat:
            print(msg.format(self.timestat.getDelta() if ms==True else self.timestat.getDeltaInSeconds()))

    @staticmethod
    def getPaddings(value):
        if value != ((value >> 7) << 7):
            buf = (((value >> 7) + 1) << 7)
            a = int((buf - value)/2)
            b = buf - value - a
            return(a, b)
        else:
            return(32, 32)
    
    def getFrameFromRgb(self, frame_rgb) -> torch.tensor:
        if isinstance(frame_rgb, np.ndarray):
            return torch.from_numpy(
                np.transpose(frame_rgb, (2, 0, 1)).astype(np.float32) / 255.0
                ).type(torch.get_default_dtype()).to(self.device)

    def frameToRgb(self, frame:torch.tensor):
        frame.squeeze_().clip_(0.0, 1.0).mul_(255)
        t_res_1 = frame[:, self.padding[2]:self.padding[2]+self.height,
                           self.padding[0]:self.padding[0]+self.width].permute(1, 2, 0).type(torch.uint8)
        
        return np.array(t_res_1.cpu(), dtype=np.uint8)
        # return t_res_1

    @staticmethod
    def resizeFrame(frame, height, width) -> torch.Tensor:
        buf = frame.clone().squeeze_(0)
        buf = interpolate(buf, size=int(width), mode="nearest")
        buf = buf.permute(0, 2, 1)
        buf = interpolate(buf, size=int(height), mode="nearest")
        buf = buf.permute(0, 2, 1)
        buf.clamp_(0.0, 1.0)
        return buf

    def processRgb(self, frame_rgb):
        self.statCheckpoint()
        with torch.no_grad():
            if self.padder == None:
                self.height = frame_rgb.shape[0]
                self.width = frame_rgb.shape[1]
                self.padding = (self.getPaddings(self.width) + self.getPaddings(self.height))
                self.padder = torch.nn.ReplicationPad2d(self.padding)
            
            self.statPrintMessage("frame [padding begin]: {:.2f}")
            frame = self.padder(self.getFrameFromRgb(frame_rgb).unsqueeze(0))
            self.statPrintMessage("frame [padding done]: {:.2f}")
            
            if self.frame_a is not None:
                res_frame = self.forward(frame)
                self.statPrintMessage("frame [rgb begin]: {:.2f}")
                rgb = self.frameToRgb(res_frame)
                self.statPrintMessage("frame [rgb done]: {:.2f}")
                return rgb
            
            self.frame_a = frame
            del frame
            
            self.frame_a_depth = self.depthNet(self.frame_a)
            self.frame_a_depth_inv = (1e-6 + 1 / torch.exp(self.frame_a_depth))
            return None

    def forward(self, frame_b):
        self.statPrintMessage("frame [netbegin]: {:.2f}")
        frame_b_depth = self.depthNet(frame_b)
        
        self.statPrintMessage("frame [depth]: {:.2f}")
        frame_b_depth_inv = 1e-6 + 1 / torch.exp(frame_b_depth)

        filter0_ab = self.forward_singlePath(self.initScaleNets_filter, torch.cat((self.frame_a, frame_b), dim=1), 'filter')
        cur_filter_output = [
            self.forward_singlePath(self.initScaleNets_filter1, filter0_ab, name=None).float(),
            self.forward_singlePath(self.initScaleNets_filter2, filter0_ab, name=None).float()
            ]
        del filter0_ab
        self.statPrintMessage("frame [filters]: {:.2f}")

        flownets_ab = self.forward_flownets(self.flownets,
                                            torch.cat((self.frame_a, frame_b), dim=1),
                                            time_offsets=self.time_offsets)
        flownets_ba = self.forward_flownets(self.flownets,
                                            torch.cat((frame_b, self.frame_a), dim=1),
                                            time_offsets=self.time_offsets[::-1])
        self.statPrintMessage("frame [flownets]: {:.2f}")
        
        
        cur_offset_output = [ # return float32
            self.FlowProject(flownets_ab, self.frame_a_depth_inv)[0],
            self.FlowProject(flownets_ba, frame_b_depth_inv)[0]
        ]
        del flownets_ab, flownets_ba
        self.statPrintMessage("frame [flowProjects]: {:.2f}")
        
        cur_output, ref0, ref2 = self.FilterInterpolate(self.frame_a.float(), frame_b.float(),
                                                        [cur_offset_output[0], cur_offset_output[1]],
                                                        cur_filter_output, self.filter_size**2)
        
        self.statPrintMessage("frame [interpolate]: {:.2f}")
        
        # =============RECTIFYING========
        frame_a_ctx = self.ctxNet(self.frame_a)
        frame_b_ctx = self.ctxNet(frame_b)
        ctx0, ctx2 = self.FilterInterpolate_ctx(torch.cat((frame_a_ctx, self.frame_a_depth), dim=1).float(),
                                                torch.cat((frame_b_ctx, frame_b_depth), dim=1).float(),
                                                cur_offset_output,
                                                [cur_filter_output[0].float(), cur_filter_output[1].float()])
        del frame_a_ctx, frame_b_ctx

        rectify_input = torch.cat((cur_output, ref0, ref2,
                                cur_offset_output[0], cur_offset_output[1],
                                cur_filter_output[0], cur_filter_output[1],
                                ctx0, ctx2), dim=1)
        
        del cur_filter_output, cur_offset_output, ctx0, ctx2, ref0, ref2
        
        result = self.rectifyNet(rectify_input.type(torch.get_default_dtype())) + cur_output.type(torch.get_default_dtype())
        del rectify_input
        self.statPrintMessage("frame [rectify]: {:.2f}")
        
        self.frame_a = frame_b
        self.frame_a_depth = frame_b_depth
        self.frame_a_depth_inv = frame_b_depth_inv
        self.statPrintMessage("frame [net end]: {:.2f}")
        return result

    def forward_flownets(self, model, input, time_offsets = None):
        if time_offsets == None:
            time_offsets = [0.5]
        elif type(time_offsets) == float:
            time_offsets = [time_offsets]
        elif type(time_offsets) == list:
            pass
        temp = model(input)  # this is a single direction motion results, but not a bidirectional one

        temps = [self.div_flow * temp * time_offset for time_offset in time_offsets]# single direction to bidirection should haven it.
        temps = [nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)(temp)  for temp in temps]# nearest interpolation won't be better i think
        return temps

    '''keep this function'''
    def forward_singlePath(self, modulelist, input, name):
        stack = Stack()

        k = 0
        temp = []
        for layers in modulelist:  # self.initScaleNets_offset:
            # print(type(layers).__name__)
            # print(k)
            # if k == 27:
            #     print(k)
            #     pass
            # use the pop-pull logic, looks like a stack.
            if k == 0:
                temp = layers(input)
            else:
                # met a pooling layer, take its input
                if isinstance(layers, nn.AvgPool2d) or isinstance(layers,nn.MaxPool2d):
                    stack.push(temp)

                temp = layers(temp)

                # met a unpooling layer, take its output
                if isinstance(layers, nn.Upsample):
                    if name == 'offset':
                        temp = torch.cat((temp,stack.pop()),dim=1)  # short cut here, but optical flow should concat instead of add
                    else:
                        temp += stack.pop()  # short cut here, but optical flow should concat instead of add
            k += 1
        return temp

    '''keep this funtion'''
    def get_MonoNet5(self, channel_in, channel_out, name):

        '''
        Generally, the MonoNet is aimed to provide a basic module for generating either offset, or filter, or occlusion.

        :param channel_in: number of channels that composed of multiple useful information like reference frame, previous coarser-scale result
        :param channel_out: number of output the offset or filter or occlusion
        :param name: to distinguish between offset, filter and occlusion, since they should use different activations in the last network layer

        :return: output the network model
        '''
        model = []

        # block1
        model += self.conv_relu(channel_in * 2, 16, (3, 3), (1, 1))
        model += self.conv_relu_maxpool(16, 32, (3, 3), (1, 1), (2, 2))  # THE OUTPUT No.5
        # block2
        model += self.conv_relu_maxpool(32, 64, (3, 3), (1, 1), (2, 2))  # THE OUTPUT No.4
        # block3
        model += self.conv_relu_maxpool(64, 128, (3, 3), (1, 1), (2, 2))  # THE OUTPUT No.3
        # block4
        model += self.conv_relu_maxpool(128, 256, (3, 3), (1, 1), (2, 2))  # THE OUTPUT No.2
        # block5
        model += self.conv_relu_maxpool(256, 512, (3, 3), (1, 1), (2, 2))

        # intermediate block5_5
        model += self.conv_relu(512, 512, (3, 3), (1, 1))

        # block 6
        model += self.conv_relu_unpool(512, 256, (3, 3), (1, 1), 2)  # THE OUTPUT No.1 UP
        # block 7
        model += self.conv_relu_unpool(256, 128, (3, 3), (1, 1), 2)  # THE OUTPUT No.2 UP
        # block 8
        model += self.conv_relu_unpool(128, 64, (3, 3), (1, 1), 2)  # THE OUTPUT No.3 UP

        # block 9
        model += self.conv_relu_unpool(64, 32, (3, 3), (1, 1), 2)  # THE OUTPUT No.4 UP

        # block 10
        model += self.conv_relu_unpool(32,  16, (3, 3), (1, 1), 2)  # THE OUTPUT No.5 UP

        # output our final purpose
        branch1 = []
        branch2 = []
        branch1 += self.conv_relu_conv(16, channel_out,  (3, 3), (1, 1))
        branch2 += self.conv_relu_conv(16, channel_out,  (3, 3), (1, 1))

        return  (nn.ModuleList(model), nn.ModuleList(branch1), nn.ModuleList(branch2))

    '''keep this function'''
    @staticmethod
    def FlowProject(inputs, depth = None):
        if depth is not None:
            outputs = [DepthFlowProjectionModule(input.requires_grad)(input.float(), depth.float()) for input in inputs]
        else:
            outputs = [FlowProjectionModule(input.requires_grad)(input.float()) for input in inputs]
        return outputs


    '''keep this function'''
    @staticmethod
    def FilterInterpolate_ctx(ctx0,ctx2,offset,filter):
        ##TODO: which way should I choose

        ctx0_offset = FilterInterpolationModule()(ctx0,offset[0].detach(),filter[0].detach())
        ctx2_offset = FilterInterpolationModule()(ctx2,offset[1].detach(),filter[1].detach())

        return ctx0_offset, ctx2_offset
        # ctx0_offset = FilterInterpolationModule()(ctx0.detach(), offset[0], filter[0])
        # ctx2_offset = FilterInterpolationModule()(ctx2.detach(), offset[1], filter[1])
        #
        # return ctx0_offset, ctx2_offset
    '''Keep this function'''
    @staticmethod
    def FilterInterpolate(ref0, ref2, offset, filter,filter_size2):
        ref0_offset = FilterInterpolationModule()(ref0, offset[0],filter[0])
        ref2_offset = FilterInterpolationModule()(ref2, offset[1],filter[1])
        return ref0_offset/2.0 + ref2_offset/2.0, ref0_offset,ref2_offset

    '''keep this function'''
    @staticmethod
    def conv_relu_conv(input_filter, output_filter, kernel_size,
                        padding):

        # we actually don't need to use so much layer in the last stages.
        layers = nn.Sequential(
            nn.Conv2d(input_filter, input_filter, kernel_size, 1, padding),
            nn.ReLU(inplace=False),
            nn.Conv2d(input_filter, output_filter, kernel_size, 1, padding),
            # nn.ReLU(inplace=False),
            # nn.Conv2d(output_filter, output_filter, kernel_size, 1, padding),
            # nn.ReLU(inplace=False),
            # nn.Conv2d(output_filter, output_filter, kernel_size, 1, padding),
        )
        return layers


    '''keep this fucntion'''
    @staticmethod
    def conv_relu(input_filter, output_filter, kernel_size,
                        padding):
        layers = nn.Sequential(*[
            nn.Conv2d(input_filter,output_filter,kernel_size,1, padding),

            nn.ReLU(inplace=False)
        ])
        return layers

    '''keep this function'''
    @staticmethod
    def conv_relu_maxpool(input_filter, output_filter, kernel_size,
                            padding,kernel_size_pooling):

        layers = nn.Sequential(*[
            nn.Conv2d(input_filter,output_filter,kernel_size,1, padding),

            nn.ReLU(inplace=False),

            # nn.BatchNorm2d(output_filter),

            nn.MaxPool2d(kernel_size_pooling)
        ])
        return layers

    '''klkeep this function'''
    @staticmethod
    def conv_relu_unpool(input_filter, output_filter, kernel_size,
                            padding,unpooling_factor):

        layers = nn.Sequential(*[

            nn.Upsample(scale_factor=unpooling_factor, mode='bilinear', align_corners=True),

            nn.Conv2d(input_filter,output_filter,kernel_size,1, padding),

            nn.ReLU(inplace=False),

            # nn.BatchNorm2d(output_filter),


            # nn.UpsamplingBilinear2d(unpooling_size,scale_factor=unpooling_size[0])
        ])
        return layers
