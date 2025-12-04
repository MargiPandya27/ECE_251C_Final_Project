# models/dawn.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .lifting import LiftingScheme2D, LiftingScheme


class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(BottleneckBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        # This disable the conv if compression rate is equal to 1
        self.disable_conv = in_planes == out_planes
        if not self.disable_conv:
            self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                                   padding=0, bias=False)

    def forward(self, x):
        if self.disable_conv:
            return self.relu(self.bn1(x))
        else:
            return self.conv1(self.relu(self.bn1(x)))


# ---------------------------
# Enhanced CBAM attention definitions with improvements
# ---------------------------
class EnhancedChannelAttention(nn.Module):
    """Enhanced channel attention with ECA-style adaptive kernel and residual"""
    def __init__(self, in_planes, ratio=8, use_eca=True):
        super(EnhancedChannelAttention, self).__init__()
        self.use_eca = use_eca and in_planes >= 8
        
        # Always need avg_pool for both modes
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        if self.use_eca:
            # ECA-Net style: adaptive kernel size based on channel dimension
            k = int(abs((math.log(in_planes, 2) + 1) / 2))
            k = k if k % 2 else k + 1
            self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
            self.sigmoid = nn.Sigmoid()
        else:
            # Standard CBAM channel attention
            hidden = max(1, in_planes // ratio)
            self.max_pool = nn.AdaptiveMaxPool2d(1)
            self.fc1 = nn.Conv2d(in_planes, hidden, 1, bias=False)
            self.relu1 = nn.ReLU(inplace=True)
            self.fc2 = nn.Conv2d(hidden, in_planes, 1, bias=False)
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.use_eca:
            # ECA-Net: efficient channel attention
            y = self.avg_pool(x)
            y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
            y = self.sigmoid(y)
            return y
        else:
            avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
            max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
            return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # compute along channel axis
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        return self.sigmoid(out)


class CBAM(nn.Module):
    """Enhanced CBAM with residual connection"""
    def __init__(self, planes, ratio=8, kernel_size=7, use_residual=True, use_eca=True):
        super(CBAM, self).__init__()
        self.use_residual = use_residual
        self.ca = EnhancedChannelAttention(planes, ratio=ratio, use_eca=use_eca)
        self.sa = SpatialAttention(kernel_size=kernel_size)
        # Add batch norm for stability
        self.bn = nn.BatchNorm2d(planes)

    def forward(self, x):
        identity = x
        x = x * self.ca(x)
        x = x * self.sa(x)
        if self.use_residual:
            x = self.bn(x + identity)
        return x


# ---------------------------
# Haar wrapper (optional)
# ---------------------------
class Haar(nn.Module):
    def __init__(self, in_planes, lifting_size, kernel_size, no_bottleneck,
                 share_weights, simple_lifting, regu_details, regu_approx):
        super(Haar, self).__init__()
        try:
            from pytorch_wavelets import DWTForward
            self.wavelet = DWTForward(J=1, mode='zero', wave='db1')
        except Exception as e:
            raise ImportError("pytorch_wavelets is required for Haar. Install with `pip install pytorch-wavelets`") from e

        self.share_weights = share_weights
        # bottleneck setup consistent with original
        if no_bottleneck:
            self.bootleneck = BottleneckBlock(in_planes * 1, in_planes * 1)
        else:
            self.bootleneck = BottleneckBlock(in_planes * 4, in_planes * 2)

        # Enhanced attention for approximation and details
        self.att_x = CBAM(in_planes, use_eca=True)
        self.att_details = CBAM(in_planes * 3, use_eca=True)

    def forward(self, x):
        # returns LL and H where H is list with shape (batch, channels*3, h/2, w/2) structure
        LL, H = self.wavelet(x)
        LH = H[0][:, :, 0, :, :]
        HL = H[0][:, :, 1, :, :]
        HH = H[0][:, :, 2, :, :]

        # apply attention
        x_att = self.att_x(LL)
        details = torch.cat([LH, HL, HH], 1)
        details_att = self.att_details(details)

        r = 0  # No regularisation here

        if self.bootleneck:
            return self.bootleneck(x_att), r, details_att
        else:
            return x_att, r, details_att


# ---------------------------
# LevelDAWN with enhanced attention
# ---------------------------
class LevelDAWN(nn.Module):
    def __init__(self, in_planes, lifting_size, kernel_size, no_bottleneck,
                 share_weights, simple_lifting, regu_details, regu_approx,
                 separate_detail_attention=True, use_eca=True):
        super(LevelDAWN, self).__init__()
        self.regu_details = regu_details
        self.regu_approx = regu_approx
        self.separate_detail_attention = separate_detail_attention
        if self.regu_approx + self.regu_details > 0.0:
            # use Smooth L1 to be less sensitive to outliers
            self.loss_details = nn.SmoothL1Loss()

        self.wavelet = LiftingScheme2D(in_planes, share_weights,
                                       size=lifting_size, kernel_size=kernel_size,
                                       simple_lifting=simple_lifting)
        self.share_weights = share_weights
        if no_bottleneck:
            self.bootleneck = BottleneckBlock(in_planes * 1, in_planes * 1)
        else:
            self.bootleneck = BottleneckBlock(in_planes * 4, in_planes * 2)

        # Architecture matching the diagram: specialized attention per component
        if separate_detail_attention:
            # LL → Channel Attention only
            self.att_ll_channel = EnhancedChannelAttention(in_planes, use_eca=use_eca)
            # LH → Spatial Attention only
            self.att_lh_spatial = SpatialAttention(kernel_size=7)
            # HL → Spatial Attention only  
            self.att_hl_spatial = SpatialAttention(kernel_size=7)
            # HH → Fusion/Detail Attention (full CBAM)
            self.att_hh_fusion = CBAM(in_planes, use_eca=use_eca)
            # Multi-scale fusion attention after concatenation (LL + LH + HL + HH)
            self.att_fusion = CBAM(in_planes * 4, use_eca=use_eca)
        else:
            # Original approach: LL gets full CBAM, details get single CBAM
            self.att_x = CBAM(in_planes, use_eca=use_eca)
            self.att_details = CBAM(in_planes * 3, use_eca=use_eca)

    def forward(self, x):
        (c, d, LL, LH, HL, HH) = self.wavelet(x)

        # Architecture matching the diagram:
        if self.separate_detail_attention:
            # LL → Channel Attention only
            ll_att = LL * self.att_ll_channel(LL)
            
            # LH → Spatial Attention only
            lh_att = LH * self.att_lh_spatial(LH)
            
            # HL → Spatial Attention only
            hl_att = HL * self.att_hl_spatial(HL)
            
            # HH → Fusion/Detail Attention (full CBAM)
            hh_att = self.att_hh_fusion(HH)
            
            # Multi-scale fusion: concatenate all components
            fused = torch.cat([ll_att, lh_att, hl_att, hh_att], 1)
            # Apply fusion attention
            details_att = self.att_fusion(fused)
            
            # For backward compatibility, use LL as x_att
            x_att = ll_att
        else:
            # Original approach: LL gets full CBAM, details get single CBAM
            x_att = self.att_x(LL)
            details = torch.cat([LH, HL, HH], 1)
            details_att = self.att_details(details)

        r = None
        if (self.regu_approx + self.regu_details != 0.0):
            # Constraint on the details
            rd = 0.0
            rc = 0.0
            if self.regu_details:
                rd = self.regu_details * d.abs().mean()
                rd = rd + self.regu_details * LH.abs().mean()
                rd = rd + self.regu_details * HH.abs().mean()

            # Constraint on the approximation
            if self.regu_approx:
                # keep similar constraints as original
                rc = self.regu_approx * torch.dist(c.mean(), x.mean(), p=2)
                rc = rc + self.regu_approx * torch.dist(LL.mean(), c.mean(), p=2)
                rc = rc + self.regu_approx * torch.dist(HL.mean(), d.mean(), p=2)

            if self.regu_approx == 0.0:
                r = rd
            elif self.regu_details == 0.0:
                r = rc
            else:
                r = rd + rc

        if self.bootleneck:
            return self.bootleneck(x_att), r, details_att
        else:
            return x_att, r, details_att

    def image_levels(self, x):
        (c, d, LL, LH, HL, HH) = self.wavelet(x)
        x_cat = torch.cat([LL, LH, HL, HH], 1)

        if self.bootleneck:
            return self.bootleneck(x_cat), (LL, LH, HL, HH)
        else:
            return x_cat, (LL, LH, HL, HH)


# ---------------------------
# DAWN main model (unchanged logic, attention integrated via levels)
# ---------------------------
class DAWN(nn.Module):
    def __init__(self, num_classes, big_input=True, first_conv=3,
                 number_levels=4,
                 lifting_size=[2, 1], kernel_size=4, no_bootleneck=False,
                 classifier="mode1", share_weights=False, simple_lifting=False,
                 COLOR=True, regu_details=0.01, regu_approx=0.01, haar_wavelet=False):
        super(DAWN, self).__init__()
        self.big_input = big_input
        if COLOR:
            channels = 3
        else:
            channels = 1

        self.initialization = False
        self.nb_channels_in = first_conv

        # First convolution
        if first_conv != 3 and first_conv != 1:
            self.first_conv = True
            # Old parameter that tune the gabor filters
            self.conv1 = nn.Sequential(
                nn.Conv2d(channels, first_conv,
                          kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(first_conv),
                nn.ReLU(True),
                nn.Conv2d(first_conv, first_conv,
                          kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(first_conv),
                nn.ReLU(True),
            )
        else:
            self.first_conv = False
        if big_input:
            img_size = 224
        else:
            img_size = 32

        print("DAWN:")
        print("- first conv:", first_conv)
        print("- image size:", img_size)
        print("- nb levels :", number_levels)
        print("- levels U/P:", lifting_size)
        print("- channels: ", channels)

        # Construct the levels recursively
        self.levels = nn.ModuleList()

        in_planes = first_conv
        out_planes = first_conv
        for i in range(number_levels):
            bootleneck = True
            if no_bootleneck and i == number_levels - 1:
                bootleneck = False
            if i == 0:
                if haar_wavelet:
                    self.levels.add_module(
                        'level_' + str(i),
                        Haar(in_planes,
                             lifting_size, kernel_size, bootleneck,
                             share_weights, simple_lifting, regu_details, regu_approx)
                    )
                else:
                    self.levels.add_module(
                        'level_' + str(i),
                        LevelDAWN(in_planes,
                                  lifting_size, kernel_size, bootleneck,
                                  share_weights, simple_lifting, regu_details, regu_approx,
                                  separate_detail_attention=True, use_eca=True)
                    )
            else:
                self.levels.add_module(
                    'level_' + str(i),
                    LevelDAWN(in_planes,
                              lifting_size, kernel_size, bootleneck,
                              share_weights, simple_lifting, regu_details, regu_approx,
                              separate_detail_attention=True, use_eca=True)
                )
            in_planes *= 1
            img_size = img_size // 2
            # Here you can change this number if you want compression
            out_planes += in_planes * 3

        if no_bootleneck:
            in_planes *= 1
        self.img_size = img_size

        self.num_planes = out_planes

        print("Final channel:", self.num_planes)
        print("Final size   :", self.img_size)

        # Enhanced classifier definitions with dropout
        if classifier == "mode1":
            # Improved mode1 with dropout for regularization
            self.fc = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(out_planes, num_classes)
            )
        elif classifier == "mode2":
            if in_planes // 2 < num_classes:
                raise "Impossible to use mode2 in such scenario, abort"
            self.fc = nn.Sequential(
                nn.Linear(in_planes, in_planes // 2),
                nn.BatchNorm1d(in_planes // 2),
                nn.ReLU(True),
                nn.Dropout(0.2),
                nn.Linear(in_planes // 2, num_classes)
            )
        elif classifier == "mode3":
            # New enhanced classifier with more capacity
            hidden_dim = max(out_planes // 2, num_classes * 4)
            self.fc = nn.Sequential(
                nn.Linear(out_planes, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(True),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.ReLU(True),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim // 2, num_classes)
            )
        else:
            raise "Unknown classifier"

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Learnable weights for multi-level feature fusion
        self.level_weights = nn.Parameter(torch.ones(number_levels + 1) / (number_levels + 1))

    def process_levels(self, x):
        """This method is used for visualization purposes"""
        w, h = x.shape[-2:]

        # Choose to make X average
        x = x[:, 0, :, :]
        x = x.repeat(1, self.nb_channels_in, 1, 1)
        x_in = x
        print(x_in[:, 0, :, :])

        out = []
        out_down = []
        for l in self.levels:
            w = w // 2
            h = h // 2
            x_down = nn.AdaptiveAvgPool2d((w, h))(x_in)
            x, r, details = l(x)
            out_down += [x_down]
            out += [x]
        return out, out_down

    def forward(self, x):
        if self.initialization:
            # This mode is to train the weights of the lifting scheme only
            w, h = x.shape[-2:]
            rs = []
            rs_diff = []

            # Choose to make X average
            x = torch.mean(x, 1, True)
            x = x.repeat(1, self.nb_channels_in, 1, 1)
            x_in = x

            # Do all the levels
            for l in self.levels:
                w = w // 2
                h = h // 2
                x_down = nn.AdaptiveAvgPool2d((w, h))(x_in)
                x, r, details = l(x)
                diff = torch.dist(x, x_down, p=2)
                rs += [r]
                rs_diff += [diff]
            return rs_diff, rs
        else:
            if self.first_conv:
                x = self.conv1(x)

            # Apply the different levels sequentially
            rs = []  # List of constraints on details and mean
            det = []  # List of averaged pooled details
            level_features = []  # Store features for cross-level attention

            for i, l in enumerate(self.levels):
                x, r, details = l(x)
                # Add the constraint of this level
                rs += [r]
                # Globally avgpool all the details
                details_pooled = self.avgpool(details)
                det += [details_pooled]
                level_features.append(details_pooled)
            
            # At the last level (only) we GAP the approximation coefficients
            aprox = self.avgpool(x)
            level_features.append(aprox)
            
            # Apply learnable weighted fusion
            if len(self.level_weights) == len(level_features):
                # Normalize weights
                weights = F.softmax(self.level_weights, dim=0)
                # Weighted combination
                weighted_features = []
                for i, feat in enumerate(level_features):
                    weighted_features.append(weights[i] * feat)
                x = torch.cat(weighted_features, 1)
            else:
                # Fallback to original concatenation
                det += [aprox]
                x = torch.cat(det, 1)
            
            x = x.view(-1, x.size()[1])

            return self.fc(x), rs

    def image_levels(self, x):
        """This method is used for visualization purposes"""
        if self.first_conv:
            x = self.conv1(x)

        images = []
        for l in self.levels:
            x, curr_images = l.image_levels(x)
            images += [(curr_images[0], curr_images[1],
                        curr_images[2], curr_images[3])]
        return images
