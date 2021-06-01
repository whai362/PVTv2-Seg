_base_ = './fpn_pvtv2_B2_ade20k_40k.py'

# model settings
model = dict(
    pretrained='pretrained/pvt_v2_b1.pth',
    backbone=dict(
        type='pvt_v2_b1',
        style='pytorch'),
    neck=dict(in_channels=[64, 128, 320, 512]))
