
from model.model import MFE


def model_chose(model,args):
    num_classes = 1
    if model == 'MFE':
        net = MFE(num_scales=args.num_scales,
                      upsample_factor=args.upsample_factor, feature_channels=256, )
    return net
def run_model(net, model, SeqData,args):

    # Old_Feat = SeqData[:,:,:-1, :,:] * 0  # interface for iteration input
    # OldFlag = 1  # 1: i
    if model =='MFE':
        img0= SeqData[:,:,0,:,:].repeat(1, 3, 1, 1)
        img1 = SeqData[:, :, 1, :, :].repeat(1, 3, 1, 1)
        img2 = SeqData[:, :, 2, :, :].repeat(1, 3, 1, 1)
        out,warp_img,img=net(img0, img1, img2,
                                     attn_window_list=args.attn_splits_list,
				                     Local_Window_list=args.corr_radius_list,
				                     prop_radius_list=args.prop_radius_list,
				                     )
        outputs=[ out,warp_img,img]
    elif model == 'FLMY2':
        img0 = SeqData[:, :, 0, :, :].repeat(1, 3, 1, 1)
        img1 = SeqData[:, :, 1, :, :].repeat(1, 3, 1, 1)
        img2 = SeqData[:, :, 2, :, :].repeat(1, 3, 1, 1)
        out, warp_img, img = net(img0, img1, img2,
                                 attn_splits_list=args.attn_splits_list,
                                 corr_radius_list=args.corr_radius_list,
                                 prop_radius_list=args.prop_radius_list,
                                 )
        outputs = [out, warp_img, img]
    elif model == 'nodbcm':
        img0 = SeqData[:, :, 0, :, :].repeat(1, 3, 1, 1)
        img1 = SeqData[:, :, 1, :, :].repeat(1, 3, 1, 1)
        img2 = SeqData[:, :, 2, :, :].repeat(1, 3, 1, 1)
        out = net(img0, img1, img2,
                  attn_splits_list=args.attn_splits_list,
                  corr_radius_list=args.corr_radius_list,
                  prop_radius_list=args.prop_radius_list,
                  )
        outputs = out

    return outputs
