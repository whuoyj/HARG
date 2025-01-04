import os
from layer import *
from config import config
from multi_head_attention import MultiHeadAttention
from MultiStageTCN import MultiStageModel
from AutomaticWeightedLoss import AutomaticWeightedLoss
import torch.nn.functional as F

class GCN_Transformer(nn.Module):
    def __init__(self, d_model, visual_dim, target_dim, feat_dim, adj_matix, num_v, dropout=0.5):
        super().__init__()

        encoders = nn.ModuleList([
            EncoderLayer([config['max_frames'], d_model],
                         MultiHeadAttention(config['num_heads'], d_model),
                         PositionWiseFeedForward(d_model, config['d_ff'])),
            EncoderLayer([config['max_frames'], d_model],
                         MultiHeadAttention(config['num_heads'], d_model),
                         PositionWiseFeedForward(d_model, config['d_ff'])),
        ])
        self.encoder = MyEncoder(encoders)

        decoders = nn.ModuleList([
            DecoderLayer([config['max_target'], d_model],
                         MultiHeadAttention(config['num_heads'], d_model),
                         MultiHeadAttention(config['num_heads'], d_model),
                         PositionWiseFeedForward(d_model, config['d_ff'])),
            DecoderLayer([config['max_target'], d_model],
                         MultiHeadAttention(config['num_heads'], d_model),
                         MultiHeadAttention(config['num_heads'], d_model),
                         PositionWiseFeedForward(d_model, config['d_ff'])),
            DecoderLayer([config['max_target'], d_model],
                         MultiHeadAttention(config['num_heads'], d_model),
                         MultiHeadAttention(config['num_heads'], d_model),
                         PositionWiseFeedForward(d_model, config['d_ff']))
        ])

        self.decoder = MyDecoder(decoders)

        self.i_linear = nn.ModuleList([nn.Linear(config["feat_dims"]*4+config["semantic_dim"], d_model), nn.Linear(target_dim, d_model)])
        self.o_linear = nn.ModuleList([nn.Linear(d_model, config['num_class']), nn.Linear(config['max_target'], 1)])
        self.pos1 = PositionEncoder(d_model, config['max_frames'])
        self.pos2 = PositionEncoder(d_model, config['max_target'])
        self.loss = nn.CrossEntropyLoss()

        self.gcl = GraphConvolution(visual_dim, 1024, num_v, dropout=dropout)
        self.gcl2 = GraphConvolution(1024, feat_dim, num_v, dropout=dropout)
        self.spatial = nn.Linear(config["spatial_dim"], config["feat_dims"])

    def forward(self, src, src_mask, tgt, tgt_mask, device):
        #target, data and mask
        target = torch.topk(tgt, 1, dim=2)[1]
        target = torch.squeeze(target, dim=2)
        src_mask[:, -config['pred_step']:] = 0
        tgt_mask[:, -config['pred_step']:] = 0

        src[:, -config['pred_step']:, :] = 0
        tgt[:, -config['pred_step']:, :] = 0
        #feat
        visual_feat_s = src[:, :-config['pred_step'], :1536]
        visual_feat_p = src[:, :-config['pred_step'], 1536:3072]
        visual_feat_o = src[:, :-config['pred_step'], 3072:4608]

        spatial_feat = src[:, :, 4608:4628]
        semantic_feat = src[:, :, 4628:]
        if config["dataset"] == "AGP":
            semantic_feat = semantic_feat[:, :, 300:]

        spatial_feat = self.spatial(spatial_feat)

        visual_feat_src = torch.stack((visual_feat_s, visual_feat_p, visual_feat_o), axis=2)

        # graph
        visual_feat = self.gcl(self.adj_matrix, visual_feat_src)
        visual_feat = self.gcl2(self.adj_matrix, visual_feat)
        tmp_feat = torch.zeros(visual_feat.size(0), config["max_frames"], config["feat_dims"]).to(device)
        tmp_feat[:, :-config['pred_step'], :] = visual_feat[:, :, 0, :].squeeze()
        visual_feat_s = tmp_feat
        tmp_feat[:, :-config['pred_step'], :] = visual_feat[:, :, 1, :].squeeze()
        visual_feat_p = tmp_feat
        tmp_feat[:, :-config['pred_step'], :] = visual_feat[:, :, 2, :].squeeze()
        visual_feat_o = tmp_feat

        src_gcn = torch.cat([visual_feat_s, visual_feat_p, visual_feat_o, spatial_feat, semantic_feat], 2)
        #postion
        src = self.pos1(self.i_linear[0](src_gcn))
        tgt = self.pos2(self.i_linear[1](tgt))

        #Encoder-Decoder
        x = self.encoder(src, src_mask)  # [nb, len1, hid]
        x = self.decoder(tgt, x, src_mask, tgt_mask)  # [nb, len2, hid]
        predict = self.o_linear[0](x).view(-1, config['max_target'], config['num_class'])
        #predict = predict.permute(0, 2, 1)
        #predict = self.o_linear[1](predict).view(-1, config['num_class'])
        #loss = self.loss(predict, target[:,-1])
        #return predict,target[:,-1],loss

        loss = self.loss(predict[:,-config["pred_step"]:,:].reshape(-1, config['num_class']), target[:,-config["pred_step"]:].reshape(-1))

        #return predict[:,-6,:].reshape(-1, config['num_class']),target[:,-6],loss
        return predict[:,-config["pred_step"]:,:].reshape(-1, config['num_class']), target[:,-config["pred_step"]:].reshape(-1), loss



class Transformer(nn.Module):
    def __init__(self, d_model, visual_dim, target_dim):
        super().__init__()

        encoders = nn.ModuleList([
            EncoderLayer([config['max_frames'], d_model],
                         MultiHeadAttention(config['num_heads'], d_model),
                         PositionWiseFeedForward(d_model, config['d_ff'])),
        ])
        self.encoder = MyEncoder(encoders)

        decoders = nn.ModuleList([
            DecoderLayer([config['max_target'], d_model],
                         MultiHeadAttention(config['num_heads'], d_model),
                         MultiHeadAttention(config['num_heads'], d_model),
                         PositionWiseFeedForward(d_model, config['d_ff'])),
            DecoderLayer([config['max_target'], d_model],
                         MultiHeadAttention(config['num_heads'], d_model),
                         MultiHeadAttention(config['num_heads'], d_model),
                         PositionWiseFeedForward(d_model, config['d_ff'])),
        ])

        self.decoder = MyDecoder(decoders)
        # config["feat_dims"]*2+config['semantic_dim']+config['spatial_dim']
        self.i_linear = nn.ModuleList([nn.Linear(config["feat_dims"]*4+config["semantic_dim"], d_model), nn.Linear(target_dim, d_model),
                                       nn.Linear(config["visual_dim"], config["feat_dims"])])
        self.o_linear = nn.ModuleList([nn.Linear(d_model, config["num_class"]), nn.Linear(config["max_target"], 1)])
        self.pos1 = PositionEncoder(d_model, config["max_frames"])
        self.pos2 = PositionEncoder(d_model, config["max_target"])
        self.loss = nn.CrossEntropyLoss()
        #self.semantic = nn.Linear(config["semantic_dim"], config["feat_dims"])
        self.spatial = nn.Linear(config["spatial_dim"], config["feat_dims"])

    def forward(self, src, src_mask, tgt, tgt_mask, device):

        # target, data and mask
        target = torch.topk(tgt, 1, dim=2)[1]
        target = torch.squeeze(target, dim=2)
        src_mask[:, -config["pred_step"]:] = 0
        tgt_mask[:, -config["pred_step"]:] = 0

        src[:, -config["pred_step"]:, :] = 0
        tgt[:, -config["pred_step"]:, :] = 0
        # feat
        visual_feat_s = src[:, :-config["pred_step"], :1536]
        visual_feat_p = src[:, :-config["pred_step"], 1536:3072]
        visual_feat_o = src[:, :-config["pred_step"], 3072:4608]

        spatial_feat = src[:, :, 4608:4628]
        semantic_feat = src[:, :, 4628:]
        if config["dataset"] == "AGP":
            semantic_feat = semantic_feat[:, :, 300:]

        spatial_feat = self.spatial(spatial_feat)
        #semantic_feat = self.semantic(semantic_feat)

        tmp_feat = torch.zeros(visual_feat_s.size(0), config["max_frames"], config["feat_dims"]).to(device)
        tmp_feat[:, :-config["pred_step"], :] = self.i_linear[2](visual_feat_s)
        visual_feat_s = tmp_feat
        tmp_feat[:, :-config["pred_step"], :] = self.i_linear[2](visual_feat_p)
        visual_feat_p = tmp_feat
        tmp_feat[:, :-config["pred_step"], :] = self.i_linear[2](visual_feat_o)
        visual_feat_o = tmp_feat

        src_feat = torch.cat([visual_feat_s, visual_feat_p, visual_feat_o, spatial_feat, semantic_feat], 2)
        #src_feat = torch.cat([visual_feat_s, visual_feat_o, spatial_feat, semantic_feat], 2)

        src = self.pos1(self.i_linear[0](src_feat))
        tgt = self.pos2(self.i_linear[1](tgt))
        x = self.encoder(src, src_mask)  # [nb, len1, hid]

        x = self.decoder(tgt, x, src_mask, tgt_mask)  # [nb, len2, hid]
        predict = self.o_linear[0](x).view(-1, config['max_target'], config['num_class'])
        #predict = predict.permute(0, 2, 1)
        #predict = self.o_linear[1](predict).view(-1, config['num_class'])

        loss = self.loss(predict[:,-config["pred_step"]:,:].reshape(-1, config['num_class']), target[:,-config["pred_step"]:].reshape(-1))


        #return predict[:,-1:,:].reshape(-1, config['num_class']),target[:,-1],loss
        return predict[:, -config["pred_step"]:, :].reshape(-1, config['num_class']), target[:,-config["pred_step"]:].reshape(-1), loss



class GGCN(nn.Module):
    def __init__(self, visual_dim, target_dim, feat_dim, num_v, dropout=0.5):
        super(GGCN, self).__init__()

        self.gcl = GraphConvolution(visual_dim, 1024, num_v, dropout=dropout)
        self.gcl2 = GraphConvolution(1024, feat_dim, num_v, dropout=dropout)
        self.loss = nn.BCELoss()
        self.spatial = nn.Linear(config["spatial_dim"], config["feat_dims"])
        self.fc = nn.Linear(config["feat_dims"]*3, config["feat_dims"])
        self.fc1 = nn.Linear(config["feat_dims"], config["num_class"])
        #self.out = StandConvolution1([config["feat_dims"], config["feat_dims"] * 2, config["feat_dims"] * 4], config["num_class"], dropout)
        self.out = StandConvolution2([config["feat_dims"], config["feat_dims"]*2, config["feat_dims"]*4, config["feat_dims"]*8], config["num_class"], dropout)


    def forward(self, A, src, tgt, device):

        b, _, _, N, _ = src.size()
        src_gcn = torch.zeros(b, N, config["feat_dims"]).to(device)
        for i in range(b):
            visual_feat = self.gcl(A[i]+torch.eye(A[i].size(0)).to(A[i]).detach().float(), src[i])
            visual_feat = self.gcl2(A[i]+torch.eye(A[i].size(0)).to(A[i]).detach().float(), visual_feat)
            src_gcn[i] = visual_feat[0][0]
        out_gcn = src_gcn.reshape(b, config["Max_Object"], config["Max_Time"], config["feat_dims"])
        out = self.out(out_gcn)
        m = torch.nn.Sigmoid()
        pred = m(out)
        loss = self.loss(pred, tgt)

        return out,tgt,loss

class GGCN_weight(nn.Module):
    def __init__(self, visual_dim, target_dim, feat_dim, num_v, dropout=0.5):
        super(GGCN_weight, self).__init__()

        self.gcl = GraphConvolution(visual_dim, 1024, num_v, dropout=dropout)
        self.gcl2 = GraphConvolution(1024, feat_dim, num_v, dropout=dropout)
        self.gcl_2 = GraphConvolution(visual_dim, 1024, num_v, dropout=dropout)
        self.gcl2_2 = GraphConvolution(1024, feat_dim, num_v, dropout=dropout)
        self.loss = nn.BCELoss()
        self.spatial = nn.Linear(config["spatial_dim"], config["feat_dims"])
        self.fc = nn.Linear(config["feat_dims"]*3, config["feat_dims"])
        self.fc1 = nn.Linear(config["feat_dims"], config["num_class"])
        #self.out = StandConvolution1([config["feat_dims"], config["feat_dims"] * 2, config["feat_dims"] * 4], config["num_class"], dropout)
        self.out1 = StandConvolution2_weight([config["feat_dims"], config["feat_dims"]*2, config["feat_dims"]*4, config["feat_dims"]*8], config["num_class"], dropout)
        self.out2 = StandConvolution2_weight([config["feat_dims"], config["feat_dims"] * 2, config["feat_dims"] * 4, config["feat_dims"] * 8], config["num_class"], dropout)
        self.fc = nn.Linear(config["feat_dims"] * 8 * 4, config["num_class"])


    def forward(self, A1, src1, A2, src2, tgt, device):

        b, _, _, N, _ = src1.size()
        src_gcn1 = torch.zeros(b, N, config["feat_dims"]).to(device)
        src_gcn2 = torch.zeros(b, N, config["feat_dims"]).to(device)
        for i in range(b):
            visual_feat = self.gcl(A1[i]+torch.eye(A1[i].size(0)).to(A1[i]).detach().float(), src1[i])
            visual_feat = self.gcl2(A1[i]+torch.eye(A1[i].size(0)).to(A1[i]).detach().float(), visual_feat)
            src_gcn1[i] = visual_feat[0][0]
            visual_feat = self.gcl_2(A2[i] + torch.eye(A2[i].size(0)).to(A2[i]).detach().float(), src2[i])
            visual_feat = self.gcl2_2(A2[i] + torch.eye(A2[i].size(0)).to(A2[i]).detach().float(), visual_feat)
            src_gcn2[i] = visual_feat[0][0]
        out_gcn1 = src_gcn1.reshape(b, config["Max_Time"], config["Max_Object"], config["feat_dims"])
        out1 = self.out1(out_gcn1)
        out_gcn2 = src_gcn2.reshape(b, config["Max_Object"], config["Max_Time"], config["feat_dims"])
        out2 = self.out2(out_gcn2)
        out = self.fc(torch.cat([out1,out2], 1))
        m = torch.nn.Sigmoid()
        pred = m(out)
        loss = self.loss(pred, tgt)

        return out,tgt,loss

class VRD(nn.Module):
    def __init__(self, visual_dim, target_dim, feat_dim, dropout=0.5):
        super(VRD, self).__init__()

        self.target_dim = target_dim


        self.vis_hid = config["feat_dims"]
        self.sem_hid = config["feat_dims"]

        self.fc_vis = MLP(2048, self.vis_hid, self.vis_hid)
        #self.fc_sem = MLP(300, self.sem_hid, self.sem_hid)
        # ！！！！！！！！！！feature
        self.fc_fusion = FC(self.vis_hid, config["feat_dims"])
        self.fc_rel = FC(config["feat_dims"]*2, target_dim)



    def forward(self, src, tgt_rel, device):

        NO = int(config["Max_Object"])
        NT = int(config["Max_Time"])
        b, _, _, N, _ = src.size()



        # ==============================VRD=====================================
        # feature: src/tmp/src_ori/src_rel
        src = src[:, 0, 0, :, :].reshape(b * NT, NO * 2, config['input_video_dim'])



        # preocess:
        x_v = self.fc_vis(src[:,:,:2048])
        #x_s = self.fc_sem(src[:,:,1536:])
        node_feats = x_v #torch.cat([x_v,x_s], -1)
        node_feats = self.fc_fusion(node_feats)

        edge_feats = torch.zeros(b * NT, NO * NO, config["feat_dims"] * 2).to(device)# [4*16, 64, 512]
        for i in range(NO):
            for j in range(NO, 2 * NO):
                edge_feats[:, i*NO+j-NO, :] = torch.cat([node_feats[:, i, :], node_feats[:, j, :]], dim=1)

        #！！！！！！！！！output
        output = self.fc_rel(edge_feats)

        output = output.reshape(-1, self.target_dim)
        tgt_rel = tgt_rel.reshape(-1).long()
        if config["dataset"] == 'MOMA':
            w = torch.FloatTensor([1, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]).to(device)
        else:
            #w = torch.FloatTensor([1, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]).to(device)
            #w = torch.FloatTensor([1, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]).to(device)
            w = torch.FloatTensor([1, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000]).to(device)
        #w = torch.FloatTensor([1, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]).to(device)

        if config["mode"] == "VRD":
            loss_vrd = F.cross_entropy(output, tgt_rel, weight=w)#, ignore_index=0)

        # ===========================VRD-END=====================================



        # ===========================test_VRD===================================
            loss = loss_vrd
        else:
            loss = 0
        out = output
        tgt = tgt_rel

        return out,tgt,loss

class GGCN_relation(nn.Module):
    def __init__(self, visual_dim, relation_dim, feat_dim, num_v, dropout=0.5):
        super(GGCN_relation, self).__init__()

        if "RG" in config['models'] or "E2N" in config['models'] or "TCN" in config['models'] or "HDRG" in config['models']:
            self.VRD = VRD(visual_dim=visual_dim,
                       target_dim=relation_dim,
                       feat_dim=feat_dim,
                       dropout=dropout)
            self.relation_feature = self.read_relation_npy()

        self.nfeat = 300
        self.visual_dim = visual_dim
        self.headnum = 4
        self.dropout = nn.Dropout(0.0)

        self.gcl_o = GraphConvolution(visual_dim, 1024, dropout=dropout)
        self.gcl2_o = GraphConvolution(1024, feat_dim, dropout=dropout)
        if "RG" in config['models'] or "E2N" in config['models'] or "TCN" in config['models'] or "HDRG" in config['models']:
            self.gcl_r = GraphConvolution(self.nfeat, 1024, dropout=dropout)
            self.gcl2_r = GraphConvolution(1024, feat_dim, dropout=dropout)

        #self.autoloss = AutomaticWeightedLoss()
        #self.spatial = nn.Linear(config["spatial_dim"], config["feat_dims"])
        #self.semantic = nn.Linear(config["semantic_dim"], config["feat_dims"])
        #self.visual = nn.Linear(config["visual_dim"], config["feat_dims"])

        self.out_o = StandConvolution2([config["feat_dims"], config["feat_dims"] * 2, config["feat_dims"] * 4, config["feat_dims"] * 8], config["num_class"], dropout)
        if "RG" in config['models'] or "E2N" in config['models'] or "TCN" in config['models'] or "HDRG" in config['models']:
            self.out_r = StandConvolution2_RG([config["feat_dims"], config["feat_dims"] * 2, config["feat_dims"] * 4, config["feat_dims"] * 8], config["num_class"], dropout)
        if "E2N" in config['models'] or "TCN" in config['models'] or "HDRG" in config['models']:
            self.e2n_out_o2 = StandConvolution2([config["feat_dims"], config["feat_dims"] * 2, config["feat_dims"] * 4, config["feat_dims"] * 8], config["num_class"], dropout)
        if "HDRG" in config['models']:
            self.out_a = StandConvolution2_HDRG([config["feat_dims"], config["feat_dims"] * 2, config["feat_dims"] * 4, config["feat_dims"] * 8], config["num_class"], dropout)

        # all
        #self.i3d_fc_g = nn.Linear(config["feat_dims"] * 8 * 6, config["feat_dims"] * 8)
        #self.i3d_fc1 = nn.Linear(config["feat_dims"] * 8 * 7, config["num_class"])
        #self.i3d_fc2 = nn.Linear(config["feat_dims"] * 8 * 7, config["num_class2"])

        if "HDRG" in config['models']:
            # OG+RG+E2N+HDRG
            self.fc1 = nn.Linear(config["feat_dims"] * 8 * 6, config["num_class"])
            self.fc2 = nn.Linear(config["feat_dims"] * 8 * 8, config["num_class2"])
            self.fc3 = nn.Linear(config["feat_dims"] * 8 * 8, config["num_class4"])
        elif "E2N" in config['models'] or "TCN" in config['models']:
            # OG+RG+E2N
            self.fc1 = nn.Linear(config["feat_dims"] * 8 * 6, config["num_class"])
            self.fc2 = nn.Linear(config["feat_dims"] * 8 * 6, config["num_class2"])
            self.fc3 = nn.Linear(config["feat_dims"] * 8 * 6, config["num_class4"])

        elif "RG" in config['models']:
            # OG+RG
            self.fc1 = nn.Linear(config["feat_dims"] * 8 * 4, config["num_class"])
            self.fc2 = nn.Linear(config["feat_dims"] * 8 * 4, config["num_class2"])
            self.fc3 = nn.Linear(config["feat_dims"] * 8 * 4, config["num_class4"])

        else:
            # OG
            self.fc1 = nn.Linear(config["feat_dims"] * 8 * 2, config["num_class"])
            self.fc2 = nn.Linear(config["feat_dims"] * 8 * 2, config["num_class2"])
            self.fc3 = nn.Linear(config["feat_dims"] * 8 * 2, config["num_class4"])


        if "E2N" in config['models'] or "TCN" in config['models'] or "HDRG" in config['models']:
            self.e2n = MultiHeadAttention(config["feat_dims"], self.headnum)
            self.e2n_embed = nn.Linear(config["feat_dims"] * 2, config["feat_dims"])
            #self.e2n_embed2 = nn.Linear(config["feat_dims"] * 2, config["feat_dims"])

        if "TCN" in config['models'] or "HDRG" in config['models']:
            self.TCN = MultiStageModel(config["num_stages"], config["num_layers"], config["feat_dims"], config["feat_dims"], config["num_class"])

        if "HDRG" in config['models']:
            self.gcl_a = GraphConvolution(feat_dim, feat_dim, dropout=dropout)
            self.gcl2_a = GraphConvolution(feat_dim, feat_dim, dropout=dropout)

        # OG+RG+OG2
        #self.e2n_fc = nn.Linear(config["feat_dims"] * 8 * 5, config["num_class"])


    def forward(self, Ao, srco, Ar, srcr, tgt1, tgt2, tgt3, tgt4, i3d, device):

        if "RG" in config['models'] or "E2N" in config['models'] or "TCN" in config['models'] or "HDRG" in config['models']:
            VRD_out, VRD_tgt, VRD_loss = self.VRD(srco, tgt2, device)
            Ar_, relation_semantic_feat = self.compute_feature(VRD_out, device)

        NO = int(config["Max_Object"])
        NT = int(config["Max_Time"])
        NF = self.nfeat
        b, _, _, No, _ = srco.size()
        _, _, _, Nr, _ = srcr.size()
        ####relation_spatial_feat = srcr[:,:,:,:,:20]
        # relation_semantic_feat = srcr[:,:,:,:,:]
        ####relation_visual_feat = srcr[:,:,:,:,920:]
        srco = srco[:, :, :, :, :self.visual_dim]

        srco_new = torch.zeros(b, NO, 1, 1, (NO + 1) * NT, srco.size()[-1]).to(device)
        Ao_new = torch.zeros(b, NO, (NO + 1) * NT, (NO + 1) * NT).to(device)
        srcr_new = torch.zeros(b, NO, 1, 1, NO * NT, srcr.size()[-1]).to(device)
        Ar_new = torch.zeros(b, NO, NO * NT, NO * NT).to(device)

        for i in range(NO):
            for t in range(NT):
                srco_new[:, i, 0, 0, t*(NO+1), :] = srco[:, 0, 0, t * NO*2, :]
                srco_new[:, i, 0, 0, t*(NO+1)+1:(t+1)*(NO+1), :] = srco[:, 0, 0, t * NO*2 + NO:t * NO*2 + 2*NO, :]
                Ao_new[:, i, t * (NO + 1), t * (NO + 1)] = Ao[:, t * NO * 2, t * NO * 2]
                Ao_new[:, i, t * (NO + 1), t * (NO + 1) + 1: (t + 1) * (NO + 1)] = Ao[:, t * NO * 2, t * NO * 2 + NO:t * NO * 2 + 2 * NO]
                Ao_new[:, i, t * (NO + 1) + 1: (t + 1) * (NO + 1), t * (NO + 1)] = Ao[:, t * NO * 2 + NO:t * NO * 2 + 2 * NO, t * NO * 2]
                Ao_new[:, i, t * (NO + 1) + 1: (t + 1) * (NO + 1), t * (NO + 1) + 1: (t + 1) * (NO + 1)] = Ao[:, t * NO * 2 + NO:t * NO * 2 + 2 * NO, t * NO * 2 + NO:t * NO * 2 + 2 * NO]
                srcr_new[:, i, 0, 0, t * NO:(t + 1) * NO, :] = relation_semantic_feat[:, 0, 0, t * NO*NO + i * NO:t * NO*NO + (i + 1) * NO, :]
                Ar_new[:, i, t*NO:(t+1)*NO, t*NO:(t+1)*NO] = Ar[:, t*NO*NO+i*NO:t*NO*NO+(i+1)*NO, t*NO*NO+i*NO:t*NO*NO+(i+1)*NO]

        srco_new = srco_new.reshape(b * NO, 1, 1, (NO+1)*NT, srco.size()[-1])
        Ao_new = Ao_new.reshape(b * NO, (NO+1)*NT, (NO+1)*NT)
        srcr_new = srcr_new.reshape(b * NO, 1, 1, NO*NT, relation_semantic_feat.size()[-1])
        Ar_new = Ar_new.reshape(b * NO, NO*NT, NO*NT)


        srco_gcn = torch.zeros(b * NO, (NO+1)*NT, config["feat_dims"]).to(device)


        for i in range(b * NO):
            visual_feat = self.gcl_o(Ao_new[i] + torch.eye(Ao_new[i].size(0)).to(Ao_new[i]).detach().float(), srco_new[i])
            visual_feat = self.gcl2_o(Ao_new[i] + torch.eye(Ao_new[i].size(0)).to(Ao_new[i]).detach().float(), visual_feat)
            srco_gcn[i] = visual_feat[0][0]


        out_gcn_o = srco_gcn.reshape(b * NO, NT, NO + 1, config["feat_dims"])
        out_o = self.out_o(out_gcn_o)

        if "RG" in config['models'] or "E2N" in config['models'] or "TCN" in config['models'] or "HDRG" in config['models']:
            srcr_gcn = torch.zeros(b * NO, NO*NT, config["feat_dims"]).to(device)
            srcr_concat = srcr_new  # torch.cat((relation_semantic_feat, relation_spatial_feat), dim=4) #torch.cat((relation_spatial_feat, relation_semantic_feat, relation_visual_feat), dim=4)

            for i in range(b * NO):
                relation_feat = self.gcl_r(Ar_new[i] + torch.eye(Ar_new[i].size(0)).to(Ar_new[i]).detach().float(), srcr_concat[i])
                relation_feat = self.gcl2_r(Ar_new[i] + torch.eye(Ar_new[i].size(0)).to(Ar_new[i]).detach().float(), relation_feat)

                srcr_gcn[i] = relation_feat[0][0]  # + relation_feat[0][0]


            out_gcn_r = srcr_gcn.reshape(b * NO, NT, NO, config["feat_dims"])
            out_r = self.out_r(out_gcn_r)






        # all
        #out_g = torch.cat((out_o, out_r), dim=1) #self.i3d_fc_g(torch.cat((out_o, out_r, update_o), dim=1)) #self.i3d_fc_g(torch.cat((out_o, out_r), dim=1)) #self.i3d_fc_g(out_o) #
        #out_g = out_g.unsqueeze(dim=1).repeat(1,30,1)
        #out1 = self.i3d_fc1(torch.cat((out_g, i3d), dim=2))
        #out1 = torch.max(input=out1, dim=1)[0]
        #out2 = self.i3d_fc2(torch.cat((out_g, i3d), dim=2))
        #out2 = torch.max(input=out2, dim=1)[0]

        if "HDRG" in config['models']:
            out_e2n = self.compute_e2n(out_gcn_o, out_gcn_r, device)
            out_tcn = self.compute_tcn(out_e2n, device)
            out_hdrg = self.compute_hdrg(out_e2n, out_tcn[-1], device)
            out1 = self.fc1(self.dropout(torch.cat((out_o, out_r, self.e2n_out_o2(out_e2n)), dim=1)))
            out2 = self.fc2(self.dropout(torch.cat((out_o, out_r, self.e2n_out_o2(out_e2n), out_hdrg), dim=1))) #self.fc2(out_hdrg) #
            out3 = self.fc3(self.dropout(torch.cat((out_o, out_r, self.e2n_out_o2(out_e2n), out_hdrg), dim=1))) #self.fc3(out_hdrg) #
        elif "TCN" in config['models']:
            out_e2n = self.compute_e2n(out_gcn_o, out_gcn_r, device)
            out_tcn = self.compute_tcn(out_e2n, device)
            out1 = self.fc1(torch.cat((out_o, out_r, self.e2n_out_o2(out_e2n)), dim=1))
            out2 = self.fc2(torch.cat((out_o, out_r, self.e2n_out_o2(out_e2n)), dim=1))
            out3 = self.fc3(torch.cat((out_o, out_r, self.e2n_out_o2(out_e2n)), dim=1))
        elif "E2N" in config['models']:
            e2n_input = out_gcn_r.reshape(b * NO * NT, NO, config["feat_dims"])
            out_e2n_o = self.e2n(e2n_input, e2n_input, e2n_input)
            out_e2n_o = torch.mean(out_e2n_o, dim=1, keepdim=True)
            out_e2n_o = torch.cat((out_e2n_o, e2n_input), dim=1)

            out_e2n_o = torch.cat((out_e2n_o.reshape(b * NO, NT, NO + 1, config["feat_dims"]), out_gcn_o), dim=3)
            out_e2n_o = self.e2n_embed(out_e2n_o)
            update_o = self.e2n_out_o2(out_e2n_o)

            out_o = torch.mean(out_o.reshape(b, NO, -1), dim=1, keepdim=False)
            out_r = torch.mean(out_r.reshape(b, NO, -1), dim=1, keepdim=False)
            update_o = torch.mean(update_o.reshape(b, NO, -1), dim=1, keepdim=False)

            out1 = self.fc1(torch.cat((out_o, out_r, update_o), dim=1))
            out2 = self.fc2(torch.cat((out_o, out_r, update_o), dim=1))
            out3 = self.fc3(torch.cat((out_o, out_r, update_o), dim=1))
        elif "RG" in config['models']:
            # OG+RG
            out1 = self.fc1(torch.cat((out_o, out_r), dim=1))
            out2 = self.fc2(torch.cat((out_o, out_r), dim=1))
            out3 = self.fc3(torch.cat((out_o, out_r), dim=1))
        else:
            # OG
            out1 = self.fc1(out_o)
            out2 = self.fc2(out_o)
            out3 = self.fc3(out_o)

        if "HDRG" in config['models']:

            # I3D
            # out = self.fc(i3d)
            # out = torch.max(input=out, dim=1)[0]
            # m = torch.nn.Sigmoid()
            pred1 = out1
            pred2 = out2
            pred3 = out3

            pred4 = torch.max(input=out_tcn, dim=2)[0]

            tgt4 = tgt4.reshape(b * NT, config["num_class"])

            loss4 = 0
            for p in pred4:
                mse = nn.MSELoss(reduction='none')
                loss4 += F.binary_cross_entropy_with_logits(p.transpose(2, 1).contiguous().view(-1, config["num_class"]),
                                                           tgt4)
                loss4 += 0.15 * torch.mean(
                    torch.clamp(mse(torch.sigmoid(p[:, :, 1:]), torch.sigmoid(p.detach()[:, :, :-1])), min=0, max=16))

            out4 = pred4[-1].transpose(2, 1).contiguous().view(-1, config["num_class"])

            loss1 = F.binary_cross_entropy_with_logits(pred1, tgt1)
            loss2 = F.cross_entropy(pred2, tgt2)
            loss3 = F.cross_entropy(pred3, tgt3)
            loss = loss1 * 0.5 + loss2 + loss3 + loss4

        else:

            pred1 = out1
            pred2 = out2
            pred3 = out3
            out4 = out1
            tgt4 = tgt1

            loss1 = F.binary_cross_entropy_with_logits(pred1, tgt1)
            loss2 = F.cross_entropy(pred2, tgt2)
            loss3 = F.cross_entropy(pred3, tgt3)
            loss = loss1 + loss2 + loss3 * 0.001

        return out1,tgt1,out2,tgt2,out3,tgt3,out4,tgt4,loss

    def compute_e2n(self, out_gcn_o, out_gcn_r, device):

        NO = int(config["Max_Object"])
        NT = int(config["Max_Time"])
        b, _, _, _ = out_gcn_o.size()

        e2n_input = out_gcn_r.reshape(b * NT, NO * NO, config["feat_dims"])
        out_e2n_tmp = torch.zeros(b * NT, 2 * NO, config["feat_dims"]).to(device)
        for i in range(NO):
            out_e2n_a = self.e2n(e2n_input[:, i * NO:(i + 1) * NO, :], e2n_input[:, i * NO:(i + 1) * NO, :], e2n_input[:, i * NO:(i + 1) * NO, :])
            out_e2n_a = torch.mean(out_e2n_a, dim=1, keepdim=True)
            out_e2n_tmp[:, i, :] = out_e2n_a[:, 0, :]


        e2n_input = e2n_input.reshape(b * NT, NO, NO, config["feat_dims"]).permute(0, 2, 1, 3).reshape(b * NT, NO * NO, config["feat_dims"])
        for i in range(NO):
            out_e2n_o = self.e2n(e2n_input[:, i * NO:(i + 1) * NO, :], e2n_input[:, i * NO:(i + 1) * NO, :], e2n_input[:, i * NO:(i + 1) * NO, :])
            out_e2n_o = torch.mean(out_e2n_o, dim=1, keepdim=True)
            out_e2n_tmp[:, i+NO, :] = out_e2n_o[:, 0, :]

        # out_e2n = torch.zeros(b * NT, 2 * NO, config["feat_dims"]).to(device)

        # e2n test1
        # for i in range(NO):
        #     out_e2n[:, i, :] = self.e2n_embed(torch.cat((out_e2n_tmp[:, i, :], out_e2n_tmp[:, 8:16, :].reshape(b * NT, -1)), dim=1))

        # e2n test2
        # for i in range(NO):
        #     out_e2n[:,i,:] = self.e2n_embed2(torch.cat((self.e2n_embed(torch.cat((out_e2n_tmp[:,i,:], out_e2n_tmp[:,8:16,:].reshape(b*NT, -1)), dim = 1)), out_gcn_o.reshape(b*NT, 2*NO, config["feat_dims"])[:,i,:]), dim = 1))

        # e2n test3
        # for i in range(NO):
        #     out_e2n[:, i, :] = self.e2n_embed(torch.cat((out_e2n_tmp[:, i, :], out_gcn_o.reshape(b*NT, 2*NO, config["feat_dims"])[:,i,:]), dim=1))

        # e2n test4
        # for i in range(NO):
        #     out_e2n[:, i, :] = out_gcn_o.reshape(b * NT, 2 * NO, config["feat_dims"])[:, i, :]

        # e2n test5
        # out_e2n = out_gcn_o.reshape(b * NT, 2 * NO, config["feat_dims"])

        # e2n_test6
        # out_e2n = out_e2n_tmp

        # e2n_test7
        out_e2n = self.e2n_embed(torch.cat((out_e2n_tmp, out_gcn_o.reshape(b * NT, 2 * NO, config["feat_dims"])), dim=2))

        return out_e2n.reshape(b, NT, 2 * NO, config["feat_dims"])

    def compute_tcn(self, out_e2n, device):

        b, NT, NO, C = out_e2n.size()
        ttmp = out_e2n.clone()

        tcn_input = out_e2n.permute(0, 2, 3, 1).reshape(b*NO, C, NT)
        mask = torch.ones(b*NO, config['num_class'], NT).to(device)

        tcn_output = self.TCN(tcn_input, mask)


        return tcn_output.reshape(config["num_stages"],b,NO,config['num_class'],NT)

    def compute_hdrg(self, out_e2n, out_tcn, device):

        b, NT, NO, C = out_e2n.size()
        b, NO, n, NT = out_tcn.size()
        NO = int(NO / 2)

        srca = torch.zeros(b, 1, 1, n*NO, C).to(device)
        Aa = torch.zeros(b, n*NO, n*NO).to(device)
        time_interval = torch.ones(b, n*NO, 2).to(device) * (-1)
        true_node = []
        for i in range(b):
            true_node.append([])

        # compute node feature and node meta
        for i in range(NO):
            for j in range(n):
                srca[:, 0, 0, i*n+j, :] = torch.sum(torch.mul(torch.round(torch.sigmoid(out_tcn[:,i,j,:])).unsqueeze(-1), out_e2n[:,:,i,:]), dim=1)

                for k in range(b):
                    if torch.nonzero(torch.round(torch.sigmoid(out_tcn[k,i,j,:]))).size()[0] != 0:
                        time_interval[k, i * n + j, 0] = torch.nonzero(torch.round(torch.sigmoid(out_tcn[k, i, j, :])))[0]
                        time_interval[k, i * n + j, 1] = torch.nonzero(torch.round(torch.sigmoid(out_tcn[k, i, j, :])))[-1]
                        true_node[k].append([i,j])

        # compute adjacent
        for i in range(b):
            for ai, ci in true_node[i]:
                for aj, cj in true_node[i]:
                    #ai, ci = ni
                    #aj, cj = nj

                    ta1 = time_interval[i, ai * n + ci]
                    ta2 = time_interval[i, aj * n + cj]

                    # same actor
                    if ai == aj:
                        #if ta1[0] != -1 and ta2[0] != -1 and max(ta1[0], ta2[0]) - min(ta1[1], ta2[1]) < 5:
                        #    Aa[i, ai * n + ci, aj * n + cj] = 1
                        ccc = 0
                    # different actor
                    else:
                        if ta1[0] != -1 and ta2[0] != -1:
                            if ta1[1] >= ta2[0] and ta2[1] >= ta1[0]:
                                Aa[i, ai * n + ci, aj * n + cj] = 1


        srca_gcn = torch.zeros(b, n*NO, C).to(device)

        for i in range(b):
            action_feat = self.gcl_a(Aa[i] + torch.eye(Aa[i].size(0)).to(Aa[i]).detach().float(), srca[i])
            action_feat = self.gcl2_a(Aa[i] + torch.eye(Aa[i].size(0)).to(Aa[i]).detach().float(), action_feat)
            srca_gcn[i] = action_feat[0][0]

        out_gcn_a = srca_gcn.reshape(b, NO, n, C)
        out_a = self.out_a(out_gcn_a)

        return out_a

    def compute_feature(self, VRD_out, device):
        NO = int(config["Max_Object"]) * int(config["Max_Object"])
        NT = int(config["Max_Time"])
        Nr = NO * NT
        b = int(VRD_out.size()[0] / Nr)

        relation_class = torch.max(VRD_out, 1)[1]
        srcr = torch.zeros(b, 1, 1, Nr, config["semantic_dim"]).to(device)


        for i in range(b):
            for j in range(Nr):
                rc = relation_class[i*Nr+j]
                if rc != 0:
                    srcr[i, 0, 0, j, :] = self.relation_feature[rc]

        return relation_class, srcr

    def read_relation_npy(self):

        CONTACT_RELATION_ID = config["CONTACT_RELATION_ID"]
        del CONTACT_RELATION_ID['']
        #{'above': 1, 'behind': 2, 'beneath': 3, 'carrying': 4, 'carrying_on_back': 5, 'covered_by': 6, 'drinking_from': 7, 'holding': 8, 'in': 9, 'in_contact': 10, 'in_front_of': 11, 'leaning_on': 12, 'looking_at': 13, 'lying_on': 14, 'not_contacting': 15, 'on_the_side_of': 16, 'pressing': 17, 'sitting_on': 18, 'standing_on': 19, 'talking_to': 20, 'wearing': 21, 'wiping': 22, 'writing_on': 23}

        semantic_feature_prefix = os.path.join(config["data_root"], 'relation_graph', 'relation_semantic_feature')

        relation_feature = np.zeros(shape=[len(CONTACT_RELATION_ID)+1, config["semantic_dim"]], dtype=np.float32)
        for rc in CONTACT_RELATION_ID:
            rf = np.load(os.path.join(semantic_feature_prefix, rc + '.npy'))
            relation_feature[CONTACT_RELATION_ID[rc]] = rf

        return torch.from_numpy(relation_feature)

class GGCN_Transformer(nn.Module):
    def __init__(self, d_model, visual_dim, target_dim, relation_dim, feat_dim, num_v, dropout=0.5):
        super(GGCN_Transformer, self).__init__()

        self.nfeat = 900
        self.gcl_o = GraphConvolution(visual_dim, 1024, num_v, dropout=dropout)
        self.gcl2_o = GraphConvolution(1024, feat_dim, num_v, dropout=dropout)
        self.gcl_r = GraphConvolution(self.nfeat, 1024, num_v, dropout=dropout)
        self.gcl2_r = GraphConvolution(1024, feat_dim, num_v, dropout=dropout)
        self.gcl_rs = GraphConvolution(self.nfeat, 1024, num_v, dropout=dropout)
        self.gcl2_rs = GraphConvolution(1024, feat_dim, num_v, dropout=dropout)
        self.loss = nn.BCELoss()
        self.spatial = nn.Linear(config["spatial_dim"], config["feat_dims"])
        self.semantic = nn.Linear(config["semantic_dim"], config["feat_dims"])
        self.visual = nn.Linear(config["visual_dim"], config["feat_dims"])


        self.trans_1 = nn.Linear(self.nfeat, self.nfeat)
        self.trans_2 = nn.Linear(self.nfeat, self.nfeat)

        encoders = nn.ModuleList([
            EncoderLayer([config['Max_Time'], d_model],
                         MultiHeadAttention(config['num_heads'], d_model),
                         PositionWiseFeedForward(d_model, config['d_ff'])),
            EncoderLayer([config['Max_Time'], d_model],
                         MultiHeadAttention(config['num_heads'], d_model),
                         PositionWiseFeedForward(d_model, config['d_ff'])),
        ])
        self.encoder = MyEncoder(encoders)

        decoders = nn.ModuleList([
            DecoderLayer([config['Max_Time'], d_model],
                         MultiHeadAttention(config['num_heads'], d_model),
                         MultiHeadAttention(config['num_heads'], d_model),
                         PositionWiseFeedForward(d_model, config['d_ff'])),
            DecoderLayer([config['Max_Time'], d_model],
                         MultiHeadAttention(config['num_heads'], d_model),
                         MultiHeadAttention(config['num_heads'], d_model),
                         PositionWiseFeedForward(d_model, config['d_ff'])),
            DecoderLayer([config['Max_Time'], d_model],
                         MultiHeadAttention(config['num_heads'], d_model),
                         MultiHeadAttention(config['num_heads'], d_model),
                         PositionWiseFeedForward(d_model, config['d_ff']))
        ])

        self.decoder = MyDecoder(decoders)
        self.i_linear = nn.ModuleList([nn.Linear(self.nfeat*(int(config["Max_Object"])-1), d_model), nn.Linear(target_dim, d_model)])
        self.o_linear = nn.ModuleList([nn.Linear(d_model, config['num_class']), nn.Linear(target_dim, 1)])
        self.pos1 = PositionEncoder(d_model, target_dim)
        self.pos2 = PositionEncoder(d_model, target_dim)

        #self.out = StandConvolution1([config["feat_dims"], config["feat_dims"] * 2, config["feat_dims"] * 4], config["num_class"], dropout)
        self.out_o = StandConvolution2([config["feat_dims"], config["feat_dims"] * 2, config["feat_dims"] * 4, config["feat_dims"] * 8], config["num_class"], dropout)
        self.out_r = StandConvolution2([config["feat_dims"], config["feat_dims"] * 2, config["feat_dims"] * 4, config["feat_dims"] * 8], config["num_class"], dropout)
        self.out_rs = StandConvolution2([config["feat_dims"], config["feat_dims"] * 2, config["feat_dims"] * 4, config["feat_dims"] * 8], config["num_class"], dropout)
        self.fc = nn.Linear(config["feat_dims"] * 8 * 4, config["num_class"])


    def forward(self, Ao, srco, Ar, srcr, tgt, device):

        NO = int(config["Max_Object"])
        NT = int(config["Max_Time"])
        NF = self.nfeat
        b, _, _, No, _ = srco.size()
        _, _, _, Nr, _ = srcr.size()
        relation_spatial_feat = srcr[:,:,:,:,:20]
        #relation_spatial_feat = self.spatial(relation_spatial_feat)
        relation_semantic_feat = srcr[:,:,:,:,20:920]
        #relation_semantic_feat = self.semantic(relation_semantic_feat)
        relation_visual_feat = srcr[:,:,:,:,920:]
        relation_visual_feat = self.visual(relation_visual_feat)

        srco_gcn = torch.zeros(b, No, config["feat_dims"]).to(device)
        srco_gcn222 = torch.zeros(b, 1, 1, No, config["feat_dims"]).to(device)
        srcr_feat = torch.zeros(b, NT, (NO-1)*NF).to(device)
        #srcr_gcns = torch.zeros(b, Nr, config["feat_dims"]).to(device)
        srcr_concat = relation_semantic_feat #torch.cat((relation_spatial_feat, relation_semantic_feat), dim=4)

        feat_box = torch.zeros(b, Nr, self.nfeat).to(device)
        for i in range(b):
            feat_box[i] = srcr_concat[i,0,0]
        x1 = self.trans_1(feat_box)
        x2 = self.trans_2(feat_box)
        g_sim = torch.bmm(x1, x2.permute(0, 2, 1))
        g_sim = F.softmax(g_sim, dim=2)

        for i in range(b):
            visual_feat = self.gcl_o(Ao[i]+torch.eye(Ao[i].size(0)).to(Ao[i]).detach().float(), srco[i])
            visual_feat = self.gcl2_o(Ao[i]+torch.eye(Ao[i].size(0)).to(Ao[i]).detach().float(), visual_feat)
            srco_gcn[i] = visual_feat[0][0]
            srco_gcn222[i] = visual_feat

            #for r in range(NO-1):
            #    for t in range(NT):
            #        srcr_concat[:,:,t*(NO-1)+r,:] = torch.cat((visual_feat[:,:,t*NO,:], visual_feat[:,:,t*NO+r+1,:], spatial_feat[i,:,:,t*(NO-1)+r,:], semantic_feat[i,:,:,t*(NO-1)+r,:]), dim=2)

        for r in range(NO-1):
            for t in range(NT):
                #srcr_concat[:,:,:,t*(NO-1)+r, :] = torch.cat((spatial_feat[:,:, :, t*(NO-1)+r, :], semantic_feat[:,:, :, t*(NO-1)+r, :], srco_gcn222[:,:,:,t*NO, :], srco_gcn222[:,:,:,t*(NO-1)+r, :]), dim=3)
                #srcr_concat[:, :, :, t * (NO - 1) + r, :] = torch.cat((srcr[:, :, :, t * (NO - 1) + r, :], srco[:, :, :, t * NO, :], srco[:, :, :, t * (NO - 1) + r, :]), dim=3)
                #srcr_feat[:, t, r * NF:(r + 1) * NF] = srcr_concat[:, 0, 0, r * NT + t, :]
                srcr_feat[:,t,r*NF:(r+1)*NF] = srcr_concat[:,0,0,t*(NO-1)+r, :]
        #for i in range(b):
        #    relation_feat = self.gcl_r(Ar[i]+torch.eye(Ar[i].size(0)).to(Ar[i]).detach().float(), srcr_concat[i])
        #    relation_feat = self.gcl2_r(Ar[i]+torch.eye(Ar[i].size(0)).to(Ar[i]).detach().float(), relation_feat)

        #    #relation_feats = self.gcl_rs(g_sim[i] + torch.eye(g_sim[i].size(0)).to(g_sim[i]).detach().float(), srcr_concat[i])
        #    #relation_feats = self.gcl2_rs(g_sim[i] + torch.eye(g_sim[i].size(0)).to(g_sim[i]).detach().float(), relation_feats)

        #    srcr_gcn[i] = relation_feat[0][0]# + relation_feat[0][0]
        #    #srcr_gcns[i] = relation_feats[0][0]


        out_gcn_o = srco_gcn.reshape(b, NO, NT, config["feat_dims"])
        out_o = self.out_o(out_gcn_o)

        target = torch.eye(NT).to(device)
        target = target.repeat(b, 1, 1)

        src = self.pos1(self.i_linear[0](srcr_feat))
        target = self.pos2(self.i_linear[1](target))
        src_mask = torch.ones(b, NT).to(device)
        tgt_mask = src_mask

        x = self.encoder(src, src_mask)
        x = self.decoder(target, x, src_mask, tgt_mask)
        out_r = x.reshape(b,-1)

        out = self.fc(torch.cat((out_o, out_r), dim=1))
        m = torch.nn.Sigmoid()
        pred = m(out)
        loss = self.loss(pred, tgt)

        return out,tgt,loss

class finetune_I3D(nn.Module):
    def __init__(self, visual_dim, relation_dim, feat_dim, num_v, dropout=0.5):
        super(finetune_I3D, self).__init__()


        self.fc = nn.Linear(config["feat_dims"] * 8, config["num_class"])
        self.loss = nn.BCELoss()


    def forward(self, Ao, srco, Ar, srcr, tgt, i3d, device):


        #I3D
        out = self.fc(i3d)
        out = torch.max(input=out, dim=1)[0]
        m = torch.nn.Sigmoid()
        pred = m(out)
        loss = self.loss(pred, tgt)

        return out,tgt,loss

class GGCN_OGRG_Transformer(nn.Module):
    def __init__(self, d_model, visual_dim, target_dim, relation_dim, feat_dim, num_v, dropout=0.5):
        super(GGCN_OGRG_Transformer, self).__init__()

        self.nfeat = 900
        self.nfeat_t = 900 #config["feat_dims"]

        self.gcl_o = GraphConvolution(visual_dim, 1024, num_v, dropout=dropout)
        self.gcl2_o = GraphConvolution(1024, feat_dim, num_v, dropout=dropout)
        self.gcl_r = GraphConvolution(self.nfeat, 1024, num_v, dropout=dropout)
        self.gcl2_r = GraphConvolution(1024, feat_dim, num_v, dropout=dropout)
        #self.gcl_rs = GraphConvolution(self.nfeat, 1024, num_v, dropout=dropout)
        #self.gcl2_rs = GraphConvolution(1024, feat_dim, num_v, dropout=dropout)
        self.loss = nn.BCELoss()
        self.spatial = nn.Linear(config["spatial_dim"], config["feat_dims"])
        self.semantic = nn.Linear(config["semantic_dim"], config["feat_dims"])
        self.visual = nn.Linear(config["visual_dim"], config["feat_dims"])

        #self.trans_1 = nn.Linear(self.nfeat_t, self.nfeat_t)
        #self.trans_2 = nn.Linear(self.nfeat_t, self.nfeat_t)

        encoders = nn.ModuleList([
            EncoderLayer([config['Max_Time'], d_model],
                         MultiHeadAttention(config['num_heads'], d_model),
                         PositionWiseFeedForward(d_model, config['d_ff'])),
            EncoderLayer([config['Max_Time'], d_model],
                         MultiHeadAttention(config['num_heads'], d_model),
                         PositionWiseFeedForward(d_model, config['d_ff'])),
        ])
        self.Trans_encoder = MyEncoder(encoders)

        decoders = nn.ModuleList([
            DecoderLayer([config['Max_Time'], d_model],
                         MultiHeadAttention(config['num_heads'], d_model),
                         MultiHeadAttention(config['num_heads'], d_model),
                         PositionWiseFeedForward(d_model, config['d_ff'])),
            DecoderLayer([config['Max_Time'], d_model],
                         MultiHeadAttention(config['num_heads'], d_model),
                         MultiHeadAttention(config['num_heads'], d_model),
                         PositionWiseFeedForward(d_model, config['d_ff'])),
            DecoderLayer([config['Max_Time'], d_model],
                         MultiHeadAttention(config['num_heads'], d_model),
                         MultiHeadAttention(config['num_heads'], d_model),
                         PositionWiseFeedForward(d_model, config['d_ff']))
        ])

        self.Trans_decoder = MyDecoder(decoders)
        self.Trans_i_linear = nn.ModuleList(
            [nn.Linear(self.nfeat_t * (int(config["Max_Object"]) - 1), d_model), nn.Linear(target_dim, d_model)])
        #self.o_linear = nn.ModuleList([nn.Linear(d_model, config['num_class']), nn.Linear(target_dim, 1)])
        self.Trans_pos1 = PositionEncoder(d_model, target_dim)
        self.Trans_pos2 = PositionEncoder(d_model, target_dim)

        #self.out = StandConvolution1([config["feat_dims"], config["feat_dims"] * 2, config["feat_dims"] * 4], config["num_class"], dropout)
        self.out_o = StandConvolution2([config["feat_dims"], config["feat_dims"] * 2, config["feat_dims"] * 4, config["feat_dims"] * 8], config["num_class"], dropout)
        self.out_r = StandConvolution2([config["feat_dims"], config["feat_dims"] * 2, config["feat_dims"] * 4, config["feat_dims"] * 8], config["num_class"], dropout)
        #self.out_rs = StandConvolution2([config["feat_dims"], config["feat_dims"] * 2, config["feat_dims"] * 4, config["feat_dims"] * 8], config["num_class"], dropout)
        #self.fc_g = nn.Linear(config["feat_dims"] * 8 * 4, config["feat_dims"] * 8)
        self.fc = nn.Linear(config["feat_dims"] * 8 * 4, config["num_class"])


    def forward(self, Ao, srco, Ar, srcr, tgt, i3d, device):

        NO = int(config["Max_Object"])
        NT = int(config["Max_Time"])
        NF = self.nfeat_t
        b, _, _, No, _ = srco.size()
        _, _, _, Nr, _ = srcr.size()
        relation_spatial_feat = srcr[:,:,:,:,:20]
        #relation_spatial_feat = self.spatial(relation_spatial_feat)
        relation_semantic_feat = srcr[:,:,:,:,20:920]
        #relation_semantic_feat = self.semantic(relation_semantic_feat)
        relation_visual_feat = srcr[:,:,:,:,920:]
        relation_visual_feat = self.visual(relation_visual_feat)

        srco_gcn = torch.zeros(b, No, config["feat_dims"]).to(device)
        srco_gcn222 = torch.zeros(b, 1, 1, No, config["feat_dims"]).to(device)
        srcr_gcn = torch.zeros(b, Nr, config["feat_dims"]).to(device)

        #srcr_gcns = torch.zeros(b, Nr, config["feat_dims"]).to(device)
        srcr_concat = torch.cat((relation_spatial_feat, relation_semantic_feat), dim=4) #torch.cat((relation_spatial_feat, relation_semantic_feat, relation_visual_feat), dim=4)

        #feat_box = torch.zeros(b, Nr, self.nfeat).to(device)
        #for i in range(b):
        #    feat_box[i] = srcr_concat[i,0,0]
        #x1 = self.trans_1(feat_box)
        #x2 = self.trans_2(feat_box)
        #g_sim = torch.bmm(x1, x2.permute(0, 2, 1))
        #g_sim = F.softmax(g_sim, dim=2)

        for i in range(b):
            visual_feat = self.gcl_o(Ao[i]+torch.eye(Ao[i].size(0)).to(Ao[i]).detach().float(), srco[i])
            visual_feat = self.gcl2_o(Ao[i]+torch.eye(Ao[i].size(0)).to(Ao[i]).detach().float(), visual_feat)
            srco_gcn[i] = visual_feat[0][0]
            srco_gcn222[i] = visual_feat

            #for r in range(NO-1):
            #    for t in range(NT):
            #        srcr_concat[:,:,t*(NO-1)+r,:] = torch.cat((visual_feat[:,:,t*NO,:], visual_feat[:,:,t*NO+r+1,:], spatial_feat[i,:,:,t*(NO-1)+r,:], semantic_feat[i,:,:,t*(NO-1)+r,:]), dim=2)

        #for r in range(NO-1):
        #    for t in range(NT):
        #        #srcr_concat[:,:,:,t*(NO-1)+r, :] = torch.cat((spatial_feat[:,:, :, t*(NO-1)+r, :], semantic_feat[:,:, :, t*(NO-1)+r, :], srco_gcn222[:,:,:,t*NO, :], srco_gcn222[:,:,:,t*(NO-1)+r, :]), dim=3)
        #        srcr_concat[:, :, :, t * (NO - 1) + r, :] = torch.cat((srcr[:, :, :, t * (NO - 1) + r, :], srco[:, :, :, t * NO, :], srco[:, :, :, t * (NO - 1) + r, :]), dim=3)
        #srcr_feat = torch.zeros(b, 1, 1, Nr, NF).to(device)
        #for r in range(NO - 1):
        #    for t in range(NT):
        #        srcr_feat[:, :, :, t*(NO-1)+r, :] = srcr_concat[:, :, :, r * NT + t, :]  # srcr_feat[:,t,r*NF:(r+1)*NF] = srcr_concat[:,0,0,t*(NO-1)+r, :]

        for i in range(b):
            relation_feat = self.gcl_r(Ar[i]+torch.eye(Ar[i].size(0)).to(Ar[i]).detach().float(), srcr_concat[i])
            relation_feat = self.gcl2_r(Ar[i]+torch.eye(Ar[i].size(0)).to(Ar[i]).detach().float(), relation_feat)

            #relation_feats = self.gcl_rs(g_sim[i] + torch.eye(g_sim[i].size(0)).to(g_sim[i]).detach().float(), srcr_concat[i])
            #relation_feats = self.gcl2_rs(g_sim[i] + torch.eye(g_sim[i].size(0)).to(g_sim[i]).detach().float(), relation_feats)

            srcr_gcn[i] = relation_feat[0][0]# + relation_feat[0][0]
            #srcr_gcns[i] = relation_feats[0][0]

        out_gcn_o = srco_gcn.reshape(b, NO, NT, config["feat_dims"])
        out_gcn_r = srcr_gcn.reshape(b, NO-1, NT, config["feat_dims"])
        out_o = self.out_o(out_gcn_o)
        out_r = self.out_r(out_gcn_r)

        srcr_feat = torch.zeros(b, NT, (NO - 1) * NF).to(device)
        for r in range(NO-1):
            for t in range(NT):
                #srcr_concat[:,:,:,t*(NO-1)+r, :] = torch.cat((spatial_feat[:,:, :, t*(NO-1)+r, :], semantic_feat[:,:, :, t*(NO-1)+r, :], srco_gcn222[:,:,:,t*NO, :], srco_gcn222[:,:,:,t*(NO-1)+r, :]), dim=3)
                #srcr_concat[:, :, :, t * (NO - 1) + r, :] = torch.cat((srcr[:, :, :, t * (NO - 1) + r, :], srco[:, :, :, t * NO, :], srco[:, :, :, t * (NO - 1) + r, :]), dim=3)

                #srcr_feat[:, t, r * NF:(r + 1) * NF] = srcr_gcn[:, r * NT + t, :]  # reverse
                #srcr_feat[:, t, r * NF:(r + 1) * NF] = srcr_gcn[:, t * (NO - 1) + r, :]
                #srcr_feat[:, t, r * NF:(r + 1) * NF] = srcr_concat[:, 0, 0, r * NT + t, :]  # reverse
                srcr_feat[:, t, r * NF:(r + 1) * NF] = srcr_concat[:, 0, 0, t * (NO - 1) + r, :]

        target = torch.eye(NT).to(device)
        target = target.repeat(b, 1, 1)

        src = self.Trans_pos1(self.Trans_i_linear[0](srcr_feat))
        target = self.Trans_pos2(self.Trans_i_linear[1](target))
        src_mask = torch.ones(b, NT).to(device)
        tgt_mask = src_mask

        x = self.Trans_encoder(src, src_mask)
        x = self.Trans_decoder(target, x, src_mask, tgt_mask)
        out_t = x.reshape(b, -1)

        # all
        #out_g = self.fc_g(torch.cat((out_o, out_r), dim=1))
        #out_g = out_g.unsqueeze(dim=1).repeat(1,30,1)
        #out = self.fc(torch.cat((out_g, i3d), dim=2))
        #out = torch.max(input=out, dim=1)[0]

        # OR+RG
        out = self.fc(torch.cat((out_o, out_r), dim=1))
        #out = self.Trans_fc(out_t)

        # I3D
        #out = self.fc(i3d)
        #out = torch.max(input=out, dim=1)[0]
        m = torch.nn.Sigmoid()
        pred = m(out)
        loss = self.loss(pred, tgt)

        return out,tgt,loss


