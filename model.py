import os
from layer import *
from config import config
from multi_head_attention import MultiHeadAttention
from MultiStageTCN import MultiStageModel
import torch.nn.functional as F


class VRD(nn.Module):
    def __init__(self, visual_dim, target_dim, feat_dim, dropout=0.5):
        super(VRD, self).__init__()

        self.target_dim = target_dim

        self.vis_hid = config["feat_dims"]
        self.sem_hid = config["feat_dims"]

        self.fc_vis = MLP(2048, self.vis_hid, self.vis_hid)
        self.fc_fusion = FC(self.vis_hid, config["feat_dims"])
        self.fc_rel = FC(config["feat_dims"]*2, target_dim)


    def forward(self, src, tgt_rel, device):

        NO = int(config["Max_Object"])
        NT = int(config["Max_Time"])
        b, _, _, N, _ = src.size()

        src = src[:, 0, 0, :, :].reshape(b * NT, NO * 2, config['input_video_dim'])

        x_v = self.fc_vis(src[:,:,:2048])
        node_feats = x_v
        node_feats = self.fc_fusion(node_feats)

        edge_feats = torch.zeros(b * NT, NO * NO, config["feat_dims"] * 2).to(device)
        for i in range(NO):
            for j in range(NO, 2 * NO):
                edge_feats[:, i*NO+j-NO, :] = torch.cat([node_feats[:, i, :], node_feats[:, j, :]], dim=1)

        output = self.fc_rel(edge_feats)

        output = output.reshape(-1, self.target_dim)
        tgt_rel = tgt_rel.reshape(-1).long()
        w = torch.FloatTensor([1, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]).to(device)

        if config["mode"] == "VRD":
            loss_vrd = F.cross_entropy(output, tgt_rel, weight=w)
            loss = loss_vrd
        else:
            loss = 0
        out = output
        tgt = tgt_rel

        return out,tgt,loss

class GGCN_relation(nn.Module):
    def __init__(self, visual_dim, relation_dim, feat_dim, num_v, dropout=0.5):
        super(GGCN_relation, self).__init__()

        '''self.VRD = VRD(visual_dim=visual_dim,
                       target_dim=relation_dim,
                       feat_dim=feat_dim,
                       dropout=dropout)'''
        self.relation_feature = self.read_relation_npy()

        self.nfeat = 300
        self.visual_dim = visual_dim
        self.headnum = 4
        self.dropout = nn.Dropout(0.0)

        self.gcl_o = GraphConvolution(visual_dim, 1024, dropout=dropout)
        self.gcl2_o = GraphConvolution(1024, feat_dim, dropout=dropout)
        self.gcl_r = GraphConvolution(self.nfeat, 1024, dropout=dropout)
        self.gcl2_r = GraphConvolution(1024, feat_dim, dropout=dropout)

        self.out_o = StandConvolution2([config["feat_dims"], config["feat_dims"] * 2, config["feat_dims"] * 4, config["feat_dims"] * 8], config["num_class"], dropout)
        self.out_r = StandConvolution2_RG([config["feat_dims"], config["feat_dims"] * 2, config["feat_dims"] * 4, config["feat_dims"] * 8], config["num_class"], dropout)
        self.e2n_out_o2 = StandConvolution2([config["feat_dims"], config["feat_dims"] * 2, config["feat_dims"] * 4, config["feat_dims"] * 8], config["num_class"], dropout)
        self.out_a = StandConvolution2_HARG([config["feat_dims"], config["feat_dims"] * 2, config["feat_dims"] * 4, config["feat_dims"] * 8], config["num_class"], dropout)

        self.fc1 = nn.Linear(config["feat_dims"] * 8 * 6, config["num_class"])
        self.fc2 = nn.Linear(config["feat_dims"] * 8 * 8, config["num_class2"])
        self.fc3 = nn.Linear(config["feat_dims"] * 8 * 8, config["num_class4"])

        self.e2n = MultiHeadAttention(config["feat_dims"], self.headnum)
        self.e2n_embed = nn.Linear(config["feat_dims"] * 2, config["feat_dims"])
        self.TCN = MultiStageModel(config["num_stages"], config["num_layers"], config["feat_dims"], config["feat_dims"], config["num_class"])

        self.gcl_a = GraphConvolution(feat_dim, feat_dim, dropout=dropout)
        self.gcl2_a = GraphConvolution(feat_dim, feat_dim, dropout=dropout)


    def forward(self, Ao, srco, Ar, srcr, tgt1, tgt2, tgt3, tgt4, device):


        NO = int(config["Max_Object"])
        NT = int(config["Max_Time"])
        b, _, _, No, _ = srco.size()
        _, _, _, Nr, _ = srcr.size()

        srco_gcn = torch.zeros(b, No, config["feat_dims"]).to(device)
        srco = srco[:, :, :, :, :self.visual_dim]

        for i in range(b):
            visual_feat = self.gcl_o(Ao[i] + torch.eye(Ao[i].size(0)).to(Ao[i]).detach().float(), srco[i])
            visual_feat = self.gcl2_o(Ao[i] + torch.eye(Ao[i].size(0)).to(Ao[i]).detach().float(), visual_feat)
            srco_gcn[i] = visual_feat[0][0]


        out_gcn_o = srco_gcn.reshape(b, NT, 2 * NO, config["feat_dims"])
        out_o = self.out_o(out_gcn_o)

        srcr_gcn = torch.zeros(b, Nr, config["feat_dims"]).to(device)
        srcr_concat = srcr

        for i in range(b):
            relation_feat = self.gcl_r(Ar[i] + torch.eye(Ar[i].size(0)).to(Ar[i]).detach().float(), srcr_concat[i])
            relation_feat = self.gcl2_r(Ar[i] + torch.eye(Ar[i].size(0)).to(Ar[i]).detach().float(), relation_feat)

            srcr_gcn[i] = relation_feat[0][0]

        out_gcn_r = srcr_gcn.reshape(b, NT, NO * NO, config["feat_dims"])
        out_r = self.out_r(out_gcn_r)

        out_e2n = self.compute_e2n(out_gcn_o, out_gcn_r, device)
        out_tcn = self.compute_tcn(out_e2n, device)
        out_harg = self.compute_harg(out_e2n[:, :, :NO, :], out_tcn[-1][:, :NO, :, :], device)
        out1 = self.fc1(self.dropout(torch.cat((out_o, out_r, self.e2n_out_o2(out_e2n)), dim=1)))
        out2 = self.fc2(self.dropout(torch.cat((out_o, out_r, self.e2n_out_o2(out_e2n), out_harg), dim=1)))
        out3 = self.fc3(self.dropout(torch.cat((out_o, out_r, self.e2n_out_o2(out_e2n), out_harg), dim=1)))

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

        out_e2n = self.e2n_embed(torch.cat((out_e2n_tmp, out_gcn_o.reshape(b * NT, 2 * NO, config["feat_dims"])), dim=2))

        return out_e2n.reshape(b, NT, 2 * NO, config["feat_dims"])

    def compute_tcn(self, out_e2n, device):

        b, NT, NO, C = out_e2n.size()

        tcn_input = out_e2n.permute(0, 2, 3, 1).reshape(b*NO, C, NT)
        mask = torch.ones(b*NO, config['num_class'], NT).to(device)

        tcn_output = self.TCN(tcn_input, mask)


        return tcn_output.reshape(config["num_stages"],b,NO,config['num_class'],NT)

    def compute_harg(self, out_e2n, out_tcn, device):
        b, NT, NO, C = out_e2n.size()
        b, NO, n, NT = out_tcn.size()

        sigmoid_out_tcn = torch.sigmoid(out_tcn)
        rounded_out_tcn = torch.round(sigmoid_out_tcn)

        non_zero_indices = torch.nonzero(rounded_out_tcn)

        true_node = [[] for _ in range(b)]
        for indices in non_zero_indices:
            batch_idx, node_idx, j_idx, _ = indices
            if ([node_idx.item(), j_idx.item()]) not in true_node[batch_idx.item()]:
                true_node[batch_idx.item()].append([node_idx.item(), j_idx.item()])
                
        out_tcn_rounded = rounded_out_tcn.unsqueeze(-1)
        out_e2n_reshaped = out_e2n.permute(0, 2, 1, 3).unsqueeze(2)  # [b, NO, 1, NT, C]
        srca = torch.sum(out_tcn_rounded * out_e2n_reshaped, dim=3)  # [b, NO, n, C]
        srca = srca.view(b, 1, 1, NO * n, C)

        tcn_activation = rounded_out_tcn.bool()

        time_interval = torch.full((b, NO * n, 2), -1, device=device, dtype=torch.long)

        for k in range(b):
            first_activation = (tcn_activation[k].cumsum(dim=-1) == 1).long().argmax(dim=-1)
            last_activation = (tcn_activation[k].flip(dims=(-1,)).cumsum(dim=-1) == 1).long().argmax(dim=-1)
            last_activation = NT - 1 - last_activation
            
            total_activation = tcn_activation[k].sum(dim=-1)
            first_activation[total_activation == 0] = -1
            last_activation[total_activation == 0] = -1

            time_interval[k, :, 0] = first_activation.view(-1)
            time_interval[k, :, 1] = last_activation.view(-1)

        Aa = torch.zeros(b, n * NO, n * NO).to(device)
        # compute adjacent
        node_pairs = []
        for i in range(b):
            node_pairs.append([(ai, ci, aj, cj) for ai, ci in true_node[i] for aj, cj in true_node[i]])

        for i in range(b):
            pairs = node_pairs[i]
            if pairs:
                ai, ci, aj, cj = zip(*pairs)
                ai = torch.tensor(ai, device=device)
                ci = torch.tensor(ci, device=device)
                aj = torch.tensor(aj, device=device)
                cj = torch.tensor(cj, device=device)

                ta1 = time_interval[i, ai * n + ci]
                ta2 = time_interval[i, aj * n + cj]

                same_actor = ai == aj
                time_diff = torch.where(same_actor, torch.clamp_max(ta1[:, 0] - ta2[:, 1], 5), torch.zeros_like(ta1[:, 0]))

                same_actor_condition = same_actor & (ta1[:, 0] != -1) & (ta2[:, 0] != -1) & (time_diff < 5)

                different_actor_condition = (~same_actor) & (ta1[:, 0] != -1) & (ta2[:, 0] != -1) & (ta1[:, 1] >= ta2[:, 0]) & (ta2[:, 1] >= ta1[:, 0])

                Aa[i, ai * n + ci, aj * n + cj] = (same_actor_condition | different_actor_condition).float()

        srca_gcn = torch.zeros(b, n * NO, C).to(device)

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

        semantic_feature_prefix = os.path.join(config["data_root"], 'relation_graph', 'relation_semantic_feature')

        relation_feature = np.zeros(shape=[len(CONTACT_RELATION_ID)+1, config["semantic_dim"]], dtype=np.float32)
        for rc in CONTACT_RELATION_ID:
            rf = np.load(os.path.join(semantic_feature_prefix, rc + '.npy'))
            relation_feature[CONTACT_RELATION_ID[rc]] = rf

        return torch.from_numpy(relation_feature)



