config = {
    # training
    "cuda": 'cuda:0',
    "learning_rate": 5e-5,
    "display_batch_interval": 50,
    "max_epoches": 50,
    "batch_size": 4,
    "weight_decay_step": 10,
    "weight_decay_rate": 0.2,

    "dropout_prob": 0.9,
    "dropout_rate": 0.0,

    # data
    "dataset": "MOMA",
    "data_root": "/home/ouyangjun/workspace/MOMA/MOMA-1.0/HDRG",

    #dataloader
    "num_class": 52,
    "num_class2": 68,
    "num_class3": 24,
    "num_class4": 17,
    "Max_Object": 8,
    "Max_Time": 32,
    "TrainObjectExtractMannel": 'sortedhalf', #random, sorted, sortedhalf
    "ValObjectExtractMannel": 'sorted', #random, sorted, sortedhalf
    "TrainFrameExtractMannel": 'random', #random, continuous, split
    "ValFrameExtractMannel": 'average', #random, continuous, split

    # model
    "strict": False,
    "non_zero_acc": True,
    "mode": "val", #
    "models": "HARG",
    "num_nodes": 3,
    "num_stages": 4,
    "num_layers": 8,

    # feature
    "input_video_dim": 2348,
    "input_relation_dim": 300,
    "feat_dims": 256,
    "visual_dim": 2348,
    "semantic_dim": 300,
    "spatial_dim": 20,
    "adj": 'spatem_rel',
    "spa_adj_concat": 'relation_concat_A_O',
    "tem_adj_concat": 'adj_concat_15',
    "rel_adj_concat": 'adj_concat_15_A_O',
    "CONTACT_RELATION_ID": {'':0, 'above':1, 'behind':2, 'beneath':3, 'carrying':4, 'carrying_on_back':5, 'covered_by':6, 'drinking_from':7, 'holding':8, 'in':9, 'in_contact':10, 'in_front_of':11, 'leaning_on':12, 'looking_at':13, 'lying_on':14, 'not_contacting':15, 'on_the_side_of':16, 'pressing':17, 'sitting_on':18, 'standing_on':19, 'talking_to':20, 'wearing':21, 'wiping':22, 'writing_on':23},

    # file
    "model_save_path": "./Models/MOMA/",
    "model_load_name1": "./Models/MOMA/model_lsset_50.pth",
    "acc_file": "./Results/MOMA/object_GCN.txt",

}
