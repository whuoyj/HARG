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
    "dataset": "MOMA-LRG", #VRP
    "data_root": "/home/ouyangjun/workspace/MOMA/MOMA-LRG/HDRG", #

    #dataloader
    "num_class": 13, # 35
    "num_class2": 91,
    "num_class3": 45,
    "num_class4": 20,
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
    "tem_adj_concat": 'adj_concat_13',
    "rel_adj_concat": 'adj_concat_13_A_O',
    "CONTACT_RELATION_ID": {'':0, 'aiming_and_throwing':1, 'behind':2, 'beneath':3, 'blowing':4, 'carrying':5, 'carrying_on_their_back':6, 'covered_by':7, 'eating_or_drinking_from':8, 'grabbing':9, 'grabbing_from_someone':10, 'handing_over':11, 'hitting':12, 'holding':13, 'in':14, 'in_front_of':15, 'installing':16, 'jumping_off':17, 'kicking':18, 'leaning_on':19, 'lifting':20, 'looking_at':21, 'lying_on':22, 'massaging':23, 'on_the_side_of':24, 'on_top_of':25, 'picking_up_from_the_table':26, 'placing_onto_the_table':27, 'pointing_at':28, 'pouring_into':29, 'pressing':30, 'pushing':31, 'putting_on':32, 'removing':33, 'riding_on':34, 'sitting_down_on':35, 'sitting_on':36, 'standing_up_from':37, 'stepping_on':38, 'taking_off':39, 'throwing_away':40, 'touching':41, 'wearing_on_their_head':42, 'wiping':43, 'writing_on':44},

    # file
    "model_save_path": "./Models/MOMA-LRG/",
    "model_load_name1": "./Models/MOMA-LRG/model_lsset_50.pth",
    "acc_file": "./Results/MOMA-LRG/object_GCN.txt",
    }
