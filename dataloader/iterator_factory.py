import os
import numpy as np

import torch
from config import config
from .video_iterator import spatem_rel_FeatureIter


def get_spatem_rel(data_root,

                 **kwargs):
    """ feature iter for action genome prediction
    """
    train = spatem_rel_FeatureIter(visual_feature_prefix=os.path.join(data_root, 'object_features_resnet_2'), #ok
                        spatial_feature_prefix=os.path.join(data_root, 'relation_graph', 'person_object_spatial_feature'),
                        semantic_feature_prefix=os.path.join(data_root, 'object_features_semantic'), #ok
                        relation_semantic_prefix=os.path.join(data_root, 'relation_graph', 'relation_semantic_feature'), #ok
                        relation_visual_prefix=os.path.join(data_root, 'relation_graph', 'pair_features_resnet'),
                        spa_adjacent_matrix_prefix=os.path.join(data_root, 'adjacent_new', 'spa_adj', config["spa_adj_concat"]), #ok
                        tem_adjacent_matrix_prefix=os.path.join(data_root, 'adjacent_new', 'tem_adj', config["tem_adj_concat"]), #ok
                        rel_adjacent_matrix_prefix=os.path.join(data_root, 'adjacent_new', 'rel_adj', config["rel_adj_concat"]), #ok
                        dataset_level_list='dataset/dataset-level-train.txt', #ok
                        dataset_level_relation_list=os.path.join(data_root, 'relation_graph', 'relation-semantic.txt'),
                        dataset_level_relation_list_1=os.path.join(data_root, 'relation_graph', 'relation-semantic_1.txt'),
                        video_level_object_list='dataset/video-level-object/', #ok
                        video_level_relation_list='dataset/video-level-relation-A-O/', #ok
                        name='train',
                        )


    val   = spatem_rel_FeatureIter(visual_feature_prefix=os.path.join(data_root, 'object_features_resnet_2'),
                        spatial_feature_prefix=os.path.join(data_root, 'relation_graph', 'person_object_spatial_feature'),
                        semantic_feature_prefix=os.path.join(data_root, 'object_features_semantic'),
                        relation_semantic_prefix=os.path.join(data_root, 'relation_graph', 'relation_semantic_feature'),
                        relation_visual_prefix=os.path.join(data_root, 'relation_graph', 'pair_features_resnet'),
                        spa_adjacent_matrix_prefix=os.path.join(data_root, 'adjacent_new', 'spa_adj', config["spa_adj_concat"]),
                        tem_adjacent_matrix_prefix=os.path.join(data_root, 'adjacent_new', 'tem_adj', config["tem_adj_concat"]),
                        rel_adjacent_matrix_prefix=os.path.join(data_root, 'adjacent_new', 'rel_adj', config["rel_adj_concat"]),
                        dataset_level_list='dataset/dataset-level-test.txt',
                        dataset_level_relation_list=os.path.join(data_root, 'relation_graph', 'relation-semantic.txt'),
                        dataset_level_relation_list_1=os.path.join(data_root, 'relation_graph', 'relation-semantic_1.txt'),
                        video_level_object_list='dataset/video-level-object/',
                        video_level_relation_list='dataset/video-level-relation-A-O/',
                        name='test',
                        )
    return (train, val)

def get_spatem_rel_LRG(data_root,

                 **kwargs):
    """ feature iter for action genome prediction
    """
    train = spatem_rel_FeatureIter(visual_feature_prefix=os.path.join(data_root, 'object_features_resnet_2'), #ok
                        spatial_feature_prefix=os.path.join(data_root, 'relation_graph', 'person_object_spatial_feature'), #no need
                        semantic_feature_prefix=os.path.join(data_root, 'object_features_semantic'), #ok
                        relation_semantic_prefix=os.path.join(data_root, 'relation_graph', 'relation_semantic_feature'), #ok
                        relation_visual_prefix=os.path.join(data_root, 'relation_graph', 'pair_features_resnet'), #no need
                        spa_adjacent_matrix_prefix=os.path.join(data_root, 'adjacent_new', 'spa_adj', config["spa_adj_concat"]), #ok
                        tem_adjacent_matrix_prefix=os.path.join(data_root, 'adjacent_new', 'tem_adj', config["tem_adj_concat"]), #ok
                        rel_adjacent_matrix_prefix=os.path.join(data_root, 'adjacent_new', 'rel_adj', config["rel_adj_concat"]), #ok
                        dataset_level_list='dataset-LRG/dataset-level-train.txt', #ok
                        dataset_level_relation_list=os.path.join(data_root, 'relation_graph', 'relation-semantic.txt'), #ok
                        dataset_level_relation_list_1=os.path.join(data_root, 'relation_graph', 'relation-semantic_1.txt'), #ok
                        video_level_object_list='dataset-LRG/video-level-object/', #ok
                        video_level_relation_list='dataset-LRG/video-level-relation-A-O/', #ok
                        name='train',
                        )


    #val   = spatem_rel_FeatureIter(visual_feature_prefix=os.path.join(data_root, 'object_features_resnet_2'),
    #                    spatial_feature_prefix=os.path.join(data_root, 'relation_graph', 'person_object_spatial_feature'),
    #                    semantic_feature_prefix=os.path.join(data_root, 'object_features_semantic'),
    #                    relation_semantic_prefix=os.path.join(data_root, 'relation_graph', 'relation_semantic_feature'),
    #                    relation_visual_prefix=os.path.join(data_root, 'relation_graph', 'pair_features_resnet'),
    #                    spa_adjacent_matrix_prefix=os.path.join(data_root, 'adjacent_new', 'spa_adj', config["spa_adj_concat"]),
    #                    tem_adjacent_matrix_prefix=os.path.join(data_root, 'adjacent_new', 'tem_adj', config["tem_adj_concat"]),
    #                    rel_adjacent_matrix_prefix=os.path.join(data_root, 'adjacent_new', 'rel_adj', config["rel_adj_concat"]),
    #                    dataset_level_list='dataset-LRG/dataset-level-val.txt',
    #                    dataset_level_relation_list=os.path.join(data_root, 'relation_graph', 'relation-semantic.txt'),
    #                    dataset_level_relation_list_1=os.path.join(data_root, 'relation_graph', 'relation-semantic_1.txt'),
    #                    video_level_object_list='dataset-LRG/video-level-object/',
    #                    video_level_relation_list='dataset-LRG/video-level-relation-A-O/',
    #                    name='val',
    #                    )

    val = spatem_rel_FeatureIter(visual_feature_prefix=os.path.join(data_root, 'object_features_resnet_2'),
                        spatial_feature_prefix=os.path.join(data_root, 'relation_graph',  'person_object_spatial_feature'),
                        semantic_feature_prefix=os.path.join(data_root, 'object_features_semantic'),
                        relation_semantic_prefix=os.path.join(data_root, 'relation_graph', 'relation_semantic_feature'),
                        relation_visual_prefix=os.path.join(data_root, 'relation_graph', 'pair_features_resnet'),
                        spa_adjacent_matrix_prefix=os.path.join(data_root, 'adjacent_new', 'spa_adj', config["spa_adj_concat"]),
                        tem_adjacent_matrix_prefix=os.path.join(data_root, 'adjacent_new', 'tem_adj', config["tem_adj_concat"]),
                        rel_adjacent_matrix_prefix=os.path.join(data_root, 'adjacent_new', 'rel_adj', config["rel_adj_concat"]),
                        dataset_level_list='dataset-LRG/dataset-level-test.txt',
                        dataset_level_relation_list=os.path.join(data_root, 'relation_graph', 'relation-semantic.txt'),
                        dataset_level_relation_list_1=os.path.join(data_root, 'relation_graph', 'relation-semantic_1.txt'),
                        video_level_object_list='dataset-LRG/video-level-object/',
                        video_level_relation_list='dataset-LRG/video-level-relation-A-O/',
                        name='test',
                        )
    return (train, val)

def creat(batch_size=1, data_root='', num_workers=0, **kwargs):

    if config["dataset"] == "MOMA-LRG":
        train, val = get_spatem_rel_LRG(data_root, **kwargs)
    else:
        train, val = get_spatem_rel(data_root, **kwargs)

    train_loader = torch.utils.data.DataLoader(train,
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        drop_last=True)

    val_loader = torch.utils.data.DataLoader(val,
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True)


    return (train_loader, val_loader)