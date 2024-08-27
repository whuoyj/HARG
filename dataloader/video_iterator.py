# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
np.set_printoptions(suppress=True)
import csv
import gc
import random
from config import config

import torch
import torch.utils.data as data
from torch.autograd import Variable

CONTACT_RELATION_ID = config["CONTACT_RELATION_ID"]


class spatem_rel_FeatureIter(data.Dataset):

    def __init__(self,
                 visual_feature_prefix,
                 spatial_feature_prefix,
                 semantic_feature_prefix,
                 relation_semantic_prefix,
                 relation_visual_prefix,
                 spa_adjacent_matrix_prefix,
                 tem_adjacent_matrix_prefix,
                 rel_adjacent_matrix_prefix,
                 dataset_level_list,
                 dataset_level_relation_list,
                 dataset_level_relation_list_1,
                 video_level_object_list,
                 video_level_relation_list,
                 name="<NO_NAME>"):
        super(spatem_rel_FeatureIter, self).__init__()

        self.ClassNum1 = config["num_class"]
        self.ClassNum2 = config["num_class2"]
        self.MaxObject = config["Max_Object"]
        self.MaxTime = config["Max_Time"]
        if name == 'train':
            self.ObjectExtractMannel = config["TrainObjectExtractMannel"]
        else:
            self.ObjectExtractMannel = config["ValObjectExtractMannel"]
        if name == 'train':
            self.FrameExtractMannel = config["TrainFrameExtractMannel"]
        else:
            self.FrameExtractMannel = config["ValFrameExtractMannel"]
        self.FeatureChannel = config["input_video_dim"]
        self.RelationFeatureChannel = config["input_relation_dim"]
        self.num_result, self.actor_result, self.object_result, self.frame_result = self.read_label_csv(dataset_level_list)
        self.video_name = []
        self.visual_feature_path = []
        self.label_result1 = []  # atomic_action
        self.label_result2 = []  # sub_activity
        self.label_result3 = []  # temporal_atomic_action
        self.label_result4 = []  # activity
        self.video_level_object_list = video_level_object_list
        self.video_level_relation_list = video_level_relation_list
        self.spa_adjacent_matrix_prefix = spa_adjacent_matrix_prefix
        self.tem_adjacent_matrix_prefix = tem_adjacent_matrix_prefix
        self.rel_adjacent_matrix_prefix = rel_adjacent_matrix_prefix
        self.semantic_feature_prefix = semantic_feature_prefix
        self.spatial_feature_prefix = spatial_feature_prefix
        self.relation_visual_prefix = relation_visual_prefix
        self.relation_semantic_prefix = relation_semantic_prefix
        self.relation_semantic = self.read_relation_semantic(dataset_level_relation_list, self.num_result)
        self.relation_semantic_1 = self.read_relation_semantic(dataset_level_relation_list_1, self.num_result)

        with open(dataset_level_list) as input_file:
            lines = input_file.readlines()
            for num,line in enumerate(lines):
                self.video_name.append(line.strip().split(',')[0])
                self.visual_feature_path.append(os.path.join(visual_feature_prefix, line.strip().split(',')[0]))
                self.label_result1.append(line.strip().split(',')[-3])  # atomic_action
                self.label_result2.append(line.strip().split(',')[-2])  # sub_activity
                self.label_result3.append(line.strip().split(',')[-4])  # temporal_atomic_action
                self.label_result4.append(line.strip().split(',')[-1])  # activity




    def __getitem__(self, index):

        video_name = self.video_name[index]
        NA = int(self.num_result[video_name][0])
        NO = int(self.num_result[video_name][1])
        NT = int(self.num_result[video_name][2])
        actor_name = self.actor_result[video_name]
        object_name = self.object_result[video_name]
        object_name = actor_name + object_name
        frame_name = self.frame_result[video_name]

        video_level_object_path = os.path.join(self.video_level_object_list, os.path.splitext(video_name)[0] + '.npy')
        video_level_relation_path = os.path.join(self.video_level_relation_list,
                                                 os.path.splitext(video_name)[0] + '.npy')
        ob_in_video = np.load(video_level_object_path)
        rel_in_video = np.load(video_level_relation_path)

        spa_adjacent_path = os.path.join(self.spa_adjacent_matrix_prefix, os.path.splitext(video_name)[0] + '.npy')
        tem_adjacent_path = os.path.join(self.tem_adjacent_matrix_prefix, os.path.splitext(video_name)[0] + '.npy')
        rel_adjacent_path = os.path.join(self.rel_adjacent_matrix_prefix, os.path.splitext(video_name)[0] + '.npy')

        visual_feature_path = self.visual_feature_path[index]
        video_feature = np.zeros(shape=[self.MaxObject * 2, self.MaxTime, self.FeatureChannel], dtype=np.float32)
        relation_feature = np.zeros(shape=[self.MaxObject * self.MaxObject, self.MaxTime, self.RelationFeatureChannel],
                                    dtype=np.float32)
        relation_label_result = np.zeros(shape=(self.MaxTime, self.MaxObject * self.MaxObject), dtype=np.float32)

        object_map_graph = np.zeros(shape=self.MaxObject * 2, dtype=int)
        if NA <= self.MaxObject:
            object_map_graph[:NA] = range(NA)

            for i in range(NA, self.MaxObject):
                object_map_graph[i] = -1
        else:
            if self.ObjectExtractMannel == 'random':
                resultList = random.sample(range(NA), self.MaxObject)
                object_map_graph[:self.MaxObject] = resultList
            elif self.ObjectExtractMannel == 'sorted':
                ob_sum = ob_in_video[:NA, :].sum(axis=1)
                sortIndex = np.argsort(-ob_sum)
                for i in range(0, self.MaxObject):
                    object_map_graph[i] = sortIndex[i]
            else:
                ob_sum = ob_in_video[:NA, :].sum(axis=1)
                sortIndex = np.argsort(-ob_sum)
                sortIndex = (np.array(sortIndex)).tolist()
                for i in range(0, int(self.MaxObject / 2)):
                    object_map_graph[i] = sortIndex[i]
                resultList = random.sample(list(sortIndex[int(self.MaxObject / 2):]), int(self.MaxObject / 2))
                object_map_graph[int(self.MaxObject / 2):self.MaxObject] = resultList

        if NO <= self.MaxObject:
            object_map_graph[self.MaxObject:self.MaxObject + NO] = range(NA, NA + NO)

            for i in range(self.MaxObject + NO, self.MaxObject * 2):
                object_map_graph[i] = -1
        else:
            if self.ObjectExtractMannel == 'random':
                resultList = random.sample(range(NA, NA + NO), self.MaxObject)
                object_map_graph[self.MaxObject:] = resultList
            elif self.ObjectExtractMannel == 'sorted':
                ob_sum = ob_in_video[NA:, :].sum(axis=1)
                sortIndex = np.argsort(-ob_sum)
                sortIndex = (np.array(sortIndex + NA)).tolist()
                for i in range(0, self.MaxObject):
                    object_map_graph[self.MaxObject + i] = sortIndex[i]
            else:
                ob_sum = ob_in_video[NA:, :].sum(axis=1)
                sortIndex = np.argsort(-ob_sum)
                sortIndex = (np.array(sortIndex + NA)).tolist()
                for i in range(0, int(self.MaxObject / 2)):
                    object_map_graph[self.MaxObject + i] = sortIndex[i]
                resultList = random.sample(list(sortIndex[int(self.MaxObject / 2):]), int(self.MaxObject / 2))
                object_map_graph[int(self.MaxObject * 3 / 2):self.MaxObject * 2] = resultList

        frame_map_graph = np.zeros(shape=self.MaxTime, dtype=int)
        if NT <= self.MaxTime:
            frame_map_graph[:NT] = range(NT)
            for i in range(NT, self.MaxTime):
                frame_map_graph[i] = -1
        else:
            if self.FrameExtractMannel == 'random':
                resultList = random.sample(range(0, NT), self.MaxTime)
                frame_map_graph = sorted(resultList)
            elif self.FrameExtractMannel == 'continuous':
                resultList = random.sample(range(0, NT - self.MaxTime), 1)
                frame_map_graph = range(resultList[0], resultList[0] + self.MaxTime)
            elif self.FrameExtractMannel == 'average':
                resultList = np.linspace(0, NT - 1, self.MaxTime, endpoint=False, dtype=int)
                frame_map_graph = sorted(resultList)
            else:
                resultList1 = random.sample(range(0, int(NT / 4)), self.MaxTime // 4)
                resultList2 = random.sample(range(int(NT / 4), int(NT / 4) * 2), self.MaxTime // 4)
                resultList3 = random.sample(range(int(NT / 4) * 2, int(NT / 4) * 3), self.MaxTime // 4)
                resultList4 = random.sample(range(int(NT / 4) * 3, NT), self.MaxTime // 4)
                frame_map_graph = sorted(resultList1 + resultList2 + resultList3 + resultList4)

        self.MaxObject = self.MaxObject * 2
        for gi in range(self.MaxObject):
            for gj in range(self.MaxTime):
                i = object_map_graph[gi]
                j = frame_map_graph[gj]
                if i != -1 and j != -1:
                    if ob_in_video[i][j] == 1:
                        object_feature = np.load(os.path.join(visual_feature_path,
                                                              frame_name[j] + '-' + str(i) + '-' + object_name[
                                                                  i] + '.npy'))
                        object_semantic_embed = np.load(
                            os.path.join(self.semantic_feature_prefix, object_name[i] + '.npy'))
                        video_feature[gi][gj] = np.concatenate((object_feature, object_semantic_embed))
        for gi in range(int(self.MaxObject / 2)):
            for gj in range(int(self.MaxObject / 2), self.MaxObject):
                for gk in range(self.MaxTime):
                    i = object_map_graph[gi]
                    j = object_map_graph[gj]
                    k = frame_map_graph[gk]

                    if i != -1 and j != -1 and k != -1:
                        if rel_in_video[i * NO + j - NA][k] == 0:
                            if not self.relation_semantic[video_name][(i * NO + j - NA) * NT + k].split(';')[
                                       0] == '' and \
                                    self.relation_semantic[video_name][(i * NO + j - NA) * NT + k].split(';')[
                                        1] == '' and \
                                    self.relation_semantic[video_name][(i * NO + j - NA) * NT + k].split(';')[2] == '':
                                print('relation-semantic.txt is error!!!')

                        if rel_in_video[i * NO + j - NA][k] == 1 and self.RelationFeatureChannel == 900:

                            if self.relation_semantic[video_name][(i * NO + j - NA) * NT + k].split(';')[0] != '':
                                relation_semantic_embed1 = np.load(os.path.join(self.relation_semantic_prefix,
                                                                                self.relation_semantic[video_name][
                                                                                    (i * NO + j - NA) * NT + k].split(
                                                                                    ';')[0] + '.npy'))
                            else:
                                relation_semantic_embed1 = np.zeros(shape=[300], dtype=np.float32)

                            if self.relation_semantic[video_name][(i * NO + j - NA) * NT + k].split(';')[1] != '':
                                relation_semantic_embed2 = np.load(os.path.join(self.relation_semantic_prefix,
                                                                                self.relation_semantic[video_name][
                                                                                    (i * NO + j - NA) * NT + k].split(
                                                                                    ';')[1] + '.npy'))
                            else:
                                relation_semantic_embed2 = np.zeros(shape=[300], dtype=np.float32)

                            if self.relation_semantic[video_name][(i * NO + j - NA) * NT + k].split(';')[2] != '':
                                relation_semantic_embed3 = np.load(os.path.join(self.relation_semantic_prefix,
                                                                                self.relation_semantic[video_name][
                                                                                    (i * NO + j - NA) * NT + k].split(
                                                                                    ';')[2] + '.npy'))
                            else:
                                relation_semantic_embed3 = np.zeros(shape=[300], dtype=np.float32)

                            relation_feature[gi * int(self.MaxObject / 2) + gj - int(self.MaxObject / 2)][
                                gk] = np.concatenate((relation_semantic_embed1, relation_semantic_embed2,
                                                      relation_semantic_embed3))

                            relation_class = self.relation_semantic_1[video_name][(i * NO + j - NA) * NT + k]
                            relation_label_result[gk][gi * int(self.MaxObject / 2) + gj - int(self.MaxObject / 2)] = \
                            CONTACT_RELATION_ID[relation_class]

                        if rel_in_video[i * NO + j - NA][k] == 1 and self.RelationFeatureChannel == 300:
                            relation_semantic_embed1 = np.load(os.path.join(self.relation_semantic_prefix,
                                                                            self.relation_semantic_1[video_name][
                                                                                (i * NO + j - NA) * NT + k] + '.npy'))
                            relation_feature[gi * int(self.MaxObject / 2) + gj - int(self.MaxObject / 2)][
                                gk] = relation_semantic_embed1

                            relation_class = self.relation_semantic_1[video_name][(i * NO + j - NA) * NT + k]
                            relation_label_result[gk][gi * int(self.MaxObject / 2) + gj - int(self.MaxObject / 2)] = \
                            CONTACT_RELATION_ID[relation_class]

        video_feature = video_feature.reshape((self.MaxObject * self.MaxTime, self.FeatureChannel), order='F')
        video_feature = video_feature[np.newaxis, np.newaxis, :]
        relation_feature = relation_feature.reshape(
            (int(self.MaxObject * self.MaxObject / 4) * self.MaxTime, self.RelationFeatureChannel), order='F')
        relation_feature = relation_feature[np.newaxis, np.newaxis, :]

        graph_A = np.zeros(shape=[self.MaxTime * self.MaxObject, self.MaxTime * self.MaxObject], dtype=int)

        spa_e = np.load(spa_adjacent_path).T
        select_node = {}
        for num, t in enumerate(frame_map_graph):
            if t != -1:
                for gi in range(self.MaxObject):
                    i = object_map_graph[gi]
                    if i != -1:
                        select_node[t*(NA+NO)+i] = num * self.MaxObject + gi

        e = []
        for i in range(len(spa_e)):
            a = spa_e[i][0]
            b = spa_e[i][1]
            if a in select_node and b in select_node:
                e.append([select_node[a], select_node[b]])

        if len(e) != 0:
            graph_A[np.array(e).T[0], np.array(e).T[1]] = 1
            graph_A[np.array(e).T[1], np.array(e).T[0]] = 1
        

        tem_e = np.load(tem_adjacent_path).T
        select_node = {}
        for num, o in enumerate(object_map_graph):
            if o != -1:
                for gi in range(self.MaxTime):
                    i = frame_map_graph[gi]
                    if i != -1:
                        select_node[o*NT+i] = gi * self.MaxObject + num

        e = []
        for i in range(len(tem_e)):
            a = tem_e[i][0]
            b = tem_e[i][1]
            if a in select_node and b in select_node:
                e.append([select_node[a], select_node[b]])

        if len(e) != 0:
            graph_A[np.array(e).T[0], np.array(e).T[1]] = 1
            graph_A[np.array(e).T[1], np.array(e).T[0]] = 1

        self.MaxObject = int(self.MaxObject / 2)
        rel_graph_A = np.zeros(shape=[self.MaxTime * self.MaxObject * self.MaxObject, self.MaxTime * self.MaxObject * self.MaxObject], dtype=int)

        rel_e = np.load(rel_adjacent_path).T
        select_node = {}
        for num_A, a in enumerate(object_map_graph[:self.MaxObject]):
            for num_O, o in enumerate(object_map_graph[self.MaxObject:]):
                if a != -1 and o != -1:
                    rel_num = num_A * self.MaxObject + num_O
                    rel_o = a * NO + o - NA
                    for gi in range(self.MaxTime):
                        i = frame_map_graph[gi]
                        if i != -1:
                            select_node[rel_o * NT + i] = gi * self.MaxObject * self.MaxObject + rel_num

        e = []
        for i in range(len(rel_e)):
            a = rel_e[i][0]
            b = rel_e[i][1]
            if a in select_node and b in select_node:
                e.append([select_node[a], select_node[b]])

        if len(e) != 0:
            rel_graph_A[np.array(e).T[0], np.array(e).T[1]] = 1
            rel_graph_A[np.array(e).T[1], np.array(e).T[0]] = 1

        label_result1 = np.zeros(shape=self.ClassNum1, dtype=np.float32)
        if self.label_result1[index] != '':
            for label in self.label_result1[index].split(';'):
                label_result1[int(label)] = 1

        label_result2 = np.zeros(shape=1, dtype=np.float32)
        if self.label_result2[index] != '':
            for label in self.label_result2[index].split(';'):
                label_result2 = int(label)

        label_result3 = np.zeros(shape=[self.MaxTime, self.ClassNum1], dtype=np.float32)
        if self.label_result3[index] != '':
            tlabel3 = self.label_result3[index].split(';')
            for num, t in enumerate(frame_map_graph):
                if t != -1 and tlabel3[t] != '':
                    for label in tlabel3[t].split('*'):
                        label_result3[num][int(label)] = 1

        label_result4 = np.zeros(shape=1, dtype=np.float32)
        if self.label_result4[index] != '':
            for label in self.label_result4[index].split(';'):
                label_result4 = int(label)

        return graph_A, video_feature, rel_graph_A, relation_feature, label_result1, label_result2, label_result4, label_result3, video_name

    def __len__(self):
        return len(self.video_name)

    def read_label_csv(self, csv_path):
        num_result = {}
        actor_result = {}
        object_result = {}
        frame_result = {}
        with open(csv_path, 'r') as op:
            csv_data = csv.reader(op, delimiter=',')
            for row in csv_data:
                num_result[row[0]] = [row[1], row[2], row[3]]
                actor_result[row[0]] = row[4].split(';')
                object_result[row[0]] = row[5].split(';')
                frame_result[row[0]] = row[6].split(';')
        op.close()
        return num_result, actor_result, object_result, frame_result

    def read_relation_semantic(self, csv_path, num_result):
        relation_result = {}
        with open(csv_path, 'r') as op:
            csv_data = csv.reader(op, delimiter=',')
            for row in csv_data:
                if row[0] in num_result:
                    NA = int(num_result[row[0]][0])
                    NO = int(num_result[row[0]][1])
                    NT = int(num_result[row[0]][2])
                    relation_result[row[0]] = []
                    for i in range(NA):
                        for j in range(NO):
                            for k in range(NT):
                                relation_result[row[0]].append(row[(i * NO + j) * NT + k + 1])
        return relation_result
