import os
import csv
import json
import shutil
import numpy as np
from PIL import Image

from ffmpeg import probe
from subprocess import call

import torch
import torchvision
import torch.nn as nn
from torchvision import transforms


def load_json(save_path: str):
    print("\n  Loading {}".format(save_path))
    with open(save_path, "r") as f:
        return json.load(f)


def load_aa():
    aa1 = load_json('./MOMA_LRG_anns/taxonomy/attribute.json')
    aa2 = load_json('./MOMA_LRG_anns/taxonomy/intransitive_action.json')
    aa = {}
    num = 0
    for t in aa1:
        for a in aa1[t]:
            aa[a[0]] = num
            num = num + 1
    for t in aa2:
        for a in aa2[t]:
            aa[a[0]] = num
            num = num + 1

    return aa


def load_act_sact(act_sact):
    act = {}
    sact = {}
    num_act = 0
    num_sact = 0
    for an in act_sact:
        act[an] = num_act
        num_act = num_act + 1
        for san in act_sact[an]:
            sact[san] = num_sact
            num_sact = num_sact + 1
    return act, sact


def load_url(path):
    lack_url = []
    with open(path, 'r') as op:
        csv_data = csv.reader(op, delimiter=',')
        for row in csv_data:
            lack_url.append(row[0])
    op.close()
    return lack_url

def read_label_csv(csv_path):
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


def cut_image(src_folder, tar_folder, bbox):
    print(tar_folder)
    img = Image.open(src_folder)
    if img.mode != "RGB":
        img = img.convert("RGB")
    width, height = img.size

    region_union = img.crop(bbox)

    region_union.save(tar_folder, 'JPEG')


def compute_bbox(bbox):
    x1, y1, w, h = bbox
    return [x1, y1, x1+w, y1+h]


def extract_frames(source_all_frame_path, dst_extract_frame_path):

    if not os.path.exists(dst_extract_frame_path):
        os.mkdir(dst_extract_frame_path)

    anns = load_json('./MOMA_LRG_anns/anns.json')

    for i in range(len(anns)):
        for j in range(len(anns[i]['activity']['sub_activities'])):
            vname = os.path.splitext(anns[i]['file_name'])[0] + '_' + anns[i]['activity']['sub_activities'][j]['id']
            v = anns[i]['activity']['sub_activities'][j]['higher_order_interactions']
            if not os.path.exists(dst_extract_frame_path + vname):
                os.mkdir(dst_extract_frame_path + vname)

            for k in range(len(v)):
                src = source_all_frame_path + v[k]['id'] + '.jpg'
                dst = dst_extract_frame_path + vname + '/' + "{:05d}".format(k) + '.jpg'

                shutil.copyfile(src, dst)

def write_dataset_level(dataset_level_file, split):

    anns = load_json('./MOMA_LRG_anns/anns.json')
    standard = load_json('./MOMA_LRG_anns/splits/standard.json')
    aa_labels = load_aa()
    act_sact = load_json('./MOMA_LRG_anns/taxonomy/act_sact.json')
    act_labels, sact_labels = load_act_sact(act_sact)

    actors = {}
    objects = {}
    atomic_actions = {}
    f_atomic_actions = {}

    with open(dataset_level_file, 'w') as out:
        for i in range(len(anns)):
            for j in range(len(anns[i]['activity']['sub_activities'])):
                vname = os.path.splitext(anns[i]['file_name'])[0] + '_' + anns[i]['activity']['sub_activities'][j]['id']
                v = anns[i]['activity']['sub_activities'][j]['higher_order_interactions']

                assert len(v) == anns[i]['activity']['sub_activities'][j]['end_time'] - \
                       anns[i]['activity']['sub_activities'][j]['start_time'] + 1
                actors[vname] = {}
                objects[vname] = {}
                atomic_actions[vname] = []
                f_atomic_actions[vname] = []

                for k in range(len(v)):
                    for l in range(len(v[k]['actors'])):
                        if v[k]['actors'][l]['id'] not in actors[vname]:
                            actors[vname][v[k]['actors'][l]['id']] = v[k]['actors'][l]['class_name']
                        else:
                            assert actors[vname][v[k]['actors'][l]['id']] == v[k]['actors'][l]['class_name']

                    for l in range(len(v[k]['objects'])):
                        if v[k]['objects'][l]['id'] not in objects[vname]:
                            objects[vname][v[k]['objects'][l]['id']] = v[k]['objects'][l]['class_name']
                        else:
                            assert objects[vname][v[k]['objects'][l]['id']] == v[k]['objects'][l]['class_name']

                    taa_label = []
                    for l in range(len(v[k]['attributes'])):
                        if aa_labels[v[k]['attributes'][l]['class_name']] not in atomic_actions[vname]:
                            atomic_actions[vname].append(aa_labels[v[k]['attributes'][l]['class_name']])
                        if aa_labels[v[k]['attributes'][l]['class_name']] not in taa_label:
                            taa_label.append(aa_labels[v[k]['attributes'][l]['class_name']])
                    for l in range(len(v[k]['intransitive_actions'])):
                        if aa_labels[v[k]['intransitive_actions'][l]['class_name']] not in atomic_actions[vname]:
                            atomic_actions[vname].append(aa_labels[v[k]['intransitive_actions'][l]['class_name']])
                        if aa_labels[v[k]['intransitive_actions'][l]['class_name']] not in taa_label:
                            taa_label.append(aa_labels[v[k]['intransitive_actions'][l]['class_name']])
                    taa_label = sorted(taa_label)
                    f_atomic_actions[vname].append(taa_label)

        for i in range(len(anns)):
            if split != '':
                if os.path.splitext(anns[i]['file_name'])[0] in standard[split]:
                    for j in range(len(anns[i]['activity']['sub_activities'])):
                        vname = os.path.splitext(anns[i]['file_name'])[0] + '_' + \
                                anns[i]['activity']['sub_activities'][j][
                                    'id']
                        v = anns[i]['activity']['sub_activities'][j]['higher_order_interactions']

                        out.write(vname)
                        out.write(',')
                        out.write(str(len(actors[vname])))
                        out.write(',')
                        out.write(str(len(objects[vname])))
                        out.write(',')
                        out.write(str(len(v)))

                        out.write(',')
                        actn = sorted(actors[vname])
                        if len(actors[vname]) > 0:
                            out.write(actors[vname][actn[0]].replace(' ', '_').split('/')[-1])
                        for k in range(1, len(actors[vname])):
                            out.write(';')
                            out.write(actors[vname][actn[k]].replace(' ', '_').split('/')[-1])

                        out.write(',')
                        objn = sorted(objects[vname])
                        if len(objects[vname]) > 0:
                            out.write(objects[vname][objn[0]].replace(' ', '_').split('/')[-1])
                        for k in range(1, len(objects[vname])):
                            out.write(';')
                            out.write(objects[vname][objn[k]].replace(' ', '_').split('/')[-1])

                        out.write(',')
                        if len(v) > 0:
                            out.write("{:05d}".format(0))
                        for k in range(1, len(v)):
                            out.write(';')
                            out.write("{:05d}".format(k))

                        out.write(',')

                        for k in range(len(v)):
                            if k > 0:
                                out.write(';')
                            if len(f_atomic_actions[vname][k]) > 0:
                                out.write(str(f_atomic_actions[vname][k][0]))
                            for l in range(1, len(f_atomic_actions[vname][k])):
                                out.write('*')
                                out.write(str(f_atomic_actions[vname][k][l]))

                        out.write(',')
                        atomic_actions[vname] = sorted(atomic_actions[vname])
                        if len(atomic_actions[vname]) > 0:
                            out.write(str(atomic_actions[vname][0]))
                        for k in range(1, len(atomic_actions[vname])):
                            out.write(';')
                            out.write(str(atomic_actions[vname][k]))

                        out.write(',')
                        out.write(str(sact_labels[anns[i]['activity']['sub_activities'][j]['class_name']]))
                        out.write(',')
                        out.write(str(act_labels[anns[i]['activity']['class_name']]))

                        out.write('\n')
            else:
                for j in range(len(anns[i]['activity']['sub_activities'])):
                    vname = os.path.splitext(anns[i]['file_name'])[0] + '_' + \
                            anns[i]['activity']['sub_activities'][j][
                                'id']
                    v = anns[i]['activity']['sub_activities'][j]['higher_order_interactions']

                    out.write(vname)
                    out.write(',')
                    out.write(str(len(actors[vname])))
                    out.write(',')
                    out.write(str(len(objects[vname])))
                    out.write(',')
                    out.write(str(len(v)))

                    out.write(',')
                    actn = sorted(actors[vname])
                    if len(actors[vname]) > 0:
                        out.write(actors[vname][actn[0]].replace(' ', '_').split('/')[-1])
                    for k in range(1, len(actors[vname])):
                        out.write(';')
                        out.write(actors[vname][actn[k]].replace(' ', '_').split('/')[-1])

                    out.write(',')
                    objn = sorted(objects[vname])
                    if len(objects[vname]) > 0:
                        out.write(objects[vname][objn[0]].replace(' ', '_').split('/')[-1])
                    for k in range(1, len(objects[vname])):
                        out.write(';')
                        out.write(objects[vname][objn[k]].replace(' ', '_').split('/')[-1])

                    out.write(',')
                    if len(v) > 0:
                        out.write("{:05d}".format(0))
                    for k in range(1, len(v)):
                        out.write(';')
                        out.write("{:05d}".format(k))

                    out.write(',')

                    for k in range(len(v)):
                        if k > 0:
                            out.write(';')
                        if len(f_atomic_actions[vname][k]) > 0:
                            out.write(str(f_atomic_actions[vname][k][0]))
                        for l in range(1, len(f_atomic_actions[vname][k])):
                            out.write('*')
                            out.write(str(f_atomic_actions[vname][k][l]))

                    out.write(',')
                    atomic_actions[vname] = sorted(atomic_actions[vname])
                    if len(atomic_actions[vname]) > 0:
                        out.write(str(atomic_actions[vname][0]))
                    for k in range(1, len(atomic_actions[vname])):
                        out.write(';')
                        out.write(str(atomic_actions[vname][k]))

                    out.write(',')
                    out.write(str(sact_labels[anns[i]['activity']['sub_activities'][j]['class_name']]))
                    out.write(',')
                    out.write(str(act_labels[anns[i]['activity']['class_name']]))

                    out.write('\n')

def write_object_level(object_level_path):

    if not os.path.exists(object_level_path):
        os.mkdir(object_level_path)

    anns = load_json('./MOMA_LRG_anns/anns.json')
    aa_labels = load_aa()

    atomic_actions = {}
    actors = {}
    objects = {}
    f_actors = {}
    f_objects = {}

    for i in range(len(anns)):
        for j in range(len(anns[i]['activity']['sub_activities'])):
            vname = os.path.splitext(anns[i]['file_name'])[0] + '_' + anns[i]['activity']['sub_activities'][j]['id']
            v = anns[i]['activity']['sub_activities'][j]['higher_order_interactions']

            assert len(v) == anns[i]['activity']['sub_activities'][j]['end_time'] - \
                   anns[i]['activity']['sub_activities'][j]['start_time'] + 1
            actors[vname] = {}
            objects[vname] = {}
            atomic_actions[vname] = []
            f_actors[vname] = []
            f_objects[vname] = []

            for k in range(len(v)):
                tact_label = {}
                for l in range(len(v[k]['actors'])):
                    if v[k]['actors'][l]['id'] not in actors[vname]:
                        actors[vname][v[k]['actors'][l]['id']] = v[k]['actors'][l]['class_name']
                    else:
                        assert actors[vname][v[k]['actors'][l]['id']] == v[k]['actors'][l]['class_name']
                    tact_label[v[k]['actors'][l]['id']] = v[k]['actors'][l]['class_name']
                f_actors[vname].append(tact_label)

                tobj_label = {}
                for l in range(len(v[k]['objects'])):
                    if v[k]['objects'][l]['id'] not in objects[vname]:
                        objects[vname][v[k]['objects'][l]['id']] = v[k]['objects'][l]['class_name']
                    else:
                        assert objects[vname][v[k]['objects'][l]['id']] == v[k]['objects'][l]['class_name']
                    tobj_label[v[k]['objects'][l]['id']] = v[k]['objects'][l]['class_name']
                f_objects[vname].append(tobj_label)

                taa_label = []
                for l in range(len(v[k]['attributes'])):
                    if aa_labels[v[k]['attributes'][l]['class_name']] not in atomic_actions[vname]:
                        atomic_actions[vname].append(aa_labels[v[k]['attributes'][l]['class_name']])
                    if aa_labels[v[k]['attributes'][l]['class_name']] not in taa_label:
                        taa_label.append(aa_labels[v[k]['attributes'][l]['class_name']])
                for l in range(len(v[k]['intransitive_actions'])):
                    if aa_labels[v[k]['intransitive_actions'][l]['class_name']] not in atomic_actions[vname]:
                        atomic_actions[vname].append(aa_labels[v[k]['intransitive_actions'][l]['class_name']])
                    if aa_labels[v[k]['intransitive_actions'][l]['class_name']] not in taa_label:
                        taa_label.append(aa_labels[v[k]['intransitive_actions'][l]['class_name']])

    for vname in actors:

        NA = len(actors[vname])
        NO = len(objects[vname])
        NT = len(f_actors[vname])

        ob_in_video = np.zeros(((NA + NO), NT), dtype=int)

        for i in range(NT):
            actn = sorted(actors[vname])
            objn = sorted(objects[vname])
            for j in range(NA):
                act = actn[j]
                if act in f_actors[vname][i]:
                    ob_in_video[j][i] = 1
            for j in range(NO):
                obj = objn[j]
                if obj in f_objects[vname][i]:
                    ob_in_video[j + NA][i] = 1

        ob_path = object_level_path + vname + '.npy'
        np.save(ob_path, ob_in_video)

def write_relation_level(relation_level_path):

    if not os.path.exists(relation_level_path):
        os.mkdir(relation_level_path)

    anns = load_json('./MOMA_LRG_anns/anns.json')
    aa_labels = load_aa()

    atomic_actions = {}
    actors = {}
    objects = {}
    f_actors = {}
    f_objects = {}
    f_relations = {}

    for i in range(len(anns)):
        for j in range(len(anns[i]['activity']['sub_activities'])):
            vname = os.path.splitext(anns[i]['file_name'])[0] + '_' + anns[i]['activity']['sub_activities'][j]['id']
            v = anns[i]['activity']['sub_activities'][j]['higher_order_interactions']

            assert len(v) == anns[i]['activity']['sub_activities'][j]['end_time'] - \
                   anns[i]['activity']['sub_activities'][j]['start_time'] + 1
            actors[vname] = {}
            objects[vname] = {}
            atomic_actions[vname] = []
            f_actors[vname] = []
            f_objects[vname] = []

            for k in range(len(v)):
                tact_label = {}
                for l in range(len(v[k]['actors'])):
                    if v[k]['actors'][l]['id'] not in actors[vname]:
                        actors[vname][v[k]['actors'][l]['id']] = v[k]['actors'][l]['class_name']
                    else:
                        assert actors[vname][v[k]['actors'][l]['id']] == v[k]['actors'][l]['class_name']
                    tact_label[v[k]['actors'][l]['id']] = v[k]['actors'][l]['class_name']
                f_actors[vname].append(tact_label)

                tobj_label = {}
                for l in range(len(v[k]['objects'])):
                    if v[k]['objects'][l]['id'] not in objects[vname]:
                        objects[vname][v[k]['objects'][l]['id']] = v[k]['objects'][l]['class_name']
                    else:
                        assert objects[vname][v[k]['objects'][l]['id']] == v[k]['objects'][l]['class_name']
                    tobj_label[v[k]['objects'][l]['id']] = v[k]['objects'][l]['class_name']
                f_objects[vname].append(tobj_label)

                taa_label = []
                for l in range(len(v[k]['attributes'])):
                    if aa_labels[v[k]['attributes'][l]['class_name']] not in atomic_actions[vname]:
                        atomic_actions[vname].append(aa_labels[v[k]['attributes'][l]['class_name']])
                    if aa_labels[v[k]['attributes'][l]['class_name']] not in taa_label:
                        taa_label.append(aa_labels[v[k]['attributes'][l]['class_name']])
                for l in range(len(v[k]['intransitive_actions'])):
                    if aa_labels[v[k]['intransitive_actions'][l]['class_name']] not in atomic_actions[vname]:
                        atomic_actions[vname].append(aa_labels[v[k]['intransitive_actions'][l]['class_name']])
                    if aa_labels[v[k]['intransitive_actions'][l]['class_name']] not in taa_label:
                        taa_label.append(aa_labels[v[k]['intransitive_actions'][l]['class_name']])

    for i in range(len(anns)):
        for j in range(len(anns[i]['activity']['sub_activities'])):
            vname = os.path.splitext(anns[i]['file_name'])[0] + '_' + anns[i]['activity']['sub_activities'][j]['id']
            v = anns[i]['activity']['sub_activities'][j]['higher_order_interactions']

            NA = len(actors[vname])
            NO = len(objects[vname])
            NT = len(f_actors[vname])
            actn = sorted(actors[vname])
            objn = sorted(objects[vname])

            if vname not in f_relations:
                f_relations[vname] = np.zeros((NA * NO, NT), dtype=int)

            for k in range(len(v)):

                for l in range(len(v[k]['relationships'])):
                    if v[k]['relationships'][l]['source_id'] in actn and v[k]['relationships'][l]['target_id'] in objn:
                        tact = actn.index(v[k]['relationships'][l]['source_id'])
                        tobj = objn.index(v[k]['relationships'][l]['target_id'])
                        f_relations[vname][tact * NO + tobj][k] = 1

                for l in range(len(v[k]['transitive_actions'])):
                    if v[k]['transitive_actions'][l]['source_id'] in actn and v[k]['transitive_actions'][l][
                        'target_id'] in objn:
                        tact = actn.index(v[k]['transitive_actions'][l]['source_id'])
                        tobj = objn.index(v[k]['transitive_actions'][l]['target_id'])
                        f_relations[vname][tact * NO + tobj][k] = 1

            ob_path = relation_level_path + vname + '.npy'
            np.save(ob_path, f_relations[vname])

def extract_object_images(object_image_path, dataset_level_file, dst_frame_path):

    if not os.path.exists(object_image_path):
        os.mkdir(object_image_path)

    anns = load_json('./MOMA_LRG_anns/anns.json')

    actors = {}
    objects = {}
    f_actors = {}
    f_objects = {}
    f_actors_bbox = {}
    f_objects_bbox = {}

    for i in range(len(anns)):
        for j in range(len(anns[i]['activity']['sub_activities'])):
            vname = os.path.splitext(anns[i]['file_name'])[0] + '_' + anns[i]['activity']['sub_activities'][j]['id']
            v = anns[i]['activity']['sub_activities'][j]['higher_order_interactions']

            assert len(v) == anns[i]['activity']['sub_activities'][j]['end_time'] - \
                   anns[i]['activity']['sub_activities'][j]['start_time'] + 1
            actors[vname] = {}
            objects[vname] = {}
            f_actors[vname] = []
            f_objects[vname] = []
            f_actors_bbox[vname] = []
            f_objects_bbox[vname] = []

            for k in range(len(v)):
                tact_label = {}
                tact_bbox = {}
                for l in range(len(v[k]['actors'])):
                    if v[k]['actors'][l]['id'] not in actors[vname]:
                        actors[vname][v[k]['actors'][l]['id']] = v[k]['actors'][l]['class_name']
                    else:
                        assert actors[vname][v[k]['actors'][l]['id']] == v[k]['actors'][l]['class_name']
                    tact_label[v[k]['actors'][l]['id']] = v[k]['actors'][l]['class_name']
                    tact_bbox[v[k]['actors'][l]['id']] = v[k]['actors'][l]['bbox']
                f_actors[vname].append(tact_label)
                f_actors_bbox[vname].append(tact_bbox)

                tobj_label = {}
                tobj_bbox = {}
                for l in range(len(v[k]['objects'])):
                    if v[k]['objects'][l]['id'] not in objects[vname]:
                        objects[vname][v[k]['objects'][l]['id']] = v[k]['objects'][l]['class_name']
                    else:
                        assert objects[vname][v[k]['objects'][l]['id']] == v[k]['objects'][l]['class_name']
                    tobj_label[v[k]['objects'][l]['id']] = v[k]['objects'][l]['class_name']
                    tobj_bbox[v[k]['objects'][l]['id']] = v[k]['objects'][l]['bbox']
                f_objects[vname].append(tobj_label)
                f_objects_bbox[vname].append(tobj_bbox)

    num_result, actor_result, object_result, frame_result = read_label_csv(dataset_level_file)

    for vname in num_result:

        actn = sorted(actors[vname])
        objn = sorted(objects[vname])
        NA = len(actors[vname])
        NO = len(objects[vname])
        NT = len(f_actors[vname])

        if not os.path.exists(object_image_path + vname):
            os.mkdir(object_image_path + vname)

        for i in range(NT):
            fname = frame_result[vname][i]

            for j in range(NA):
                act = actn[j]
                if act in f_actors[vname][i]:
                    src = os.path.join(dst_frame_path, vname, fname + '.jpg')
                    dst = os.path.join(object_image_path, vname, fname + '-' + str(j) + '-' + actor_result[vname][j] + '.jpg')
                    bbox = f_actors_bbox[vname][i][act]
                    nbbox = compute_bbox(bbox)
                    cut_image(src, dst, nbbox)

            for j in range(NO):
                obj = objn[j]
                if obj in f_objects[vname][i]:
                    src = os.path.join(dst_frame_path, vname, fname + '.jpg')
                    dst = os.path.join(object_image_path, vname, fname + '-' + str(j + NA) + '-' + object_result[vname][j] + '.jpg')
                    bbox = f_objects_bbox[vname][i][obj]
                    nbbox = compute_bbox(bbox)
                    cut_image(src, dst, nbbox)

def extract_object_features(object_image_path, object_features_path):

    if not os.path.exists(object_features_path):
        os.mkdir(object_features_path)

    to_tensor = transforms.Compose([transforms.Resize((256, 256), 2),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                    ])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = torchvision.models.resnet152(pretrained=True)
    model = model.cuda()

    model.eval()

    features = list(model.children())[:-1]
    modelout = nn.Sequential(*features).to(device)

    with torch.no_grad():

        all_video = os.listdir(object_image_path)

        for video in all_video:
            video_path = os.path.join(object_image_path, video)
            os.mkdir(object_features_path + video)
            img_names = os.listdir(video_path)

            for img_name in img_names:
                img_path = os.path.join(video_path, img_name)
                img = Image.open(img_path).convert("RGB")
                input = to_tensor(img).unsqueeze(0).to(device, torch.float)

                step_out = modelout(input)
                out = step_out.squeeze(3)
                out = out.squeeze(2)
                out = out.squeeze(0)

                np.save(os.path.join(object_features_path, video, os.path.splitext(img_name)[0] + '.npy'), out.cpu().numpy())

def write_adjacent(adjacent_path, dataset_level_file, object_level_path, relation_level_path):

    if not os.path.exists(adjacent_path):
        os.mkdir(adjacent_path)

    write_spa_adjacent(os.path.join(adjacent_path, 'spa_adj/'), object_level_path)
    write_tem_adjacent(os.path.join(adjacent_path, 'tem_adj/'), dataset_level_file, object_level_path)
    write_rel_adjacent(os.path.join(adjacent_path, 'rel_adj/'), dataset_level_file, relation_level_path)

def write_spa_adjacent(adjacent_path, object_level_path):

    if not os.path.exists(adjacent_path):
        os.mkdir(adjacent_path)

    anns = load_json('./MOMA_LRG_anns/anns.json')

    atomic_actions = {}
    actors = {}
    objects = {}
    f_actors = {}
    f_objects = {}
    object_anno = {}

    for i in range(len(anns)):
        for j in range(len(anns[i]['activity']['sub_activities'])):
            vname = os.path.splitext(anns[i]['file_name'])[0] + '_' + anns[i]['activity']['sub_activities'][j]['id']
            v = anns[i]['activity']['sub_activities'][j]['higher_order_interactions']

            assert len(v) == anns[i]['activity']['sub_activities'][j]['end_time'] - \
                   anns[i]['activity']['sub_activities'][j]['start_time'] + 1
            actors[vname] = {}
            objects[vname] = {}
            atomic_actions[vname] = []
            f_actors[vname] = []
            f_objects[vname] = []
            object_anno[vname] = []

            for k in range(len(v)):
                tact_label = {}
                for l in range(len(v[k]['actors'])):
                    if v[k]['actors'][l]['id'] not in actors[vname]:
                        actors[vname][v[k]['actors'][l]['id']] = v[k]['actors'][l]['class_name']
                    else:
                        assert actors[vname][v[k]['actors'][l]['id']] == v[k]['actors'][l]['class_name']
                    tact_label[v[k]['actors'][l]['id']] = v[k]['actors'][l]['class_name']
                f_actors[vname].append(tact_label)

                tobj_label = {}
                for l in range(len(v[k]['objects'])):
                    if v[k]['objects'][l]['id'] not in objects[vname]:
                        objects[vname][v[k]['objects'][l]['id']] = v[k]['objects'][l]['class_name']
                    else:
                        assert objects[vname][v[k]['objects'][l]['id']] == v[k]['objects'][l]['class_name']
                    tobj_label[v[k]['objects'][l]['id']] = v[k]['objects'][l]['class_name']
                f_objects[vname].append(tobj_label)

                object_anno[vname].append(v[k]['relationships'] + v[k]['transitive_actions'])

    for vname in actors:
        actn = sorted(actors[vname])
        objn = sorted(objects[vname])
        NA = len(actors[vname])
        NO = len(objects[vname])
        NT = len(f_actors[vname])
        all_spa_adj = np.zeros(((NA + NO) * NT, (NA + NO) * NT), dtype=int)
        object_in_video = np.load(object_level_path + os.path.splitext(vname)[0] + '.npy')
        object_in_video = object_in_video.transpose(1, 0)
        for i in range(NT):
            single_spa_adj = np.zeros(((NA + NO), (NA + NO)), dtype=int)
            sing_oinv = object_in_video[i]
            frame_relation = object_anno[vname][i]
            for si in range(NA + NO):
                for sj in range(NA + NO):
                    if si == sj:
                        single_spa_adj[si][sj] = sing_oinv[si] * sing_oinv[sj]
                    elif si < NA:
                        for ob_rel in frame_relation:

                            tact = ob_rel['source_id']
                            tobj = ob_rel['target_id']

                            if tact in actn and tobj in objn:
                                ti = actn.index(tact)
                                tj = objn.index(tobj)

                                if ti == si and NA + tj == sj:
                                    single_spa_adj[si][sj] = sing_oinv[si] * sing_oinv[sj]
                                    single_spa_adj[sj][si] = sing_oinv[si] * sing_oinv[sj]

            all_spa_adj[i * (NA + NO):(i + 1) * (NA + NO), i * (NA + NO):(i + 1) * (NA + NO)] = single_spa_adj
        np.save(adjacent_path + os.path.splitext(vname)[0] + '.npy', all_spa_adj)

def write_tem_adjacent(adjacent_path, dataset_level_file, object_level_path):

    if not os.path.exists(adjacent_path):
        os.mkdir(adjacent_path)

    num_result, actor_result, object_result, frame_result = read_label_csv(dataset_level_file)

    for vname in num_result:
        NA = int(num_result[vname][0])
        NO = int(num_result[vname][1])
        NT = int(num_result[vname][2])
        all_spa_adj = np.zeros(((NA + NO) * NT, (NA + NO) * NT), dtype=int)
        object_in_video = np.load(object_level_path + os.path.splitext(vname)[0] + '.npy')
        for i in range((NA + NO)):
            single_spa_adj = np.zeros((NT, NT), dtype=int)
            sing_oinv = object_in_video[i]
            for si in range(NT):
                for sj in range(NT):
                    if abs(si - sj) < 13:
                        single_spa_adj[si][sj] = sing_oinv[si] * sing_oinv[sj]
            all_spa_adj[i * NT:(i + 1) * NT, i * NT:(i + 1) * NT] = single_spa_adj
        np.save(adjacent_path + os.path.splitext(vname)[0] + '.npy', all_spa_adj)


def write_rel_adjacent(adjacent_path, dataset_level_file, relation_level_path):

    if not os.path.exists(adjacent_path):
        os.mkdir(adjacent_path)

    num_result, actor_result, object_result, frame_result = read_label_csv(dataset_level_file)

    for vname in num_result:
        NA = int(num_result[vname][0])
        NO = int(num_result[vname][1])
        NT = int(num_result[vname][2])
        all_rel_adj = np.zeros(((NA * NO) * NT, (NA * NO) * NT), dtype=int)
        relation_in_video = np.load(relation_level_path + os.path.splitext(vname)[0] + '.npy')
        for i in range((NA * NO)):
            single_rel_adj = np.zeros((NT, NT), dtype=int)
            sing_rinv = relation_in_video[i]
            for si in range(NT):
                for sj in range(NT):
                    if abs(si - sj) < 13:
                        single_rel_adj[si][sj] = sing_rinv[si] * sing_rinv[sj]
            all_rel_adj[i * NT:(i + 1) * NT, i * NT:(i + 1) * NT] = single_rel_adj
        np.save(adjacent_path + os.path.splitext(vname)[0] + '.npy', all_rel_adj)

def write_relation_semantic_1(semantic_file):

    GROUP_relationships = {'carrying': '1', 'carrying_on_their_back': '1', 'holding': '1', 'leaning_on': '1', 'looking_at': '3', 'lying_on': '1', 'pushing': '1', 'sitting_on': '1', 'stepping_on': '1', 'wearing_on_their_head': '1', 'behind': '2', 'beneath': '2', 'covered_by': '2', 'in': '2', 'in_front_of': '2', 'on_the_side_of': '2', 'on_top_of': '2', 'touching': '2', 'aiming_and_throwing': '1', 'grabbing': '1', 'grabbing_from_someone': '1', 'handing_over': '1', 'installing': '1', 'jumping_off': '1', 'kicking': '1', 'lifting': '1', 'picking_up_from_the_table': '1', 'placing_onto_the_table': '1', 'pointing_at': '1', 'pouring_into': '1', 'putting_on': '1', 'removing': '1', 'sitting_down_on': '1', 'standing_up_from': '1', 'taking_off': '1', 'throwing_away': '1', 'blowing': '0', 'eating_or_drinking_from': '0', 'hitting': '0', 'massaging': '0', 'pressing': '0', 'riding_on': '0', 'wiping': '0', 'writing_on': '0'}

    anns = load_json('./MOMA_LRG_anns/anns.json')

    atomic_actions = {}
    actors = {}
    objects = {}
    f_actors = {}
    f_objects = {}
    object_anno = {}
    relationships = {}
    relationships_num = {}

    for i in range(len(anns)):
        for j in range(len(anns[i]['activity']['sub_activities'])):
            vname = os.path.splitext(anns[i]['file_name'])[0] + '_' + anns[i]['activity']['sub_activities'][j]['id']
            v = anns[i]['activity']['sub_activities'][j]['higher_order_interactions']

            assert len(v) == anns[i]['activity']['sub_activities'][j]['end_time'] - \
                   anns[i]['activity']['sub_activities'][j]['start_time'] + 1
            actors[vname] = {}
            objects[vname] = {}
            atomic_actions[vname] = []
            f_actors[vname] = []
            f_objects[vname] = []
            object_anno[vname] = []
            relationships[vname] = []
            relationships_num[vname] = []

            for k in range(len(v)):
                tact_label = {}
                for l in range(len(v[k]['actors'])):
                    if v[k]['actors'][l]['id'] not in actors[vname]:
                        actors[vname][v[k]['actors'][l]['id']] = v[k]['actors'][l]['class_name']
                    else:
                        assert actors[vname][v[k]['actors'][l]['id']] == v[k]['actors'][l]['class_name']
                    tact_label[v[k]['actors'][l]['id']] = v[k]['actors'][l]['class_name']
                f_actors[vname].append(tact_label)

                tobj_label = {}
                for l in range(len(v[k]['objects'])):
                    if v[k]['objects'][l]['id'] not in objects[vname]:
                        objects[vname][v[k]['objects'][l]['id']] = v[k]['objects'][l]['class_name']
                    else:
                        assert objects[vname][v[k]['objects'][l]['id']] == v[k]['objects'][l]['class_name']
                    tobj_label[v[k]['objects'][l]['id']] = v[k]['objects'][l]['class_name']
                f_objects[vname].append(tobj_label)

                object_anno[vname].append(v[k]['relationships'] + v[k]['transitive_actions'])

                trels = v[k]['relationships']
                f_relationships = {}
                f_relationships_num = {}
                for trel in trels:
                    ta = trel['source_id']
                    to = trel['target_id']
                    rel_class = trel['class_name'][6:].replace(" [trg]", "").replace(" ", "_")
                    if rel_class not in GROUP_relationships:
                        continue

                    if ta not in f_relationships:
                        f_relationships[ta] = {}
                        f_relationships_num[ta] = {}

                    if to not in f_relationships[ta]:
                        f_relationships[ta][to] = []
                        f_relationships_num[ta][to] = []

                    f_relationships[ta][to].append(rel_class)
                    f_relationships_num[ta][to].append(GROUP_relationships[rel_class])

                trels = v[k]['transitive_actions']

                for trel in trels:
                    ta = trel['source_id']
                    to = trel['target_id']
                    rel_class = trel['class_name'][6:].replace(" [trg]", "").replace(" ", "_")
                    if rel_class not in GROUP_relationships:
                        continue

                    if ta not in f_relationships:
                        f_relationships[ta] = {}
                        f_relationships_num[ta] = {}

                    if to not in f_relationships[ta]:
                        f_relationships[ta][to] = []
                        f_relationships_num[ta][to] = []

                    f_relationships[ta][to].append(rel_class)
                    f_relationships_num[ta][to].append(GROUP_relationships[rel_class])

                relationships[vname].append(f_relationships)
                relationships_num[vname].append(f_relationships_num)

    with open(semantic_file, 'w') as out:

        for vname in actors:
            NA = len(actors[vname])
            NO = len(objects[vname])
            NT = len(f_actors[vname])

            out.write(vname)
            for i in range(NA):
                for j in range(NO):
                    for k in range(NT):
                        out.write(',')
                        ta = sorted(actors[vname])[i]
                        to = sorted(objects[vname])[j]
                        tf = k

                        if ta in relationships[vname][tf]:
                            if to in relationships[vname][tf][ta]:
                                trel_num = relationships_num[vname][tf][ta][to]
                                min_trel_num = trel_num.index(min(trel_num))
                                out.write(relationships[vname][tf][ta][to][min_trel_num])

            out.write('\n')