import os
import csv
import json
import numpy as np
from PIL import Image

from ffmpeg import probe
from subprocess import call

import torch
import torchvision
import torch.nn as nn
from torchvision import transforms


def get_source_info_ffmpeg(source_name):
    return_value = 0
    try:
        info = probe(source_name)
        # print(info)
        # print("---------------------------------")
        vs = next(c for c in info['streams'] if c['codec_type'] == 'video')
        format_name = info['format']['format_name']
        codec_name = vs['codec_name']
        duration_ts = float(vs['duration_ts'])
        fps = vs['r_frame_rate']
        width = vs['width']
        height = vs['height']
        print("format_name:{} \ncodec_name:{} \nduration_ts:{} \nwidth:{} \nheight:{} \nfps:{}".format(format_name,
                                                                                                       codec_name,
                                                                                                       duration_ts,
                                                                                                       width, height,
                                                                                                       fps))
    except (OSError, TypeError, ValueError, KeyError, SyntaxError) as e:
        print("init_source:{} error. {}\n".format(source_name, str(e)))
        return_value = 0
    return fps

def read_fps(path):

    fps = {}
    with open(path, 'r') as op:
        csv_data = csv.reader(op, delimiter=',')
        for row in csv_data:
            fps[row[0]] = row[-1]
    op.close()
    return fps

def read_split(path):

    split = []
    with open(path, 'r') as op:
        csv_data = csv.reader(op, delimiter=',')
        for row in csv_data:
            split.append(row[0])
    op.close()
    return split

def load_json(save_path: str):
    #print("\n  Loading {}".format(save_path))
    with open(save_path, "r") as f:
        return json.load(f)

def load_aa(path):
    aa = {}
    num = 0
    with open(path, 'r') as op:
        csv_data = csv.reader(op, delimiter=',')
        for row in csv_data:
            aa[row[0]] = num
            num = num + 1
    op.close()
    return aa


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
    #print(tar_folder)
    img = Image.open(src_folder)
    if img.mode != "RGB":
        img = img.convert("RGB")
    width, height = img.size

    region_union = img.crop(bbox)

    region_union.save(tar_folder, 'JPEG')


def compute_bbox(bbox):
    x_od = []
    y_od = []
    for k in bbox:
        x_od.append(bbox[k]['x'])
        y_od.append(bbox[k]['y'])
    x1 = min(x_od)
    y1 = min(y_od)
    x2 = max(x_od)
    y2 = max(y_od)
    if x1 == x2:
        x2 = x2 + 1
    if y1 == y2:
        y2 = y2 + 1
    return [x1, y1, x2, y2]

def extract_frames(source_path, dst_path, fps_file):
    graph_anns = load_json('./MOMA_anns/graph_anns.json')
    frame_rate = {}
    for d in graph_anns:

        if d['trim_video_id'] not in frame_rate:
            frame_rate[d['trim_video_id']] = d['fps']
        else:
            if d['fps'] != frame_rate[d['trim_video_id']]:
                print("error!")

    with open(fps_file, 'w') as out:
        for source_name in frame_rate:
            fps = get_source_info_ffmpeg(os.path.join(source_path, source_name + '.mp4')).split('/')
            fps = int(int(fps[0]) / int(fps[1]) + 0.5)
            out.write(source_name)
            out.write(',')
            out.write(str(fps))
            out.write('\n')

            if fps > 30:
                fps = 30
            os.mkdir(os.path.join(dst_path, source_name))
            call(["ffmpeg", "-i", os.path.join(source_path, source_name + '.mp4'), "-vf", "fps=" + str(fps),
                  os.path.join(os.path.join(dst_path, source_name) + "/%05d.jpg")])

def write_dataset_level(dataset_level_file, split_file, fps_file):
    act = load_json('./MOMA_anns/act_cids.json')
    sact = load_json('./MOMA_anns/sact_cids.json')
    split = read_split(split_file)
    fps = read_fps(fps_file)
    aa_labels = load_aa('./MOMA_anns/aact_cnames.txt')

    video_id = 1891

    graph_anns = load_json('./MOMA_anns/graph_anns.json')

    act_id = {}
    atomic_actions = {}
    f_atomic_actions = {}
    actors = {}
    objects = {}
    wframes = {}

    with open(dataset_level_file, 'w') as out:
        for g in graph_anns:
            vname = g['trim_video_id']

            if vname in split:
                act_id[vname] = g['raw_video_id']
                frames = sorted(os.listdir('/home/ouyangjun/workspace/MOMA/MOMA-1.0/all_frames/' + vname))
                if vname not in atomic_actions:
                    atomic_actions[vname] = []
                    f_atomic_actions[vname] = {}
                    actors[vname] = {}
                    objects[vname] = {}
                    wframes[vname] = []

                tfps = float(fps[vname])
                if tfps > 30:
                    tfps = 30
                frame_num = int(g['frame_timestamp'] * tfps + 0.5)
                if frame_num >= len(frames):
                    frame_num = len(frames) - 1

                fname = frames[frame_num]
                wframes[vname].append(fname)

                taas = g['annotation']['atomic_actions']
                taa_label = []
                for taa in taas:
                    if aa_labels[taa['class']] not in atomic_actions[vname]:
                        atomic_actions[vname].append(aa_labels[taa['class']])
                    taa_label.append(aa_labels[taa['class']])
                taa_label = sorted(taa_label)
                f_atomic_actions[vname][fname] = taa_label

                tacts = g['annotation']['actors']
                for tact in tacts:
                    if tact['id_in_video'] not in actors[vname]:
                        actors[vname][tact['id_in_video']] = '_'.join(tact['class'].split(' ')).split('/')[-1]

                tobjs = g['annotation']['objects']
                for tobj in tobjs:
                    if tobj['id_in_video'] not in objects[vname]:
                        ob = '_'.join(tobj['class'].split(' '))
                        if ob == 'cream/gel/liquid':
                            ob = 'liquid'
                        elif ob == 'stool/bench':
                            ob = 'stool'
                        objects[vname][tobj['id_in_video']] = ob

        for vname in split:
            out.write(vname)
            out.write(',')
            out.write(str(len(actors[vname])))
            out.write(',')
            out.write(str(len(objects[vname])))
            out.write(',')
            out.write(str(len(wframes[vname])))

            out.write(',')
            actn = sorted(actors[vname])
            if len(actors[vname]) > 0:
                out.write(actors[vname][actn[0]])
            for i in range(1, len(actors[vname])):
                out.write(';')
                out.write(actors[vname][actn[i]].replace(' ', '_'))

            out.write(',')
            objn = sorted(objects[vname])
            if len(objects[vname]) > 0:
                out.write(objects[vname][objn[0]])
            for i in range(1, len(objects[vname])):
                out.write(';')
                out.write(objects[vname][objn[i]].replace(' ', '_'))

            out.write(',')
            if len(wframes[vname]) > 0:
                out.write(os.path.splitext(wframes[vname][0])[0])
            for i in range(1, len(wframes[vname])):
                out.write(';')
                out.write(os.path.splitext(wframes[vname][i])[0])

            out.write(',')
            wframes[vname] = sorted(wframes[vname])
            j = 0
            for fname in wframes[vname]:
                if j > 0:
                    out.write(';')
                if len(f_atomic_actions[vname][fname]) > 0:
                    out.write(str(f_atomic_actions[vname][fname][0]))
                for i in range(1, len(f_atomic_actions[vname][fname])):
                    out.write('*')
                    out.write(str(f_atomic_actions[vname][fname][i]))
                j = j + 1

            out.write(',')
            atomic_actions[vname] = sorted(atomic_actions[vname])
            if len(atomic_actions[vname]) > 0:
                out.write(str(atomic_actions[vname][0]))
            for i in range(1, len(atomic_actions[vname])):
                out.write(';')
                out.write(str(atomic_actions[vname][i]))

            out.write(',')
            out.write(str(sact[vname]))
            out.write(',')
            out.write(str(act[act_id[vname]]))
            out.write('\n')

def write_object_level(object_level_path, split_file, fps_file, dst_frame_path):
    if not os.path.exists(object_level_path):
        os.mkdir(object_level_path)
    split = read_split(split_file)
    fps = read_fps(fps_file)
    aa_labels = load_aa('./MOMA_anns/aact_cnames.txt')

    graph_anns = load_json('./MOMA_anns/graph_anns.json')

    atomic_actions = {}
    actors = {}
    objects = {}
    wframes = {}
    f_actors = {}
    f_objects = {}

    for g in graph_anns:
        vname = g['trim_video_id']

        if vname in split:
            frames = sorted(os.listdir(dst_frame_path + vname))
            if vname not in atomic_actions:
                atomic_actions[vname] = []
                actors[vname] = {}
                objects[vname] = {}
                wframes[vname] = []
                f_actors[vname] = {}
                f_objects[vname] = {}

            tfps = float(fps[vname])
            if tfps > 30:
                tfps = 30
            frame_num = int(g['frame_timestamp'] * tfps + 0.5)
            if frame_num >= len(frames):
                frame_num = len(frames) - 1

            fname = frames[frame_num]
            wframes[vname].append(fname)

            taas = g['annotation']['atomic_actions']
            for taa in taas:
                if aa_labels[taa['class']] not in atomic_actions[vname]:
                    atomic_actions[vname].append(aa_labels[taa['class']])

            tacts = g['annotation']['actors']
            tact_label = []
            for tact in tacts:
                if tact['id_in_video'] not in actors[vname]:
                    actors[vname][tact['id_in_video']] = tact['class']
                tact_label.append(tact['id_in_video'])
            tact_label = sorted(tact_label)
            f_actors[vname][fname] = tact_label

            tobjs = g['annotation']['objects']
            tobj_label = []
            for tobj in tobjs:
                if tobj['id_in_video'] not in objects[vname]:
                    objects[vname][tobj['id_in_video']] = tobj['class']
                tobj_label.append(tobj['id_in_video'])
            tobj_label = sorted(tobj_label)
            f_objects[vname][fname] = tobj_label

    for vname in split:

        NA = len(actors[vname])
        NO = len(objects[vname])
        NT = len(wframes[vname])

        ob_in_video = np.zeros(((NA + NO), NT), dtype=int)

        for i in range(NT):
            fname = wframes[vname][i]
            actn = sorted(actors[vname])
            objn = sorted(objects[vname])
            for j in range(NA):
                act = actn[j]
                if act in f_actors[vname][fname]:
                    ob_in_video[j][i] = 1
            for j in range(NO):
                obj = objn[j]
                if obj in f_objects[vname][fname]:
                    ob_in_video[j + NA][i] = 1

        ob_path = object_level_path + vname + '.npy'
        np.save(ob_path, ob_in_video)

def write_relation_level(relation_level_path, split_file, fps_file, dst_frame_path):

    if not os.path.exists(relation_level_path):
        os.mkdir(relation_level_path)

    split = read_split(split_file)
    fps = read_fps(fps_file)
    aa_labels = load_aa('./MOMA_anns/aact_cnames.txt')

    graph_anns = load_json('./MOMA_anns/graph_anns.json')

    atomic_actions = {}
    actors = {}
    objects = {}
    wframes = {}
    f_actors = {}
    f_objects = {}
    f_relations = {}

    for g in graph_anns:
        vname = g['trim_video_id']

        if vname in split:
            frames = sorted(os.listdir(dst_frame_path + vname))
            if vname not in atomic_actions:
                atomic_actions[vname] = []
                actors[vname] = {}
                objects[vname] = {}
                wframes[vname] = []
                f_actors[vname] = {}
                f_objects[vname] = {}

            tfps = float(fps[vname])
            if tfps > 30:
                tfps = 30
            frame_num = int(g['frame_timestamp'] * tfps + 0.5)
            if frame_num >= len(frames):
                frame_num = len(frames) - 1

            fname = frames[frame_num]
            wframes[vname].append(fname)

            taas = g['annotation']['atomic_actions']
            for taa in taas:
                if aa_labels[taa['class']] not in atomic_actions[vname]:
                    atomic_actions[vname].append(aa_labels[taa['class']])

            tacts = g['annotation']['actors']
            tact_label = []
            for tact in tacts:
                if tact['id_in_video'] not in actors[vname]:
                    actors[vname][tact['id_in_video']] = tact['class']
                tact_label.append(tact['id_in_video'])
            tact_label = sorted(tact_label)
            f_actors[vname][fname] = tact_label

            tobjs = g['annotation']['objects']
            tobj_label = []
            for tobj in tobjs:
                if tobj['id_in_video'] not in objects[vname]:
                    objects[vname][tobj['id_in_video']] = tobj['class']
                tobj_label.append(tobj['id_in_video'])
            tobj_label = sorted(tobj_label)
            f_objects[vname][fname] = tobj_label

    for g in graph_anns:

        frames = sorted(os.listdir(dst_frame_path + vname))
        tfps = float(fps[vname])
        if tfps > 30:
            tfps = 30
        frame_num = int(g['frame_timestamp'] * tfps + 0.5)
        if frame_num >= len(frames):
            frame_num = len(frames) - 1

        fname = frames[frame_num]
        vname = g['trim_video_id']

        if vname in split:

            NA = len(actors[vname])
            NO = len(objects[vname])
            NT = len(wframes[vname])
            actn = sorted(actors[vname])
            objn = sorted(objects[vname])

            if vname not in f_relations:
                f_relations[vname] = np.zeros((NA * NO, NT), dtype=int)

            trelas = g['annotation']['relationships']
            for trela in trelas:
                # if trela['class'] in Rela_label:
                descrip = trela['description'][1:-1]
                tacts = descrip.split('),(')[0].split(',')
                tobjs = descrip.split('),(')[1].split(',')
                for tact in tacts:
                    for tobj in tobjs:
                        if tact in actn and tobj in objn:
                            i = actn.index(tact)
                            j = objn.index(tobj)
                            k = wframes[vname].index(fname)
                            f_relations[vname][i * NO + j][k] = 1

    for vname in split:
        ob_path = relation_level_path + vname + '.npy'
        np.save(ob_path, f_relations[vname])

def extract_object_images(object_image_path, split_file, fps_file, dataset_level_file, dst_frame_path):

    if not os.path.exists(object_image_path):
        os.mkdir(object_image_path)

    split = read_split(split_file)
    fps = read_fps(fps_file)
    aa_labels = load_aa('./MOMA_anns/aact_cnames.txt')

    graph_anns = load_json('./MOMA_anns/graph_anns.json')
    num_result, actor_result, object_result, frame_result = read_label_csv(dataset_level_file)

    atomic_actions = {}
    actors = {}
    objects = {}
    wframes = {}
    f_actors = {}
    f_objects = {}
    f_actors_bbox = {}
    f_objects_bbox = {}

    for g in graph_anns:
        vname = g['trim_video_id']

        if vname in split:
            frames = sorted(os.listdir(dst_frame_path + vname))
            if vname not in atomic_actions:
                atomic_actions[vname] = []
                actors[vname] = {}
                objects[vname] = {}
                wframes[vname] = []
                f_actors[vname] = {}
                f_objects[vname] = {}
                f_actors_bbox[vname] = {}
                f_objects_bbox[vname] = {}

            tfps = float(fps[vname])
            if tfps > 30:
                tfps = 30
            frame_num = int(g['frame_timestamp'] * tfps + 0.5)
            if frame_num >= len(frames):
                frame_num = len(frames) - 1

            fname = frames[frame_num]
            wframes[vname].append(fname)

            taas = g['annotation']['atomic_actions']
            for taa in taas:
                if aa_labels[taa['class']] not in atomic_actions[vname]:
                    atomic_actions[vname].append(aa_labels[taa['class']])

            tacts = g['annotation']['actors']
            tact_label = []
            f_actors_bbox[vname][fname] = {}
            for tact in tacts:
                if tact['id_in_video'] not in actors[vname]:
                    actors[vname][tact['id_in_video']] = tact['class']
                tact_label.append(tact['id_in_video'])
                f_actors_bbox[vname][fname][tact['id_in_video']] = tact['bbox']
            tact_label = sorted(tact_label)
            f_actors[vname][fname] = tact_label

            tobjs = g['annotation']['objects']
            tobj_label = []
            f_objects_bbox[vname][fname] = {}
            for tobj in tobjs:
                if tobj['id_in_video'] not in objects[vname]:
                    objects[vname][tobj['id_in_video']] = tobj['class']
                tobj_label.append(tobj['id_in_video'])
                f_objects_bbox[vname][fname][tobj['id_in_video']] = tobj['bbox']
            tobj_label = sorted(tobj_label)
            f_objects[vname][fname] = tobj_label

    for vname in split:

        if os.path.exists(object_image_path + vname):
            continue
        NA = len(actors[vname])
        NO = len(objects[vname])
        NT = len(wframes[vname])

        os.mkdir(object_image_path + vname)

        for i in range(NT):
            fname = wframes[vname][i]
            actn = sorted(actors[vname])
            objn = sorted(objects[vname])
            for j in range(NA):
                act = actn[j]
                if act in f_actors[vname][fname]:
                    assert fname == frame_result[vname][i] + '.jpg'

                    src = os.path.join(dst_frame_path, vname, fname)
                    dst = os.path.join(object_image_path, vname,
                                       os.path.splitext(fname)[0] + '-' + str(j) + '-' + actor_result[vname][
                                           j] + '.jpg')
                    bbox = f_actors_bbox[vname][fname][act]
                    nbbox = compute_bbox(bbox)
                    cut_image(src, dst, nbbox)
            for j in range(NO):
                obj = objn[j]
                if obj in f_objects[vname][fname]:
                    assert fname == frame_result[vname][i] + '.jpg'

                    src = os.path.join(dst_frame_path, vname, fname)
                    dst = os.path.join(object_image_path, vname,
                                       os.path.splitext(fname)[0] + '-' + str(j + NA) + '-' + object_result[vname][
                                           j] + '.jpg')
                    bbox = f_objects_bbox[vname][fname][obj]
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

def write_adjacent(adjacent_path, fps_file, dataset_level_file, object_level_path, relation_level_path, dst_frame_path):

    if not os.path.exists(adjacent_path):
        os.mkdir(adjacent_path)

    write_spa_adjacent(os.path.join(adjacent_path, 'spa_adj'), fps_file, dataset_level_file, dst_frame_path, object_level_path)
    write_tem_adjacent(os.path.join(adjacent_path, 'tem_adj'), dataset_level_file, object_level_path)
    write_rel_adjacent(os.path.join(adjacent_path, 'rel_adj'), dataset_level_file, relation_level_path)

def write_spa_adjacent(adjacent_path, fps_file, dataset_level_file, dst_frame_path, object_level_path):

    if not os.path.exists(adjacent_path):
        os.mkdir(adjacent_path)

    num_result, actor_result, object_result, frame_result = read_label_csv(dataset_level_file)
    fps = read_fps(fps_file)

    graph_anns = load_json('./MOMA_anns/graph_anns.json')

    object_anno = {}
    actors = {}
    objects = {}

    for g in graph_anns:
        vname = g['trim_video_id']

        frames = sorted(os.listdir(dst_frame_path + vname))
        if vname not in object_anno:
            object_anno[vname] = {}
            actors[vname] = {}
            objects[vname] = {}

        tfps = float(fps[vname])
        if tfps > 30:
            tfps = 30
        frame_num = int(g['frame_timestamp'] * tfps + 0.5)
        if frame_num >= len(frames):
            frame_num = len(frames) - 1

        fname = frames[frame_num]
        object_anno[vname][fname] = g['annotation']['relationships']

        tacts = g['annotation']['actors']
        for tact in tacts:
            if tact['id_in_video'] not in actors[vname]:
                actors[vname][tact['id_in_video']] = tact['class']

        tobjs = g['annotation']['objects']
        for tobj in tobjs:
            if tobj['id_in_video'] not in objects[vname]:
                objects[vname][tobj['id_in_video']] = tobj['class']

    for vname in num_result:
        actn = sorted(actors[vname])
        objn = sorted(objects[vname])
        NA = int(num_result[vname][0])
        NO = int(num_result[vname][1])
        NT = int(num_result[vname][2])
        all_spa_adj = np.zeros(((NA + NO) * NT, (NA + NO) * NT), dtype=int)
        object_in_video = np.load(object_level_path + os.path.splitext(vname)[0] + '.npy')
        object_in_video = object_in_video.transpose(1, 0)
        for i in range(NT):
            single_spa_adj = np.zeros(((NA + NO), (NA + NO)), dtype=int)
            sing_oinv = object_in_video[i]
            frame_name = frame_result[vname][i] + '.jpg'
            frame_relation = object_anno[vname][frame_name]
            for si in range(NA + NO):
                for sj in range(NA + NO):
                    if si == sj:
                        single_spa_adj[si][sj] = sing_oinv[si] * sing_oinv[sj]
                    elif si < NA:
                        for ob_rel in frame_relation:
                            descrip = ob_rel['description'][1:-1]
                            tacts = descrip.split('),(')[0].split(',')
                            tobjs = descrip.split('),(')[1].split(',')
                            for tact in tacts:
                                for tobj in tobjs:
                                    if tact in actn and tobj in objn:
                                        ti = actn.index(tact)
                                        tj = objn.index(tobj)

                                        if ti == si and NA + tj == sj:
                                            single_spa_adj[si][sj] = sing_oinv[si] * sing_oinv[sj]
                                            single_spa_adj[sj][si] = sing_oinv[si] * sing_oinv[sj]

            all_spa_adj[i * (NA + NO):(i + 1) * (NA + NO), i * (NA + NO):(i + 1) * (NA + NO)] = single_spa_adj
        np.save(adjacent_path + '/' + os.path.splitext(vname)[0] + '.npy', all_spa_adj)

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
                    if abs(si - sj) < 15:
                        single_spa_adj[si][sj] = sing_oinv[si] * sing_oinv[sj]
            all_spa_adj[i * NT:(i + 1) * NT, i * NT:(i + 1) * NT] = single_spa_adj
        np.save(adjacent_path + '/' + os.path.splitext(vname)[0] + '.npy', all_spa_adj)


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
                    if abs(si - sj) < 7:
                        single_rel_adj[si][sj] = sing_rinv[si] * sing_rinv[sj]
            all_rel_adj[i * NT:(i + 1) * NT, i * NT:(i + 1) * NT] = single_rel_adj
        np.save(adjacent_path + '/' + os.path.splitext(vname)[0] + '.npy', all_rel_adj)