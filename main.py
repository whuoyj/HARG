from dataloader import iterator_factory
from metric import *
from model import *
from config import config


import torch
import torch.nn as nn
from torch import save, load, no_grad
import time
import random

def val_relation(model, val_loader, device):
    model.eval()
    loss_sum = 0
    acc_sum1 = 0
    acc_sum2 = 0
    acc_sum3 = 0
    acc_sum4 = 0
    all_preds1 = []
    all_preds2 = []
    all_preds3 = []
    all_preds4 = []
    all_trues1 = []
    all_trues2 = []
    all_trues3 = []
    all_trues4 = []


    with no_grad():
        for i_batch, (adjacent_o, data_o, adjacent_r, data_r, target1, target2, target3, target4, video_name) in enumerate(val_loader):
            adjacent_o = adjacent_o.to(device).float()
            data_o = data_o.to(device)
            adjacent_r = adjacent_r.to(device).float()
            data_r = data_r.to(device)
            target1 = target1.to(device)
            target2 = target2.to(device)
            target3 = target3.to(device)
            target4 = target4.to(device)

            predict_logits1, targets1, predict_logits2, targets2, predict_logits3, targets3, predict_logits4, targets4, loss = model(adjacent_o, data_o, adjacent_r, data_r, target1, target2, target3, target4, device)

            loss_sum += loss.item()
            acc_sum1 += get_mAP(predict_logits1, targets1)
            all_preds1.extend(predict_logits1)
            all_trues1.extend(targets1)

            acc_sum2 += get_mAP(predict_logits2, targets2)
            all_preds2.extend(predict_logits2)
            all_trues2.extend(targets2)

            acc_sum3 += get_mAP(predict_logits3, targets3)
            all_preds3.extend(predict_logits3)
            all_trues3.extend(targets3)

            acc_sum4 += get_mAP(predict_logits4, targets4)
            all_preds4.extend(predict_logits4.detach())
            all_trues4.extend(targets4.detach())

        val_loss = loss_sum / i_batch

        all_preds1 = torch.stack(all_preds1)
        all_trues1 = torch.stack(all_trues1)
        all_preds2 = torch.stack(all_preds2)
        all_trues2 = torch.stack(all_trues2)
        all_preds3 = torch.stack(all_preds3)
        all_trues3 = torch.stack(all_trues3)
        all_preds4 = torch.stack(all_preds4)
        all_trues4 = torch.stack(all_trues4)

        val_acc1 = get_acc(all_preds1, all_trues1)
        val_acc2 = get_acc(all_preds2, all_trues2)
        val_acc3 = get_acc(all_preds3, all_trues3)
        val_acc4 = get_acc(all_preds4, all_trues4)
        val_mAP1 = get_mAP(all_preds1, all_trues1)
        val_mAP2 = get_mAP(all_preds2, all_trues2)
        val_mAP3 = get_mAP(all_preds3, all_trues3)
        val_mAP4 = get_mAP(all_preds4, all_trues4)

    return val_loss, val_acc1, val_acc2[0], val_acc2[1], val_acc3[0], val_acc3[1], val_acc4, val_mAP1, val_mAP2, val_mAP3, val_mAP4

def train_relation(model, optimizer, train_loader, device):
    t1 = time.time()
    loss_sum = 0
    acc_sum1 = 0
    acc_sum2 = 0
    acc_sum3 = 0
    acc_sum4 = 0
    all_preds1 = []
    all_preds2 = []
    all_preds3 = []
    all_preds4 = []
    all_trues1 = []
    all_trues2 = []
    all_trues3 = []
    all_trues4 = []
    model.train()

    if config['models'] != "":
        model.VRD.fc_vis.bn.eval()
    for i_batch, (adjacent_o, data_o, adjacent_r, data_r, target1, target2, target3, target4, video_name) in enumerate(train_loader):
        adjacent_o = adjacent_o.to(device).float()
        data_o = data_o.to(device)
        adjacent_r = adjacent_r.to(device).float()
        data_r = data_r.to(device)
        target1 = target1.to(device)
        target2 = target2.to(device)
        target3 = target3.to(device)
        target4 = target4.to(device)

        predict_logits1, targets1, predict_logits2, targets2, predict_logits3, targets3, predict_logits4, targets4, loss = model(adjacent_o, data_o, adjacent_r, data_r, target1, target2, target3, target4, device)

        loss_sum += loss.item()
        acc_sum1 += get_mAP(predict_logits1, targets1)
        all_preds1.extend(predict_logits1.detach())
        all_trues1.extend(targets1.detach())

        acc_sum2 += get_mAP(predict_logits2, targets2)
        all_preds2.extend(predict_logits2.detach())
        all_trues2.extend(targets2.detach())

        acc_sum3 += get_mAP(predict_logits3, targets3)
        all_preds3.extend(predict_logits3.detach())
        all_trues3.extend(targets3.detach())

        acc_sum4 += get_mAP(predict_logits4, targets4)
        all_preds4.extend(predict_logits4.detach())
        all_trues4.extend(targets4.detach())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        if i_batch % config['display_batch_interval'] == 0 and i_batch != 0:
            t2 = time.time()
            print('Epoch %d, Batch %d, loss = %.4f, acc1 = %.4f, acc2 = %.4f, acc3 = %.4f, %.3f seconds/batch' % (
                epoch, i_batch, loss_sum / i_batch, acc_sum1 / i_batch, acc_sum2 / i_batch, acc_sum3 / i_batch, (t2 - t1) / config['display_batch_interval']
            ))
            t1 = t2


    train_loss = loss_sum / i_batch

    all_preds1 = torch.stack(all_preds1)
    all_trues1 = torch.stack(all_trues1)
    all_preds2 = torch.stack(all_preds2)
    all_trues2 = torch.stack(all_trues2)
    all_preds3 = torch.stack(all_preds3)
    all_trues3 = torch.stack(all_trues3)
    all_preds4 = torch.stack(all_preds4)
    all_trues4 = torch.stack(all_trues4)

    train_acc1 = get_acc(all_preds1, all_trues1)
    train_acc2 = get_acc(all_preds2, all_trues2)
    train_acc3 = get_acc(all_preds3, all_trues3)
    train_acc4 = get_acc(all_preds4, all_trues4)
    train_mAP1 = get_mAP(all_preds1, all_trues1)
    train_mAP2 = get_mAP(all_preds2, all_trues2)
    train_mAP3 = get_mAP(all_preds3, all_trues3)
    train_mAP4 = get_mAP(all_preds4, all_trues4)

    return train_loss, train_acc1, train_acc2[0], train_acc2[1], train_acc3[0], train_acc3[1], train_acc4, train_mAP1, train_mAP2, train_mAP3, train_mAP4

def load_state(model, net_state_keys):
    state_dict = torch.load(config['model_load_name1'], map_location=config['cuda'])

    for name, param in state_dict.items():
        if name in model.state_dict().keys():
            dst_param_shape = model.state_dict()[name].shape
            if param.shape == dst_param_shape:
                model.state_dict()[name].copy_(param.view(dst_param_shape))
                net_state_keys.remove(name)


def save_model(epoch_val, epoch_test, best, epoch, model, facc, save_file):
    v = False
    if epoch_test > best:
        v = True
        best = epoch_test
        save(model.state_dict(), config["model_save_path"] + 'model_bset_' + save_file + '.pth')
        facc.write("best epoch: %d" % (epoch) +  " val " + save_file + ": %.4f" % (epoch_val))
        facc.write('\n')
        if config['dataset'] == "MOMA-LRG":
            facc.write("best epoch: %d" % (epoch) + " test " + save_file + ": %.4f" % (epoch_test))
            facc.write('\n')
    return best, v

def seed_torch(seed = 0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True


if __name__ == '__main__':

    seed_torch()

    device = torch.device(config['cuda'] if torch.cuda.is_available() else 'cpu')

    model = GGCN_relation(visual_dim=config['visual_dim'],
                          relation_dim=config['num_class3'],
                          feat_dim=config['feat_dims'], num_v=config['num_nodes'],
                          dropout=config['dropout_rate'])


    a_params = list(map(id, model.fc2.parameters()))
    a_params += list(map(id, model.TCN.parameters()))
    rest_params = filter(lambda x:id(x) not in a_params, model.parameters())

    optimizer = torch.optim.Adam([{'params': rest_params, 'lr': config['learning_rate']},
                                  {'params': model.TCN.parameters(), 'lr': config['learning_rate'] * 10},
                                  {'params': model.fc2.parameters(), 'lr': config['learning_rate'] * 0.1}
                                  ])
    criterion = nn.CrossEntropyLoss()

    def adjust_learning_rate(decay_rate=0.8):
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay_rate


    net_state_keys = list(model.state_dict().keys())

    if "val" in config['mode']:
        load_state(model, net_state_keys)

    for name, param in model.named_parameters():
        if name not in net_state_keys:
            param.requires_grad = False
        else:
            print(name)

    model = model.to(device)

    data_loader = iterator_factory.creat(config["batch_size"], config["data_root"])

    best_loss = 10
    best_acc1 = 0.0
    best_acc2 = 0.0
    best_acc3 = 0.0
    best_epoch = 1


    for epoch in range(1, config['max_epoches'] + 1):

        facc = open(config["acc_file"], 'a')

        if "train" in config['mode']:
            train_results = train_relation(model, optimizer, data_loader[0], device)

            facc.write(
                "epoch: %d train loss: %.4f, train accuracy1: %.4f, train accuracy2 top@1: %.4f, train accuracy2 top@5: %.4f, train accuracy3 top@1: %.4f, train accuracy3 top@5: %.4f, train accuracy4: %.4f, train mAP1: %.4f, train mAP2: %.4f, train mAP3: %.4f, train mAP4: %.4f" % (
                    epoch, train_results[0], train_results[1], train_results[2], train_results[3], train_results[4],
                    train_results[5], train_results[6], train_results[7], train_results[8], train_results[9],
                    train_results[10]))
            facc.write('\n')

        print("val")
        val_results = val_relation(model, data_loader[1], device)

        print(
            "epoch: %d val loss: %.4f, val accuracy1: %.4f, val accuracy2 top@1: %.4f, val accuracy2 top@5: %.4f, val accuracy3 top@1: %.4f, val accuracy3 top@5: %.4f, val accuracy4: %.4f, val mAP1: %.4f, val mAP2: %.4f, val mAP3: %.4f, val mAP4: %.4f" % (
            epoch, val_results[0], val_results[1], val_results[2], val_results[3], val_results[4], val_results[5],
            val_results[6], val_results[7], val_results[8], val_results[9], val_results[10]))

        facc.write(
            "epoch: %d val loss: %.4f, val accuracy1: %.4f, val accuracy2 top@1: %.4f, val accuracy2 top@5: %.4f, val accuracy3 top@1: %.4f, val accuracy3 top@5: %.4f, val accuracy4: %.4f, val mAP1: %.4f, val mAP2: %.4f, val mAP3: %.4f, val mAP4: %.4f" % (
            epoch, val_results[0], val_results[1], val_results[2], val_results[3], val_results[4], val_results[5],
            val_results[6], val_results[7], val_results[8], val_results[9], val_results[10]))
        facc.write('\n')
        
        if config['dataset'] == "MOMA-LRG":

            best_acc1, v1 = save_model(val_results[7], val_results[7], best_acc1, epoch, model, facc, 'mAP1')
            best_acc2, v2 = save_model(val_results[2], val_results[2], best_acc2, epoch, model, facc, 'acc2')
            best_acc3, v3 = save_model(val_results[4], val_results[4], best_acc3, epoch, model, facc, 'acc3')
            if v2:
                wacc1, wacc2, wacc3, wacc4 = val_results[7], val_results[2], val_results[4], val_results[10]

        else:

            best_acc1, v1 = save_model(val_results[7], val_results[7], best_acc1, epoch, model, facc, 'mAP1')
            best_acc2, v2 = save_model(val_results[8], val_results[8], best_acc2, epoch, model, facc, 'mAP2')
            best_acc3, v3 = save_model(val_results[9], val_results[9], best_acc3, epoch, model, facc, 'mAP3')
            if v2:
                wacc1, wacc2, wacc3, wacc4 = val_results[7], val_results[8], val_results[9], val_results[10]

        if epoch == 50:
            save(model.state_dict(), config["model_save_path"] + 'model_lsset_' + str(epoch) + '.pth')

        facc.write('\n')
        if epoch % int(config['weight_decay_step']) == 0:
            adjust_learning_rate(config['weight_decay_rate'])

        facc.close()

    print("best acc1:")
    print(best_acc1)
    print("best acc2:")
    print(best_acc2)
    print("best acc3:")
    print(best_acc3)
