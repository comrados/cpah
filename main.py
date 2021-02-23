import os
import torch
from torch import autograd
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset import Dataset
from config import opt
from models.CPAH import CPAH
from torch.optim import Adam
from utils import calc_map_k, pr_curve, p_top_k, Visualizer, write_pickle, pr_curve2
from datasets.data_handler import load_data, load_pretrain_model
import time
import pickle
import numpy as np


"""
Xie, De, et al. "Multi-Task Consistency-Preserving Adversarial Hashing for Cross-Modal Retrieval."
IEEE Transactions on Image Processing 29 (2020): 3626-3637.
DOI: 10.1109/TMM.2020.2969792
"""


def train(**kwargs):
    since = time.time()
    opt.parse(kwargs)

    if (opt.device is None) or (opt.device == 'cpu'):
        opt.device = torch.device('cpu')
    else:
        opt.device = torch.device(opt.device)

    images, tags, labels = load_data(opt.data_path, type=opt.dataset)
    train_data = Dataset(opt, images, tags, labels)
    train_dataloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True)
    L = train_data.get_labels()
    L = L.to(opt.device)
    # test
    i_query_data = Dataset(opt, images, tags, labels, test='image.query')
    i_db_data = Dataset(opt, images, tags, labels, test='image.db')
    t_query_data = Dataset(opt, images, tags, labels, test='text.query')
    t_db_data = Dataset(opt, images, tags, labels, test='text.db')

    i_query_dataloader = DataLoader(i_query_data, opt.batch_size, shuffle=False)
    i_db_dataloader = DataLoader(i_db_data, opt.batch_size, shuffle=False)
    t_query_dataloader = DataLoader(t_query_data, opt.batch_size, shuffle=False)
    t_db_dataloader = DataLoader(t_db_data, opt.batch_size, shuffle=False)

    query_labels, db_labels = i_query_data.get_labels()
    query_labels = query_labels.to(opt.device)
    db_labels = db_labels.to(opt.device)

    model = CPAH(opt.image_dim, opt.text_dim, opt.hidden_dim, opt.bit, opt.num_label).to(opt.device)

    # discriminator = DisModel(opt.hidden_dim, opt.num_label).to(opt.device)

    optimizer_gen = Adam([
        {'params': model.image_module.parameters()},
        {'params': model.text_module.parameters()},
        {'params': model.hash_module.parameters()},
        {'params': model.mask_module.parameters()},
        {'params': model.consistency_dis.parameters()},
        {'params': model.classifier.parameters()},
    ], lr=opt.lr, weight_decay=0.0005)

    optimizer_dis = Adam(model.feature_dis.parameters(), lr=opt.lr, betas=(0.5, 0.9), weight_decay=0.0001)

    #tri_loss = TripletLoss(opt, reduction='sum')
    loss_bce = torch.nn.BCELoss(reduction='sum')
    loss_ce = torch.nn.CrossEntropyLoss(reduction='sum')

    loss = []
    losses = []

    max_mapi2t = 0.
    max_mapt2i = 0.
    max_mapi2i = 0.
    max_mapt2t = 0.
    max_average = 0.

    mapt2i_list = []
    mapi2t_list = []
    mapi2i_list = []
    mapt2t_list = []
    train_times = []

    B = torch.randn(opt.training_size, opt.bit).sign().to(opt.device)

    H_i = torch.zeros(opt.training_size, opt.bit).to(opt.device)
    H_t = torch.zeros(opt.training_size, opt.bit).to(opt.device)

    torch.autograd.set_detect_anomaly(True)

    for epoch in range(opt.max_epoch):
        t1 = time.time()
        e_loss = 0
        e_losses = {'adv': 0, 'class': 0, 'quant': 0, 'pairwise': 0}
        # for i, (ind, img, txt, label) in tqdm(enumerate(train_dataloader)):
        for i, (ind, img, txt, label) in enumerate(train_dataloader):
            #print(i)
            imgs = img.to(opt.device)
            txt = txt.to(opt.device)
            labels = label.to(opt.device)

            batch_size = len(ind)

            h_img, h_txt, f_rc_img, f_rc_txt, f_rp_img, f_rp_txt = model(imgs, txt)

            H_i[ind, :] = h_img
            H_t[ind, :] = h_txt

            ###################################
            # train discriminator. CPAH paper: (5)
            ###################################
            # IMG - real, TXT - fake
            # train with real (IMG)
            optimizer_dis.zero_grad()

            d_real = model.dis_D(f_rc_img.detach())
            d_real = -torch.log(torch.sigmoid(d_real)).mean()
            d_real.backward()

            # train with fake (TXT)
            d_fake = model.dis_D(f_rc_txt.detach())
            d_fake = -torch.log(torch.ones(batch_size).to(opt.device) - torch.sigmoid(d_fake)).mean()
            d_fake.backward()

            # train with gradient penalty (GP)
            # interpolate real and fake data
            alpha = torch.rand(batch_size, opt.hidden_dim).to(opt.device)
            interpolates = alpha * f_rc_img.detach() + (1 - alpha) * f_rc_txt.detach()
            interpolates.requires_grad_()
            disc_interpolates = model.dis_D(interpolates)
            # get gradients with respect to inputs
            gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                      grad_outputs=torch.ones(disc_interpolates.size()).to(opt.device),
                                      create_graph=True, retain_graph=True, only_inputs=True)[0]
            gradients = gradients.view(gradients.size(0), -1)
            # calculate penalty
            gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10  # 10 is GP hyperparameter
            gradient_penalty.backward()

            optimizer_dis.step()

            ###################################
            # train generator
            ###################################

            # adversarial loss, CPAH paper: (6)
            # IMG is fake now
            loss_adver = -torch.log(torch.sigmoid(model.dis_D(f_rc_img))).mean()  # don't detach from graph

            # consistency classification loss, CPAH paper: (7)
            f_r = torch.vstack([f_rc_img, f_rc_txt, f_rp_img, f_rp_txt])
            l_r = [1] * len(ind) * 2 + [0] * len(ind) + [2] * len(ind)  # labels
            l_r = torch.tensor(l_r).to(opt.device)
            loss_consistency_class = loss_ce(f_r, l_r)

            # classification loss, CPAH paper: (8)
            l_f_rc_img = model.dis_classify(f_rc_img, 'img')
            l_f_rc_txt = model.dis_classify(f_rc_txt, 'txt')
            loss_class = loss_bce(l_f_rc_img, labels) + loss_bce(l_f_rc_txt, labels)
            #loss_class = torch.tensor(0).to(opt.device)

            # pairwise loss, CPAH paper: (10)
            S = (labels.mm(labels.T) > 0).float()
            theta = 0.5 * (h_img.mm(h_txt.T) + h_txt.mm(h_img.T))  # not completely sure
            loss_pairwise = -torch.sum(S*theta - torch.log(1 + torch.exp(theta)))
            #loss_pairwise = torch.tensor(0).to(opt.device)

            # quantization loss, CPAH paper: (11)
            loss_quant = torch.sum(torch.pow(B[ind, :] - h_img, 2)) + torch.sum(torch.pow(B[ind, :] - h_txt, 2))
            #loss_quant = torch.tensor(0).to(opt.device)

            err = loss_adver + opt.alpha * (loss_consistency_class + loss_class) + loss_pairwise + opt.beta * loss_quant

            e_losses['adv'] += loss_adver.detach().cpu().numpy()
            e_losses['class'] += (opt.alpha * (loss_consistency_class + loss_class)).detach().cpu().numpy()
            e_losses['pairwise'] += loss_pairwise.detach().cpu().numpy()
            e_losses['quant'] += loss_quant.detach().cpu().numpy()


            optimizer_gen.zero_grad()
            err.backward(retain_graph=True)
            optimizer_gen.step()

            e_loss = err + e_loss

        loss.append(e_loss.item())
        e_losses['sum'] = sum(e_losses.values())
        losses.append(e_losses)

        B = (0.5 * (H_i.detach() + H_t.detach())).sign()

        delta_t = time.time() - t1
        print('Epoch: {:4d}/{:4d}, time, {:3.3f}s, loss: {:15.3f},'.format(epoch + 1, opt.max_epoch, delta_t,
                                                                           loss[-1]) + 5 * ' ' + 'losses:', e_losses)
        # validate
        if opt.valid and (epoch + 1) % opt.valid_freq == 0:
            mapi2t, mapt2i, mapi2i, mapt2t = valid(model, i_query_dataloader, i_db_dataloader, t_query_dataloader,
                                                   t_db_dataloader, query_labels, db_labels)
            print(
                'Epoch: {:4d}/{:4d}, validation MAP: MAP(i->t) = {:3.4f}, MAP(t->i) = {:3.4f}, MAP(i->i) = {:3.4f}, MAP(t->t) = {:3.4f}'.format(
                    epoch + 1, opt.max_epoch, mapi2t, mapt2i, mapi2i, mapt2t))

            mapi2t_list.append(mapi2t)
            mapt2i_list.append(mapt2i)
            mapi2i_list.append(mapi2i)
            mapt2t_list.append(mapt2t)
            train_times.append(delta_t)

            if 0.5 * (mapi2t + mapt2i) > max_average:
                max_mapi2t = mapi2t
                max_mapt2i = mapt2i
                max_mapi2i = mapi2i
                max_mapt2t = mapt2t
                max_average = 0.5 * (mapi2t + mapt2i)
                save_model(model)
                path = 'checkpoints/' + opt.dataset + '_' + str(opt.bit) + str(opt.proc)
                with torch.cuda.device(opt.device):
                    torch.save([H_i, H_t], os.path.join(path, 'hash_maps_i_t.pth'))
                with torch.cuda.device(opt.device):
                    torch.save(B, os.path.join(path, 'code_map.pth'))

        # decrease the lr to its one fifth every 30 epochs
        if epoch % 30 == 0:
            for params in optimizer_gen.param_groups:
                params['lr'] = max(params['lr'] * 0.2, 1e-6)

        if epoch % 100 == 0:
            pass

    if not opt.valid:
        save_model(model)

    time_elapsed = time.time() - since
    print('\n   Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    if opt.valid:
        print('   Max MAP: MAP(i->t) = {:3.4f}, MAP(t->i) = {:3.4f}, MAP(i->i) = {:3.4f}, MAP(t->t) = {:3.4f}'.format(
            max_mapi2t, max_mapt2i, max_mapi2i, max_mapt2t))
    else:
        mapi2t, mapt2i, mapi2i, mapt2t = valid(model, i_query_dataloader, i_db_dataloader, t_query_dataloader,
                                               t_db_dataloader, query_labels, db_labels)
        print('   Max MAP: MAP(i->t) = {:3.4f}, MAP(t->i) = {:3.4f}, MAP(i->i) = {:3.4f}, MAP(t->t) = {:3.4f}'.format(
            mapi2t, mapt2i, mapi2i, mapt2t))

    res_dict = {'mapi2t': mapi2t_list,
                'mapt2i': mapt2i_list,
                'mapi2i': mapi2i_list,
                'mapt2t': mapt2t_list,
                'epoch_times': train_times,
                'losses': losses}

    path = 'checkpoints/' + opt.dataset + '_' + str(opt.bit) + str(opt.proc)
    write_pickle(os.path.join(path, 'res_dict.pkl'), res_dict)


def test(**kwargs):
    opt.parse(kwargs)

    if opt.device is not None:
        opt.device = torch.device(opt.device)
    elif opt.gpus:
        opt.device = torch.device(0)
    else:
        opt.device = torch.device('cpu')

    # pretrain_model = load_pretrain_model(opt.pretrain_model_path)

    # generator = GEN(opt.image_dim, opt.text_dim, opt.hidden_dim, opt.bit, opt.num_label).to(opt.device)
    model = CPAH(opt.image_dim, opt.text_dim, opt.hidden_dim, opt.bit, opt.num_label).to(opt.device)

    path = 'checkpoints/' + opt.dataset + '_' + str(opt.bit) + str(opt.proc)
    load_model(model, path)

    model.eval()

    images, tags, labels = load_data(opt.data_path, opt.dataset)

    i_query_data = Dataset(opt, images, tags, labels, test='image.query')
    i_db_data = Dataset(opt, images, tags, labels, test='image.db')
    t_query_data = Dataset(opt, images, tags, labels, test='text.query')
    t_db_data = Dataset(opt, images, tags, labels, test='text.db')

    i_query_dataloader = DataLoader(i_query_data, opt.batch_size, shuffle=False)
    i_db_dataloader = DataLoader(i_db_data, opt.batch_size, shuffle=False)
    t_query_dataloader = DataLoader(t_query_data, opt.batch_size, shuffle=False)
    t_db_dataloader = DataLoader(t_db_data, opt.batch_size, shuffle=False)

    qBX = generate_img_code(model, i_query_dataloader, opt.query_size)
    qBY = generate_txt_code(model, t_query_dataloader, opt.query_size)
    rBX = generate_img_code(model, i_db_dataloader, opt.db_size)
    rBY = generate_txt_code(model, t_db_dataloader, opt.db_size)

    query_labels, db_labels = i_query_data.get_labels()
    query_labels = query_labels.to(opt.device)
    db_labels = db_labels.to(opt.device)

    K = [1, 10, 100, 1000]
    p_top_k(qBX, rBY, query_labels, db_labels, K, tqdm_label='I2T')
    # pr_curve2(qBY, rBX, query_labels, db_labels)

    p_i2t, r_i2t = pr_curve(qBX, rBY, query_labels, db_labels, tqdm_label='I2T')
    p_t2i, r_t2i = pr_curve(qBY, rBX, query_labels, db_labels, tqdm_label='T2I')
    p_i2i, r_i2i = pr_curve(qBX, rBX, query_labels, db_labels, tqdm_label='I2I')
    p_t2t, r_t2t = pr_curve(qBY, rBY, query_labels, db_labels, tqdm_label='T2T')

    K = [1, 10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    pk_i2t = p_top_k(qBX, rBY, query_labels, db_labels, K, tqdm_label='I2T')
    pk_t2i = p_top_k(qBY, rBX, query_labels, db_labels, K, tqdm_label='T2I')
    pk_i2i = p_top_k(qBX, rBX, query_labels, db_labels, K, tqdm_label='I2I')
    pk_t2t = p_top_k(qBY, rBY, query_labels, db_labels, K, tqdm_label='T2T')

    mapi2t = calc_map_k(qBX, rBY, query_labels, db_labels)
    mapt2i = calc_map_k(qBY, rBX, query_labels, db_labels)
    mapi2i = calc_map_k(qBX, rBX, query_labels, db_labels)
    mapt2t = calc_map_k(qBY, rBY, query_labels, db_labels)

    pr_dict = {'pi2t': p_i2t.cpu().numpy(), 'ri2t': r_i2t.cpu().numpy(),
               'pt2i': p_t2i.cpu().numpy(), 'rt2i': r_t2i.cpu().numpy(),
               'pi2i': p_i2i.cpu().numpy(), 'ri2i': r_i2i.cpu().numpy(),
               'pt2t': p_t2t.cpu().numpy(), 'rt2t': r_t2t.cpu().numpy()}

    pk_dict = {'k': K,
               'pki2t': pk_i2t.cpu().numpy(),
               'pkt2i': pk_t2i.cpu().numpy(),
               'pki2i': pk_i2i.cpu().numpy(),
               'pkt2t': pk_t2t.cpu().numpy()}

    map_dict = {'mapi2t': float(mapi2t.cpu().numpy()),
                'mapt2i': float(mapt2i.cpu().numpy()),
                'mapi2i': float(mapi2i.cpu().numpy()),
                'mapt2t': float(mapt2t.cpu().numpy())}

    print('   Test MAP: MAP(i->t) = {:3.4f}, MAP(t->i) = {:3.4f}, MAP(i->i) = {:3.4f}, MAP(t->t) = {:3.4f}'.format(mapi2t, mapt2i, mapi2i, mapt2t))

    path = 'checkpoints/' + opt.dataset + '_' + str(opt.bit) + str(opt.proc)
    write_pickle(os.path.join(path, 'pr_dict.pkl'), pr_dict)
    write_pickle(os.path.join(path, 'pk_dict.pkl'), pk_dict)
    write_pickle(os.path.join(path, 'map_dict.pkl'), map_dict)


def generate_img_code(model, test_dataloader, num):
    B = torch.zeros(num, opt.bit).to(opt.device)

    # for i, input_data in tqdm(enumerate(test_dataloader)):
    for i, input_data in enumerate(test_dataloader):
        input_data = input_data.to(opt.device)
        b = model.generate_img_code(input_data)
        idx_end = min(num, (i + 1) * opt.batch_size)
        B[i * opt.batch_size: idx_end, :] = b.data

    B = torch.sign(B)
    return B


def generate_txt_code(model, test_dataloader, num):
    B = torch.zeros(num, opt.bit).to(opt.device)

    # for i, input_data in tqdm(enumerate(test_dataloader)):
    for i, input_data in enumerate(test_dataloader):
        input_data = input_data.to(opt.device)
        b = model.generate_txt_code(input_data)
        idx_end = min(num, (i + 1) * opt.batch_size)
        B[i * opt.batch_size: idx_end, :] = b.data

    B = torch.sign(B)
    return B


def valid(model, x_query_dataloader, x_db_dataloader, y_query_dataloader, y_db_dataloader, query_labels, db_labels):
    model.eval()

    qBX = generate_img_code(model, x_query_dataloader, opt.query_size)
    qBY = generate_txt_code(model, y_query_dataloader, opt.query_size)
    rBX = generate_img_code(model, x_db_dataloader, opt.db_size)
    rBY = generate_txt_code(model, y_db_dataloader, opt.db_size)

    mapi2t = calc_map_k(qBX, rBY, query_labels, db_labels)
    mapt2i = calc_map_k(qBY, rBX, query_labels, db_labels)

    mapi2i = calc_map_k(qBX, rBX, query_labels, db_labels)
    mapt2t = calc_map_k(qBY, rBY, query_labels, db_labels)

    model.train()
    return mapi2t.item(), mapt2i.item(), mapi2i.item(), mapt2t.item()


def load_model(model, path):
    if path is not None:
        model.load(os.path.join(path, model.module_name + '.pth'))


def save_model(model):
    path = 'checkpoints/' + opt.dataset + '_' + str(opt.bit) + str(opt.proc)
    model.save(model.module_name + '.pth', path, cuda_device=opt.device)


if __name__ == '__main__':
    import fire

    fire.Fire()
