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
    ], lr=opt.lr, weight_decay=0.0005)

    optimizer_dis = {
        'feature': Adam(model.feature_dis.parameters(), lr=opt.lr, betas=(0.5, 0.9), weight_decay=0.0001),
        'cons': Adam(model.consistency_dis.parameters(), lr=opt.lr, betas=(0.5, 0.9), weight_decay=0.0001),
        'class': Adam(model.classifier.parameters(), lr=opt.lr, betas=(0.5, 0.9), weight_decay=0.0001)
    }

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

    for epoch in range(opt.max_epoch):
        t1 = time.time()
        e_loss = 0
        e_losses = {'adv': 0, 'tri': 0, 'quant': 0}
        # for i, (ind, img, txt, label) in tqdm(enumerate(train_dataloader)):
        for i, (ind, img, txt, label) in enumerate(train_dataloader):
            imgs = img.to(opt.device)
            txt = txt.to(opt.device)
            labels = label.to(opt.device)

            batch_size = len(ind)

            h_img, h_txt, f_rc_img, f_rc_txt, f_rp_img, f_rp_txt = model(imgs, txt)

            B[ind, :] = (0.5 * (h_img + h_txt)).sign()

            ###################################
            # train discriminator. CPAH paper: (5)
            ###################################
            # IMG - real, TXT - fake
            # train with real (IMG)
            optimizer_dis['feature'].zero_grad()

            d_real = model.dis_D(f_rc_img.detach())
            d_real = -opt.gamma * torch.log(torch.sigmoid(d_real)).mean()
            d_real.backward()

            # train with fake (TXT)
            d_fake = model.dis_D(f_rc_txt.detach())
            d_fake = -opt.gamma * torch.log(torch.ones(batch_size).to(opt.device) - torch.sigmoid(d_fake)).mean()
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
            feature_gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10  # 10 is GP hyperparameter
            feature_gradient_penalty.backward()

            optimizer_dis['feature'].step()

            ###################################
            # train generator
            ###################################

            # adversarial loss, CPAH paper: (6)
            # IMG is fake now
            loss_adver = -torch.log(torch.sigmoid(model.dis_D(f_rc_img))).mean()  # don't detach from graph

            # consistency classification loss, CPAH paper: (7)
            f_r = torch.vstack([f_rc_img, f_rc_txt, f_rp_img, f_rp_txt])
            l_r = [1] * opt.batch_size * 2 + [0] * opt.batch_size + [2] * opt.batch_size  # labels
            l_r = torch.tensor(l_r).to(opt.device)
            loss_consistency_class = loss_ce(f_r, l_r)

            # classification loss, CPAH paper: (8)
            l_f_rc_img = model.dis_classify(f_rc_img, 'img')
            l_f_rc_txt = model.dis_classify(f_rc_txt, 'txt')
            # l_f_rp_img = model.dis_classify(f_rp_img, 'img')
            # l_f_rp_txt = model.dis_classify(f_rp_txt, 'txt')
            loss_class = loss_bce(l_f_rc_img, labels) + loss_bce(l_f_rc_txt, labels)

            # pairwise loss, CPAH paper: (10)
            S = (labels.mm(labels.T) > 0).float()
            theta = 0.5 * (h_img.mm(h_img.T) + h_txt.mm(h_txt.T))  # not completely sure
            loss_pairwise = -torch.sum(S*theta - torch.log(1 + torch.exp(theta)))

            # quantization loss, CPAH paper: (11)
            loss_quant = torch.sum(torch.pow(B[ind, :] - h_img, 2)) + torch.sum(torch.pow(B[ind, :] - h_txt, 2))


            ############################################################## TODO calc losses, use backprop

            H_i[ind, :] = h_i.data
            H_t[ind, :] = h_t.data
            h_t_detach = model.generate_txt_code(txt)

            #####
            # train feature discriminator
            #####
            d_real = discriminator.dis_feature(f_i.detach())
            d_real = -opt.gamma * torch.log(torch.sigmoid(d_real)).mean()
            # D_real_feature = -D_real_feature.mean()
            optimizer_dis['feature'].zero_grad()
            d_real.backward()

            # train with fake
            d_fake = discriminator.dis_feature(f_t.detach())
            d_fake = -opt.gamma * torch.log(
                torch.ones(batch_size).to(opt.device) - torch.sigmoid(d_fake)).mean()
            # D_fake_feature = D_fake_feature.mean()
            d_fake.backward()

            # train with gradient penalty (GP)
            # interpolate real and fake data
            alpha = torch.rand(batch_size, opt.hidden_dim // 4).to(opt.device)
            interpolates = alpha * f_i.detach() + (1 - alpha) * f_t.detach()
            interpolates.requires_grad_()
            disc_interpolates = discriminator.dis_feature(interpolates)
            # get gradients with respect to inputs
            gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                      grad_outputs=torch.ones(disc_interpolates.size()).to(opt.device),
                                      create_graph=True, retain_graph=True, only_inputs=True)[0]
            gradients = gradients.view(gradients.size(0), -1)
            # calculate penalty
            feature_gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10  # 10 is GP hyperparameter
            feature_gradient_penalty.backward()

            optimizer_dis['feature'].step()

            #####
            # train hash discriminator
            #####
            D_real_hash = discriminator.dis_hash(h_i.detach())
            D_real_hash = -opt.gamma * torch.log(torch.sigmoid(D_real_hash)).mean()
            optimizer_dis['hash'].zero_grad()
            D_real_hash.backward()

            # train with fake
            D_fake_hash = discriminator.dis_hash(h_t.detach())
            D_fake_hash = -opt.gamma * torch.log(
                torch.ones(batch_size).to(opt.device) - torch.sigmoid(D_fake_hash)).mean()
            D_fake_hash.backward()

            # train with gradient penalty
            alpha = torch.rand(batch_size, opt.bit).to(opt.device)
            interpolates = alpha * h_i.detach() + (1 - alpha) * h_t.detach()
            interpolates.requires_grad_()
            disc_interpolates = discriminator.dis_hash(interpolates)
            gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                      grad_outputs=torch.ones(disc_interpolates.size()).to(opt.device),
                                      create_graph=True, retain_graph=True, only_inputs=True)[0]
            gradients = gradients.view(gradients.size(0), -1)

            hash_gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
            hash_gradient_penalty.backward()

            optimizer_dis['hash'].step()

            loss_G_txt_feature = -torch.log(torch.sigmoid(discriminator.dis_feature(f_t))).mean()
            loss_adver_feature = loss_G_txt_feature

            loss_G_txt_hash = -torch.log(torch.sigmoid(discriminator.dis_hash(h_t_detach))).mean()
            loss_adver_hash = loss_G_txt_hash

            tri_i2t = tri_loss(h_i, labels, target=h_t, margin=opt.margin)
            tri_t2i = tri_loss(h_t, labels, target=h_i, margin=opt.margin)
            weighted_cos_tri = tri_i2t + tri_t2i

            i_ql = torch.sum(torch.pow(B_i[ind, :] - h_i, 2))
            t_ql = torch.sum(torch.pow(B_t[ind, :] - h_t, 2))
            loss_quant = i_ql + t_ql
            err = opt.alpha * weighted_cos_tri + opt.beta * loss_quant + opt.gamma * (
                        loss_adver_feature + loss_adver_hash)

            e_losses['adv'] += (opt.gamma * (loss_adver_feature + loss_adver_hash)).cpu().detach().numpy()
            e_losses['tri'] += (opt.alpha * weighted_cos_tri).cpu().detach().numpy()
            e_losses['quant'] += (opt.beta * loss_quant).cpu().detach().numpy()
            # print((opt.alpha * weighted_cos_tri).cpu().detach().numpy(), (opt.beta * loss_quant).cpu().detach().numpy(), (opt.gamma * (loss_adver_feature + loss_adver_hash)).cpu().detach().numpy())

            optimizer_gen.zero_grad()
            err.backward()
            optimizer_gen.step()

            e_loss = err + e_loss

        loss.append(e_loss.item())
        e_losses['sum'] = sum(e_losses.values())
        losses.append(e_losses)

        P_i = torch.inverse(L.t() @ L + opt.lamb * torch.eye(opt.num_label, device=opt.device)) @ L.t() @ B_i
        P_t = torch.inverse(L.t() @ L + opt.lamb * torch.eye(opt.num_label, device=opt.device)) @ L.t() @ B_t

        B_i = (L @ P_i + opt.mu * H_i).sign()
        B_t = (L @ P_t + opt.mu * H_t).sign()

        delta_t = time.time() - t1
        print('Epoch: {:4d}/{:4d}, time, {:3.3f}s, loss: {:15.3f},'.format(epoch + 1, opt.max_epoch, delta_t,
                                                                           loss[-1]) + 5 * ' ' + 'losses:', e_losses)

        if opt.vis_env:
            vis.plot('loss', loss[-1])

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
                    torch.save([P_i, P_t], os.path.join(path, 'feature_maps_i_t.pth'))
                with torch.cuda.device(opt.device):
                    torch.save([B_i, B_t], os.path.join(path, 'code_maps_i_t.pth'))

            if opt.vis_env:
                vis.plot('mapi2t', mapi2t)
                vis.plot('mapt2i', mapt2i)

        if epoch % 50 == 0:
            for params in optimizer_gen.param_groups:
                params['lr'] = max(params['lr'] * 0.8, 1e-6)

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


def load_model(model, path):
    if path is not None:
        model.load(os.path.join(path, model.module_name + '.pth'))


def save_model(model):
    path = 'checkpoints/' + opt.dataset + '_' + str(opt.bit) + str(opt.proc)
    model.save(model.module_name + '.pth', path, cuda_device=opt.device)


if __name__ == '__main__':
    import fire

    fire.Fire()
