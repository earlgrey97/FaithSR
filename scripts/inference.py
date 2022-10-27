import os
from argparse import Namespace

from tqdm import tqdm
import time
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
import sys
from torchvision import datasets ##

#print("this is inference.py file")
sys.path.append(".")
sys.path.append("..")

from configs import data_configs
from datasets.inference_dataset import InferenceDataset
from utils.common import tensor2im, log_input_image
from options.test_options import TestOptions
from models.psp import pSp

# global variable
sigmas_to_avg = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
hr_avg_sigma = [2.953271722793579, 3.435314989089966, 6.067221355438233, 4.2303853034973145, 6.539189243316651, 6.438661766052246, 7.886064910888672, 4.518422603607178, 3.8808063507080077, 5.771304893493652, 2.7676040172576903, 3.7998706817626955, 2.212826657295227, 2.051713490486145, 1.3500442981719971, 1.3690280437469482, 0.6566168785095214, 0.8966994047164917]   

def run():
    #print("this is inference.py/run function")
    test_opts = TestOptions().parse()

    if test_opts.resize_factors is not None:
        assert len(
            test_opts.resize_factors.split(',')) == 1, "When running inference, provide a single downsampling factor!"
        out_path_results = os.path.join(test_opts.exp_dir, 'inference_results',
                                        'downsampling_{}'.format(test_opts.resize_factors))
        out_path_coupled = os.path.join(test_opts.exp_dir, 'inference_coupled',
                                        'downsampling_{}'.format(test_opts.resize_factors))
    else:
        out_path_results = os.path.join(test_opts.exp_dir, 'inference_results')
        out_path_coupled = os.path.join(test_opts.exp_dir, 'inference_coupled')

    os.makedirs(out_path_results, exist_ok=True)
    os.makedirs(out_path_coupled, exist_ok=True)

    # update test options with options used during training
    ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
    opts = ckpt['opts']
    opts.update(vars(test_opts))
    if 'learn_in_w' not in opts:
        opts['learn_in_w'] = False
    if 'output_size' not in opts:
        opts['output_size'] = 1024
    opts = Namespace(**opts)

    net = pSp(opts)
    net.eval()
    net.cuda()

    print('Loading dataset for {}'.format(opts.dataset_type))
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    dataset = InferenceDataset(root=opts.data_path,
                               transform=transforms_dict['transform_inference'],
                               opts=opts)
    
    dataloader = DataLoader(dataset,
                            batch_size=opts.test_batch_size,
                            shuffle=False,
                            num_workers=int(opts.test_workers),
                            drop_last=True)
    
    if opts.n_images is None:
        opts.n_images = len(dataset)

    global_i = 0
    global_time = []
                
    for input_batch, input_path in tqdm(dataloader):##
        ##
        #print('----- input path -----')
        if opts.save_latents:
            file_nums = [input_path[0].split('/')[-1].replace('.jpg',''), input_path[1].split('/')[-1].replace('.jpg',''), input_path[2].split('/')[-1].replace('.jpg',''), input_path[3].split('/')[-1].replace('.jpg','')]
        #print(file_nums)
        ##
        if global_i >= opts.n_images:
            break
        with torch.no_grad():
            input_cuda = input_batch.cuda().float()
            tic = time.time()
            ## 
            result_batch, latent = run_on_batch(input_cuda, net, opts)
            ##
            toc = time.time()
            global_time.append(toc - tic)

        for i in range(opts.test_batch_size):
            ##
            if opts.save_latents:
                torch.save(latent[i], f"w_vecs/{file_nums[i]}.pt")
            ##
            result = tensor2im(result_batch[i])
            im_path = dataset.paths[global_i]

            if opts.couple_outputs or global_i % 100 == 0:
                input_im = log_input_image(input_batch[i], opts)
                resize_amount = (256, 256) if opts.resize_outputs else (opts.output_size, opts.output_size)
                if opts.resize_factors is not None:
                    # for super resolution, save the original, down-sampled, and output
                    source = Image.open(im_path)
                    res = np.concatenate([np.array(source.resize(resize_amount)),
                                          np.array(input_im.resize(resize_amount, resample=Image.NEAREST)),
                                          np.array(result.resize(resize_amount))], axis=1)
                else:
                    # otherwise, save the original and output
                    res = np.concatenate([np.array(input_im.resize(resize_amount)),
                                          np.array(result.resize(resize_amount))], axis=1)
                Image.fromarray(res).save(os.path.join(out_path_coupled, os.path.basename(im_path)))

            im_save_path = os.path.join(out_path_results, os.path.basename(im_path))
            Image.fromarray(np.array(result)).save(im_save_path)

            global_i += 1

    stats_path = os.path.join(opts.exp_dir, 'stats.txt')
    result_str = 'Runtime {:.4f}+-{:.4f}'.format(np.mean(global_time), np.std(global_time))
    print(result_str)

    with open(stats_path, 'w') as f:
        f.write(result_str)

    ## print avg of sigmas_to_avg
    sigmas_hr_avg = []
    for i in range(18):
        sigmas_hr_avg.append(np.average(sigmas_to_avg[i]))
    #print("hr avg sigma: ", sigmas_hr_avg)
##
def calc_code(w_conf, sigma_conf, w_i, sigma_i): # w is 512 * 1
    # sigma_conf < sigma_i for all i in each level
    #print("--- calc ---")
    #print(sigma_conf, sigma_i)
    sigma_sum = sigma_i + sigma_conf
    # print("sigma_i", sigma_i)
    # print("sigma_conf", sigma_conf)
    # print("sigma_sum", sigma_sum)

    # print("w_i[0]", w_i[0])
    # print("w_conf[0]", w_conf[0])
    new_code = (sigma_i/sigma_sum)*w_i + (sigma_conf/sigma_sum)*w_conf
    # print("new_code[0]", new_code[0])
    return new_code

def normalize_sigma(sigma): # sigma is list with len 18
    new_sigma = []
    for i in range(18):
        new_sigma.append(sigma[i]/hr_avg_sigma[i])
    return new_sigma

def latent_ops(codes, sigma):
    #print(codes.shape)
    sigma = sigma.tolist()
    # print("sigma", sigma)
    #sigma = normalize_sigma(sigma)
    # print("norm sigma", sigma)
    #print("----- do latent operations -----")
    #print("sigma: ", sigma)
    # print("codes")
    # print(codes)
    # print(codes.shape)
    # print("sigma")
    # print(sigma)
    coarse_ind = 3
    middle_ind = 7
    style_count = 18
	# get confident vectors, sigma values of each level
    conf_c = min(sigma[:coarse_ind]) # sigma value of confident vector
    #print("conf_c: ", conf_c)
    sigma_c = sigma.index(conf_c) # index of confident vecotor
    #print("sigma_c: ", sigma_c)
    conf_m = min(sigma[coarse_ind:middle_ind])
    sigma_m = sigma.index(conf_m)
    conf_f = min(sigma[middle_ind:])
    sigma_f = sigma.index(conf_f)
    print(sigma_c, sigma_m, sigma_f)
	# new w vectors
    new_codes = []
	## coarse
    for i in range(coarse_ind): # 0,1,2
        new_codes.append(calc_code(codes[sigma_c], sigma[sigma_c], codes[i], sigma[i]))
	## middle
    for j in range(coarse_ind, middle_ind): # 3,4,5,6
        new_codes.append(calc_code(codes[sigma_m], sigma[sigma_m], codes[j], sigma[j]))
	## fine
    for k in range(middle_ind, style_count): # 7 - 18
        new_codes.append(calc_code(codes[sigma_f], sigma[sigma_f], codes[k], sigma[k]))
	## res: 18 * 512
    #print("original code : ", codes[0])
    #print("new_codes : ", new_codes[0])
    res = torch.stack((new_codes[0],new_codes[1],new_codes[2],new_codes[3],new_codes[4],new_codes[5],
                        new_codes[6],new_codes[7],new_codes[8],new_codes[9],new_codes[10],new_codes[11],
                        new_codes[12],new_codes[13],new_codes[14],new_codes[15],new_codes[16],new_codes[17]))
    print(res.shape)
    return torch.reshape(res,(1,18,512))
##

def run_on_batch(inputs, net, opts):
    #torch.manual_seed(seed)
    #print('seed', seed)
    if opts.save_latents:
        result_batch, mid_latent = net(inputs, randomize_noise=False, resize=opts.resize_outputs, mid_latent = True)
        return result_batch, mid_latent

    if opts.latent_mask is None:
        result_batch, latent = net(inputs, randomize_noise=False, resize=opts.resize_outputs, return_latents=True, mc_samples=5)

        #latent = net.get_code(inputs, 5)
        ## do latent ops
        latent, sigma = net.get_code_w_sigma(inputs, 10)
        if opts.do_latent_ops:
            # compute sigma of hr
            # if opts.resize_factors == '1' and opts.test_batch_size == 1:
            #     sigma_list = sigma.tolist()[0]
            #     # print(len(sigma_list))
            #     for i in range(18):
            #         # print("sigma to avg ", sigmas_to_avg[i])
            #         # print("sigma_list ",sigma_list[i])
            #         sigmas_to_avg[i].append(sigma_list[i])
            
            if opts.test_batch_size == 1:
                new_latent = latent_ops(latent[0], sigma[0])
            # elif opts.test_batch_size == 2:
            #     new_latent = torch.stack((latent_ops(latent[0], sigma[0]), latent_ops(latent[1], sigma[1])))
            # elif opts.test_batch_size == 4:
            #     new_latent = torch.stack((latent_ops(latent[0], sigma[0]), latent_ops(latent[1], sigma[1]), latent_ops(latent[2], sigma[2]), latent_ops(latent[3], sigma[3])))
            else:
                print("invalid test batch size, should be 1, 2 or 4")
            ##
            result_batch, latent = net(new_latent, randomize_noise=False, resize=opts.resize_outputs, input_code=True, return_latents=True)
        else:
            result_batch, latent = net(latent, randomize_noise=False, resize=opts.resize_outputs, input_code=True, return_latents=True)
            #result_batch, latent = net(inputs, randomize_noise=False, resize=opts.resize_outputs, return_latents=True)
    else:
        latent_mask = [int(l) for l in opts.latent_mask.split(",")]
        result_batch = []
        for image_idx, input_image in enumerate(inputs):
            # get latent vector to inject into our input image
            vec_to_inject = np.random.randn(1, 512).astype('float32')
            _, latent_to_inject = net(torch.from_numpy(vec_to_inject).to("cuda"),
                                      input_code=True,
                                      return_latents=True)
            # get output image with injected style vector
            res, latent = net(input_image.unsqueeze(0).to("cuda").float(),
                      latent_mask=latent_mask,
                      inject_latent=latent_to_inject,
                      alpha=opts.mix_alpha,
                      resize=opts.resize_outputs,
                      return_latents = True)
            result_batch.append(res)
        result_batch = torch.cat(result_batch, dim=0)
    
    return result_batch, latent


if __name__ == '__main__':
    run()
