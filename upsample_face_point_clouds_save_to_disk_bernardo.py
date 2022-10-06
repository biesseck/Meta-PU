import os
import time
import math
import socket
import argparse
import importlib
import warnings
import numpy as np
from pathlib import Path

from glob import glob
from tqdm import tqdm

import model.data_loader as data_loader
import model.data_utils as d_utils
import model.networks as MODEL_GEN
from geomloss import SamplesLoss

warnings.filterwarnings("ignore")

import torch
import sys
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from warmup_scheduler import GradualWarmupScheduler
from pyntcloud import PyntCloud

parser = argparse.ArgumentParser()
parser.add_argument('--phase', default='test', help='train or test [default: test]')
parser.add_argument('--gpu', default='0', help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='model/networks', help='Model name [default: networks]')
parser.add_argument('--log_dir', default='models/logs', help='Log dir')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [1024] [default: 1024]')
parser.add_argument('--up_ratio', type=int, default=4, help='Upsampling Ratio [default: 4]')
parser.add_argument('--max_epoch', type=int, default=80, help='Epoch to run [default: 80]')
parser.add_argument('--batch_size', type=int, default=24, help='Batch Size during training [default: 24]')
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--dataset', default=None)
parser.add_argument('--gan', default=False, action='store_true')
parser.add_argument('--model_path', type=int, default=0, help='The num of epoch to restore the models from')
parser.add_argument('--lambd', default=10000, type=float)
parser.add_argument('--max_sinkhorn_iters', default=32, help="Maximum number of Sinkhorn iterations")
parser.add_argument('--FWWD', default=False, action='store_true', help="move WD loss cal in g forward (for memory balance)")
parser.add_argument('--replace', default=False, action='store_true')
parser.add_argument('--nowarmup', default=False, action='store_true')
parser.add_argument('--test_scale', type=float, default=4, help='up ratio during testing [default: 4]')
parser.add_argument('--num_workers_each_gpu', type=int, default=4, help='[default: 4]')

# BERNARDO
parser.add_argument("-dataset_path", type=str, default='',
                    help="Path of dataset root folder containing 3D face reconstructions (OBJ or PLY format)"
                    )



# BERNARDO
sys.argv += ['--phase', 'test']
sys.argv += ['--log_dir', 'model/new']
sys.argv += ['--batch_size', '1']
sys.argv += ['--model', 'model_res_mesh_pool']
sys.argv += ['--model_path', '60']
sys.argv += ['--gpu', '0']
sys.argv += ['--test_scale', '4']

sys.argv += ['-dataset_path', '/home/bjgbiesseck/GitHub/MICA/demo/output/lfw']
# sys.argv += ['-dataset_path', '/home/bjgbiesseck/GitHub/MICA/demo/output/TALFW']




USE_DATA_NORM = True
USE_RANDOM_INPUT = True
ASSIGN_MODEL_PATH = 0

FLAGS = parser.parse_args()
PHASE = FLAGS.phase
GPU_INDEX = FLAGS.gpu
MODEL_DIR = FLAGS.log_dir
RESTORE_MODEL_DIR = FLAGS.log_dir
NUM_POINT = FLAGS.num_point
UP_RATIO = FLAGS.up_ratio
MAX_EPOCH = FLAGS.max_epoch
BATCH_SIZE = FLAGS.batch_size
BASE_LEARNING_RATE = FLAGS.learning_rate
ASSIGN_MODEL_PATH = FLAGS.model_path
max_sinkhorn_iters = FLAGS.max_sinkhorn_iters
Replace = FLAGS.replace

print(socket.gethostname())
print(FLAGS)
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_INDEX

    
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.optim as optim

device = torch.device('cuda:{}'.format(int(GPU_INDEX)) if torch.cuda.is_available() else 'cpu')
print('device: {}'.format(device))
if ASSIGN_MODEL_PATH > 0:
    ori_dir = MODEL_DIR
    MODEL_DIR = os.path.join(MODEL_DIR, 'models_{}'.format(ASSIGN_MODEL_PATH))
  

def log_string(LOG_FOUT, out_str):
    LOG_FOUT.write(out_str)
    LOG_FOUT.flush()

def weight_init(m):          
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
    

def weight_clip(m):
    if hasattr(m, 'weight'):
        m.weight.data.clamp_(-0.01, 0.01)

def load_checkpoint(model, optimizer, fc_optimizer=None, name='g'):
    _file = os.path.join(RESTORE_MODEL_DIR, '{}_model_{}.pth'.format(name, ASSIGN_MODEL_PATH))
    print("=> loading checkpoint '{}'".format(_file))
    if os.path.isfile(_file):
        try:
            checkpoint = torch.load(_file)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            print('model loaded...')
            optimizer.load_state_dict(checkpoint['optimizer'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
            if name == 'g':
                fc_optimizer.load_state_dict(checkpoint['fc_optimizer'])
                for state in fc_optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda()
                print('fc optimizer loaded...')
            print('optimizer loaded...')
            
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(_file, checkpoint['epoch']))
        except:
            try:
                checkpoint = torch.load(_file)
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    
                    name = k.replace("module.", "")  
                    
                    new_state_dict[name] = v
                model.load_state_dict(new_state_dict)
                print('model loaded...')
                optimizer.load_state_dict(checkpoint['optimizer'])
                if name == 'g':
                    fc_optimizer.load_state_dict(checkpoint['fc_optimizer'])
                    for state in fc_optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.cuda()
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda()
                print('optimizer loaded...')
                start_epoch = checkpoint['epoch']
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(_file, checkpoint['epoch']))
            except:
                print('load model error')
    else:
        print("=> no checkpoint found at '{}'".format(_file))

    return model, optimizer, fc_optimizer

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id + int(time.time()))

def pause():
    input('PAUSED...')

def load_model(this_scale=4):
    torch.backends.cudnn.benchmark = True
    import model.data_loader as data_loader
    import model.data_utils as d_utils
    BATCH_SIZE = 1
    multi_gpus = False
    if ',' in GPU_INDEX:
        print('dont use multi gpu!!!!!!!')
        gpu_ids = [int(id) for id in GPU_INDEX.split(',')]
        multi_gpus = True
    else:
        gpu_ids = [int(GPU_INDEX)]

    device = torch.device('cuda:{}'.format(gpu_ids[0]) if torch.cuda.is_available() else 'cpu')
    print('device: {}'.format(device))

    # data_folder = FLAGS.dataset
    # phase = data_folder.split('/')[-2] + data_folder.split('/')[-1]
    # save_path = os.path.join(ori_dir, 'result/' + phase)
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    #
    # # samples = glob(data_folder + "/*.xyz")   # original
    # samples = glob(data_folder + "/*.obj")     # BERNARDO
    # samples.sort(reverse=True)
    # print('in',data_folder,'num of samples: ',len(samples))

    g_model = MODEL_GEN.GenModel(use_normal=False, use_bn=False, use_ibn=False, bn_decay=0.95, up_ratio=this_scale,
                                 device=device, training=False)
    g_model.eval()

    if multi_gpus:
        g_model = torch.nn.DataParallel(g_model, device_ids=gpu_ids).to(device)
    else:
        g_model = g_model.to(device)

    print('loading models...')
    try:
        print(os.path.join(ori_dir, 'g_model_{}.pth'.format(ASSIGN_MODEL_PATH)))
        dic = torch.load(os.path.join(ori_dir, 'g_model_{}.pth'.format(ASSIGN_MODEL_PATH)))
        g_model.load_state_dict(dic['state_dict'])
    except:
        try:
            weight = torch.load(os.path.join(ori_dir, 'g_model_{}.pth'.format(ASSIGN_MODEL_PATH)),
                                map_location=lambda storage, loc: storage)

            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in weight['state_dict'].items():
                name = k.replace("module.", "")

                new_state_dict[name] = v
            g_model.load_state_dict(new_state_dict)
        except:
            print('load model error')
    return g_model


def prediction_whole_model(g_model, this_scale=4, data_folder='', use_normal=False):
    # samples = glob(data_folder + "/*.xyz")   # original
    samples = glob(data_folder + "/*.obj")  # BERNARDO
    samples.sort(reverse=True)
    print('in', data_folder, 'num of samples: ', len(samples))

    save_path = data_folder

    total_time = 0
    for i, item in enumerate(samples):
        if item.endswith('.obj'):
            point_cloud = PyntCloud.from_file(item)
            input = point_cloud.points.to_numpy()
            # print('input:', input)
            # print('my_point_cloud:', my_point_cloud)
        else:
            input = np.loadtxt(item)
            # print('input:', input)

        input = np.expand_dims(input, axis=0)
        if not use_normal:
            input = input[:, :, 0:3]
        print(item, input.shape)
        with torch.no_grad():
            input_torch = torch.from_numpy(input).type(torch.cuda.FloatTensor).detach().to(device)
            beg = time.time()
            pred, _, _, _, _ = g_model(input_torch, this_scale=this_scale)
            end = time.time()
            pred = pred.detach().cpu()
            path = os.path.join(save_path, item.split('/')[-1])
            path = path[:-4] + '_upsample_MetaPU.xyz'
            # if use_normal:
            #     norm_pl = np.zeros_like(pred)
            #     data_loader.save_pl(path, np.hstack((pred[0, ...], norm_pl[0, ...])))
            # else:
            #     data_loader.save_pl(path, pred[0, ...])
            print('saving:', path, '...', end=' ')
            data_loader.save_pl(path, pred[0, ...])   # save upsampling point cloud
            print('saved!')

            # # Save a copy of input
            # path = path[:-4] + '_usample_MetaPU.xyz'
            # data_loader.save_pl(path, input[0])
        total_time += (end - beg)
    print('total time is: {}'.format(total_time))


# BERNARDO
class Tree:
    def walk(self, dir_path: Path):
        contents = list(dir_path.iterdir())
        for path in contents:
            if path.is_dir():  # extend the prefix and recurse:
                yield str(path)
                yield from self.walk(path)

    def get_all_sub_folders(self, dir_path: str):
        folders = [dir_path]
        for folder in Tree().walk(Path(os.getcwd()) / dir_path):
            # print(folder)
            folders.append(folder)
        folders.sort()
        return folders


def main(FLAGS):
    # load dataset (LFW and TALFW)
    print('upsample_face_point_clouds_save_to_disk_bernardo.py: main(): Loading sub-folders of dataset', FLAGS.dataset_path, '...')
    sub_folders = Tree().get_all_sub_folders(FLAGS.dataset_path)
    # print('sub_folders:', sub_folders)
    # print('len(sub_folders):', len(sub_folders))

    g_model = load_model(this_scale=FLAGS.test_scale)

    # compute and save point cloud normals to disk
    # for i in range(10):  # range(len(sub_folders))
    for i in range(len(sub_folders)):
        sub_folder = sub_folders[i]
        # print('sub_folder:', sub_folder)
        # load_pc_and_compute_normals(model, sub_folder)
        print('upsample_face_point_clouds_save_to_disk_bernardo.py: main(): upsampling point cloud ' + str(i) + '/' + str(len(sub_folders)))
        prediction_whole_model(g_model, this_scale=FLAGS.test_scale, data_folder=sub_folder)



if __name__ == "__main__":

    # if Replace == True:
    #     try:
    #         import shutil
    #         shutil.rmtree(os.path.join(MODEL_DIR, 'code/'))
    #     except:
    #         pass

    # args = parse_args()
    # print('__main__(): args=', args)
    args = FLAGS

    main(FLAGS)
