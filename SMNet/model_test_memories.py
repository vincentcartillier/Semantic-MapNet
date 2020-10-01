import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch_scatter import scatter_max



class SMNet(nn.Module):
    def __init__(self, cfg, device):
        super(SMNet, self).__init__()

        ego_feat_dim = cfg['ego_feature_dim']
        mem_feat_dim = cfg['mem_feature_dim']
        n_obj_classes = cfg['n_obj_classes']
        mem_update = cfg['mem_update']
        ego_downsample = cfg['ego_downsample']

        self.mem_feat_dim = mem_feat_dim
        self.mem_update = mem_update
        self.ego_downsample = ego_downsample
        self.device = device
        self.device_mem = torch.device('cpu')  # cpu

        if mem_update == 'lstm':
            self.rnn = nn.LSTMCell(ego_feat_dim, mem_feat_dim, bias=True)

            # change default LSTM initialization
            noise = 0.01
            self.rnn.weight_hh.data = -noise + 2 * noise * torch.rand_like(self.rnn.weight_hh)
            self.rnn.weight_ih.data = -noise + 2 * noise * torch.rand_like(self.rnn.weight_ih)
            self.rnn.bias_hh.data = torch.zeros_like(self.rnn.bias_hh)  # redundant with bias_ih
            self.rnn.bias_ih.data = -noise + 2 * noise * torch.rand_like(self.rnn.bias_ih)

        elif mem_update == 'gru':
            self.rnn = nn.GRUCell(ego_feat_dim, mem_feat_dim, bias=True)

            # change default LSTM initialization
            noise = 0.01
            self.rnn.weight_hh.data = -noise + 2 * noise * torch.rand_like(self.rnn.weight_hh)
            self.rnn.weight_ih.data = -noise + 2 * noise * torch.rand_like(self.rnn.weight_ih)
            self.rnn.bias_hh.data = torch.zeros_like(self.rnn.bias_hh)  # redundant with bias_ih
            self.rnn.bias_ih.data = -noise + 2 * noise * torch.rand_like(self.rnn.bias_ih)
        else:
            raise Exception('{} memory update not supported.'.format(mem_update))


        self.decoder = SemmapDecoder(mem_feat_dim, n_obj_classes)


    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)


    def forward(self, features, proj_wtm, mask_outliers, heights, map_height, map_width):

        T,C,H,W = features.shape

        mask_inliers = ~mask_outliers

        memory_size = map_height * map_width * self.mem_feat_dim * 4 / 1e9
        if memory_size > 5:
            self.device_mem = torch.device('cpu')
        else:
            self.device_mem = torch.device('cuda')
        self.decoder = self.decoder.to(self.device_mem)

        if self.mem_update == 'lstm':
            state = (torch.zeros((map_height*map_width,self.mem_feat_dim), dtype=torch.float, device=self.device_mem),
                     torch.zeros((map_height*map_width,self.mem_feat_dim), dtype=torch.float, device=self.device_mem))
        elif self.mem_update == 'gru':
            state = torch.zeros((map_height*map_width,self.mem_feat_dim), dtype=torch.float, device=self.device_mem)

        observed_masks = torch.zeros((map_height*map_width), dtype=torch.bool, device=self.device)
        height_map = torch.zeros((map_height*map_width), dtype=torch.float, device=self.device)

        for t in tqdm(range(T)):

            feature = features[t,:,:,:]
            world_to_map = proj_wtm[t,:,:,:]
            inliers = mask_inliers[t,:,:]
            height = heights[t,:,:]

            world_to_map = world_to_map.long()

            feature = feature.to(self.device)
            world_to_map = world_to_map.to(self.device)
            inliers = inliers.to(self.device)
            height = height.to(self.device)


            if self.ego_downsample:
                world_to_map = world_to_map[::4, ::4, :]
                inliers = inliers[::4,::4]
                height = height[::4,::4]

            flat_indices = (map_width*world_to_map[:,:,1] + world_to_map[:,:,0]).long()
            flat_indices = flat_indices[inliers]
            height = height[inliers]
            height += 1000
            height_map, highest_height_indices = scatter_max(
                height,
                flat_indices,
                dim=0,
                out = height_map,
            )

            m = highest_height_indices >= 0

            observed_masks += m

            if m.any():
                feature = F.interpolate(feature.unsqueeze(0), size=(480,640), mode="bilinear", align_corners=True)
                feature = feature.squeeze(0)
                if self.ego_downsample:
                    feature = feature[:,::4,::4]

                feature = feature.permute(1,2,0)  # -- (N,H,W,512)

                feature = feature[inliers, :]

                tmp_memory = feature[highest_height_indices[m], :]

                if self.mem_update == 'lstm':
                    tmp_state = (state[0][m,:].to(self.device),
                                 state[1][m,:].to(self.device))

                    tmp_state = self.rnn(tmp_memory, tmp_state)

                    state[0][m,:] = tmp_state[0].to(self.device_mem)
                    state[1][m,:] = tmp_state[1].to(self.device_mem)

                elif self.mem_update == 'gru':
                    tmp_state = state[m,:].to(self.device)

                    tmp_state = self.rnn(tmp_memory, tmp_state)

                    state[m,:] = tmp_state.to(self.device_mem)
                else:
                    raise NotImplementedError


                del tmp_memory
            del feature

        if self.mem_update == 'lstm':
            memory = state[0]
        elif self.mem_update == 'gru':
            memory = state
        
        memory = memory.view(map_height, map_width, self.mem_feat_dim)
        
        return memory




class SemmapDecoder(nn.Module):
    def __init__(self, feat_dim, n_obj_classes):

        super(SemmapDecoder, self).__init__()

        self.layer = nn.Sequential(nn.Conv2d(feat_dim, 128, kernel_size=7, stride=1, padding=3, bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(64, 48, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(48),
                                   nn.ReLU(inplace=True),
                                  )

        self.obj_layer = nn.Sequential(nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(48),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(48, n_obj_classes,
                                                 kernel_size=1, stride=1,
                                                 padding=0, bias=True),
                                      )

    def forward(self, memory):
        l1 = self.layer(memory)
        out_obj = self.obj_layer(l1)
        return out_obj






