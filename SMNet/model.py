import torch
import torch.nn as nn
import torch.nn.functional as F



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
        self.device_mem = device  # cpu
        #self.device_mem = torch.device('cuda')  # cpu

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
        elif mem_update == 'replace':
            self.linlayer = nn.Linear(ego_feat_dim, mem_feat_dim)
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
    

    def encode(self, features, proj_indices, masks_inliers):

        features = features.float()

        N,T,C,H,W = features.shape

        if self.mem_update == 'lstm':
            state = (torch.zeros((N*250*250,self.mem_feat_dim), dtype=torch.float, device=self.device_mem),
                     torch.zeros((N*250*250,self.mem_feat_dim), dtype=torch.float, device=self.device_mem))
        elif self.mem_update == 'gru':
            state = torch.zeros((N*250*250,self.mem_feat_dim), dtype=torch.float, device=self.device_mem)
        elif self.mem_update == 'replace':
            state = torch.zeros((N*250*250,self.mem_feat_dim), dtype=torch.float, device=self.device_mem)

        observed_masks = torch.zeros((N,250,250), dtype=torch.bool, device=self.device)

        for t in range(T):

            feature = features[:,t,:,:,:]
            mask_inliers = masks_inliers[:,t,:,:]
            proj_index = proj_indices[:,t,:]

            if self.ego_downsample:
                mask_inliers = mask_inliers[:,::4,::4]

            m = (proj_index>=0)    # -- (N, 250*250)

            if N > 1:
                batch_offset = torch.zeros(N, device=self.device)
                batch_offset[1:] = torch.cumsum(mask_inliers.sum(dim=1).sum(dim=1), dim=0)[:-1]
                batch_offset = batch_offset.unsqueeze(1).repeat(1, 250*250).long()

                proj_index += batch_offset

            if m.any():
                feature = F.interpolate(feature, size=(480,640), mode="bilinear", align_corners=True)
                if self.ego_downsample:
                    feature = feature[:,:,::4,::4]

                feature = feature.permute(0,2,3,1)  # -- (N,H,W,512)

                feature = feature[mask_inliers, :]

                tmp_memory = feature[proj_index[m], :]

                tmp_top_down_mask = m.view(-1)

                if self.mem_update == 'lstm':
                    tmp_state = (state[0][tmp_top_down_mask,:].to(self.device),
                                 state[1][tmp_top_down_mask,:].to(self.device))

                    tmp_state = self.rnn(tmp_memory, tmp_state)

                    state[0][tmp_top_down_mask,:] = tmp_state[0].to(self.device_mem)
                    state[1][tmp_top_down_mask,:] = tmp_state[1].to(self.device_mem)

                elif self.mem_update == 'gru':
                    tmp_state = state[tmp_top_down_mask,:].to(self.device)

                    tmp_state = self.rnn(tmp_memory, tmp_state)

                    state[tmp_top_down_mask,:] = tmp_state.to(self.device_mem)

                elif self.mem_update == 'replace':
                    tmp_memory = self.linlayer(tmp_memory)
                    state[tmp_top_down_mask,:] = tmp_memory.to(self.device_mem)

                else:
                    raise NotImplementedError

                observed_masks += m.reshape(N,250,250)

                del tmp_memory
            del feature

        if self.mem_update == 'lstm':
            memory = state[0]
        elif self.mem_update == 'gru':
            memory = state
        elif self.mem_update == 'replace':
            memory = state

        memory = memory.view(N,250,250,self.mem_feat_dim)

        memory = memory.permute(0,3,1,2)
        memory = memory.to(self.device)

        return memory, observed_masks


    def forward(self, features, proj_indices, masks_inliers):

        memory, observed_masks = self.encode(features, 
                                             proj_indices, 
                                             masks_inliers)

        semmap = self.decoder(memory)

        return semmap, observed_masks






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






