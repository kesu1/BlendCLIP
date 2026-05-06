'''
 * Copyright (c) 2023, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Le Xue
'''

# Modified from github.com/openai/CLIP
from collections import OrderedDict

import timm
from torch import nn
# from models.pointnet2.pointnet2 import Pointnet2_Ssg
from data.dataset_3d import  *

from models import losses
from torch.nn.parameter import Parameter
from easydict import EasyDict
import open_clip

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class ULIP_WITH_IMAGE(nn.Module):
    def __init__(self, point_encoder, **kwargs):
        # super().__init__(ssl_mlp_dim, ssl_emb_dim, **kwargs)
        super().__init__()
        kwargs = EasyDict(kwargs)
        self.context_length = kwargs.context_length
        self.vision_width = kwargs.vision_width
        self.visual = kwargs.vision_model

        self.transformer = Transformer(
            width=kwargs.transformer_width,
            layers=kwargs.transformer_layers,
            heads=kwargs.transformer_heads,
            attn_mask=self.build_attention_mask(),
        )

        self.vocab_size = kwargs.vocab_size
        self.token_embedding = nn.Embedding(kwargs.vocab_size, kwargs.transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, kwargs.transformer_width))
        self.ln_final = LayerNorm(kwargs.transformer_width)

        self.image_projection = nn.Parameter(torch.empty(kwargs.vision_width, kwargs.embed_dim))
        self.text_projection = nn.Parameter(torch.empty(kwargs.transformer_width, kwargs.embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

        self.point_encoder = point_encoder

        self.pc_projection = nn.Parameter(torch.empty(kwargs.pc_feat_dims, 512))
        nn.init.normal_(self.pc_projection, std=512 ** -0.5)

    def encode_image(self, image):
        with torch.no_grad():
            x = self.visual(image)
            x = x @ self.image_projection

        return x.detach()

    def encode_text(self, text):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        nn.init.normal_(self.image_projection, std=self.vision_width ** -0.5)
        nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def encode_pc(self, pc):
        pc_feat = self.point_encoder(pc)
        pc_embed = pc_feat @ self.pc_projection
        return pc_embed

    def forward(self, pc, text, image=None):

        text_embed_all = []
        for i in range(text.shape[0]):
            text_for_one_sample = text[i]
            text_embed = self.encode_text(text_for_one_sample)
            text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
            text_embed = text_embed.mean(dim=0)
            text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
            text_embed_all.append(text_embed)

        text_embed_all = torch.stack(text_embed_all)
        pc_embed = self.encode_pc(pc)
        if image is not None:
            image_embed = self.encode_image(image)
            return {'text_embed': text_embed_all,
                    'pc_embed': pc_embed,
                    'image_embed': image_embed,
                    'logit_scale': self.logit_scale.exp()}

        else:
            return {'text_embed': text_embed_all,
                    'pc_embed': pc_embed,
                    'logit_scale': self.logit_scale.exp()}
            
            
class ULIP2_WITH_OPENCLIP(nn.Module):
    def __init__(self, point_encoder, mlp_head=False, **kwargs):
        # super().__init__(ssl_mlp_dim, ssl_emb_dim, **kwargs)
        super().__init__()
        kwargs = EasyDict(kwargs)

        self.open_clip_model = kwargs.open_clip_model

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.point_encoder = point_encoder
        
        self.mlp_head = mlp_head
        
        #self.tokenizer = open_clip.get_tokenizer('ViT-bigG-14')
        #self.tokenizer = open_clip.get_tokenizer("hf-hub:laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K")

        #self.pc_projection = nn.Parameter(torch.empty(kwargs.pc_feat_dims, 1280))
        #nn.init.normal_(self.pc_projection, std=1280 ** -0.5)
        
        #self.pc_projection = nn.Parameter(torch.empty(kwargs.pc_feat_dims, 512))
        #nn.init.normal_(self.pc_projection, std=512 ** -0.5)
        
        hidden_dim = 2048
        output_dim = 512 # Target embedding dimension
        
        if mlp_head:
            self.pc_projection = nn.Sequential(
                nn.Linear(kwargs.pc_feat_dims, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
            
            # Initialize MLP weights
            for m in self.pc_projection.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        else:
            self.pc_projection = nn.Parameter(torch.empty(kwargs.pc_feat_dims, output_dim))
            nn.init.normal_(self.pc_projection, std=output_dim ** -0.5)
        
    @torch.inference_mode()
    def encode_image(self, image):
        x = self.open_clip_model.encode_image(image)
        return x

    @torch.inference_mode()
    def encode_text(self, text):
        x = self.open_clip_model.encode_text(text)
        return x

    def encode_pc(self, pc):
        pc_feat = self.point_encoder(pc)
        pc_embed = pc_feat @ self.pc_projection if not self.mlp_head else self.pc_projection(pc_feat)
        return pc_embed

    def forward(self, pc, text, image=None):
        if not text.ndim == 3:
            raise ValueError(f"Expected text tensor dimension to be 3, but got {text.ndim}")
        
        B, N, L = text.shape
        text_reshaped = text.reshape(B * N, L) # Reshape to [B*N, L]
            
        text_embed_flat = self.encode_text(text_reshaped) # Output shape [B*N, EmbedDim]
        text_embed_flat = text_embed_flat / text_embed_flat.norm(dim=-1, keepdim=True)
            
        # Reshape back and average over captions
        text_embed_all = text_embed_flat.reshape(B, N, -1) # Shape [B, N, EmbedDim]
        text_embed_all = text_embed_all.mean(dim=1) # Average over N captions -> Shape [B, EmbedDim]
        text_embed_all = text_embed_all / text_embed_all.norm(dim=-1, keepdim=True) # Normalize again after averaging

        """
        text_embed_all = []
        for i in range(text.shape[0]):
            text_for_one_sample = text[i]
            text_embed = self.encode_text(text_for_one_sample)
            text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
            text_embed = text_embed.mean(dim=0)
            text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
            text_embed_all.append(text_embed)

        text_embed_all = torch.stack(text_embed_all)
        """
        
        pc_embed = self.encode_pc(pc)
        if image is not None:
            image_embed = self.encode_image(image)
            return {'text_embed': text_embed_all,
                    'pc_embed': pc_embed,
                    'image_embed': image_embed,
                    'logit_scale': self.logit_scale.exp()}

        else:
            return {'text_embed': text_embed_all,
                    'pc_embed': pc_embed,
                    'logit_scale': self.logit_scale.exp()}


def get_loss(args):
    return losses.ULIPWithImageLoss()


def get_metric_names(model):
    return ['loss', 'ulip_loss', 'ulip_pc_image_acc', 'ulip_pc_text_acc']


def ULIP_PointBERT(args):
    vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)

    # =====================================================================
    # import the 3D backbone and specify the output point cloud feature dimension
    from models.pointbert.point_encoder import PointTransformer
    config_addr = './models/pointbert/PointTransformer_8192point.yaml'
    config = cfg_from_yaml_file(config_addr)
    point_encoder = PointTransformer(config.model, args=args)
    pc_feat_dims = 768
    # =====================================================================

    model = ULIP_WITH_IMAGE(embed_dim=512, vision_width=768, point_encoder=point_encoder, vision_model=vision_model,
                            context_length=77, vocab_size=49408,
                            transformer_width=512, transformer_heads=8, transformer_layers=12, pc_feat_dims=pc_feat_dims)

    if not args.evaluate_3d:
        # load the pretrained model
        pretrain_slip_model = torch.load('./data/initialize_models/slip_base_100ep.pt', map_location=torch.device('cpu'))
        pretrain_slip_model_params = pretrain_slip_model['state_dict']
        pretrain_slip_model_params = {param_name.replace('module.', ''): param for param_name, param in
                                      pretrain_slip_model_params.items()}

        for name, param in model.named_parameters():
            if name not in pretrain_slip_model_params:
                continue

            if isinstance(pretrain_slip_model_params[name], Parameter):
                param_new = pretrain_slip_model_params[name].data
            else:
                param_new = pretrain_slip_model_params[name]

            param.requires_grad = False
            print('load {} and freeze'.format(name))
            param.data.copy_(param_new)

    return model

def ULIP2_PointBERT_Colored(args):
    print("Get openclip model:")
    open_clip_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('ViT-bigG-14',
                                                                          pretrained='laion2b_s39b_b160k')
    
    #open_clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16',
    #                                                                      pretrained='openai')
    
    tokenizer = open_clip.get_tokenizer('ViT-bigG-14')
    
    open_clip_model.eval()
    print("Finished loading the openclip model.")

    # =====================================================================
    # import the 3D backbone and specify the output point cloud feature dimension
    from models.pointbert.point_encoder import PointTransformer, PointTransformer_Colored
    config_addr = './models/pointbert/ULIP_2_PointBERT_10k_colored_pointclouds.yaml'
    config = cfg_from_yaml_file(config_addr)
    point_encoder = PointTransformer_Colored(config.model, args=args)
    pc_feat_dims = 768
    # =====================================================================

    model = ULIP2_WITH_OPENCLIP(open_clip_model=open_clip_model, point_encoder=point_encoder, pc_feat_dims=pc_feat_dims)

    return model, tokenizer, preprocess_train, preprocess_val

def ULIP2_PointBERT(args):
    print("Get openclip model:")
    #open_clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-bigG-14',
    #                                                                      pretrained='laion2b_s39b_b160k')
    
    open_clip_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('ViT-B-16',
                                                                          pretrained='datacomp_xl_s13b_b90k')
    
    open_clip_model.eval()
    
    # freeze the openclip model
    for param in open_clip_model.parameters():
        param.requires_grad = False
    
    tokenizer = open_clip.get_tokenizer("hf-hub:laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K")
    
    print("Finished loading the openclip model.")

    # =====================================================================
    # import the 3D backbone and specify the output point cloud feature dimension
    from models.pointbert.point_encoder import PointTransformer
    config_addr = './models/pointbert/PointTransformer_8192point.yaml'
    config = cfg_from_yaml_file(config_addr)
    point_encoder = PointTransformer(config.model, args=args)
    pc_feat_dims = 768
    # =====================================================================

    model = ULIP2_WITH_OPENCLIP(open_clip_model=open_clip_model,
                                point_encoder=point_encoder,
                                pc_feat_dims=pc_feat_dims)

    return model, tokenizer, preprocess_train, preprocess_val