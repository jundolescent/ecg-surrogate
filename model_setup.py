"""
Model setup and hook registration for surrogate experiments.
"""
import torch
import torch.nn as nn
import re
from transformers import AutoModel
from models.finetune_model import ft_12lead_ECGFounder


class ModelSetup:
    """Handles model loading and hook registration for surrogate experiments."""
    
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.model = None
        self.activations = {}
    
    def load_model(self):
        """Load the appropriate model based on args."""
        if self.args.model == 'hubert-ecg':
            self._load_hubert_ecg()
        elif self.args.model == 'ECGFounder':
            self._load_ecg_founder()
        elif self.args.model == 'ECG-FM':
            self._load_ecg_fm()
        elif self.args.model == 'MERL':
            self._load_merl()
        elif self.args.model == 'HeartLang':
            self.__load_heartlang()
        else:
            raise ValueError(f"Unsupported model: {self.args.model}")
        
        self.model.eval()
        self.model.to(self.device)
    
    def _load_hubert_ecg(self):
        """Load HuBERT-ECG model."""
        self.model = AutoModel.from_pretrained(
            f"Edoardo-BS/hubert-ecg-{self.args.size}", 
            trust_remote_code=True
        )
    
    def _load_ecg_founder(self):
        """Load ECGFounder model."""
        pth = '/home/work/jslee/model/ECGFounder/ckpts/12_lead_ECGFounder.pth'
        self.model = ft_12lead_ECGFounder(
            self.device, pth, n_classes=10, linear_prob=True, return_features=True
        )
    
    def _load_ecg_fm(self):
        """Load ECG-FM model."""
        from fairseq_signals.models import build_model_from_checkpoint
        pth = '/home/work/jslee/model/ECG-FM/ckpts/mimic_iv_ecg_finetuned.pt'
        self.model = build_model_from_checkpoint(checkpoint_path=pth)
    
    def _load_merl(self):
        """Load MERL model."""
        import models.utils_builder
        config = {
            'network': {
                'ecg_model': 'resnet18',
                'num_leads': 12,
                'text_model': 'ncbi/MedCPT-Query-Encoder',
                'free_layers': 12,
                'feature_dim': 768,
                'num_classes': 10,
                'projection_head': {
                    'mlp_hidden_size': 256,
                    'projection_size': 256
                }
            }
        }
        pth = '/home/work/jslee/model/HuBERT-ECG/ckpts/res18_best_ckpt.pth'
        ckpt = torch.load(pth, map_location='cpu', weights_only=False)
        self.model = models.utils_builder.ECGCLIP(config['network'])
        self.model.load_state_dict(ckpt, strict=False)
        
    def __load_heartlang(self):
        from model.HeartLang.modeling_finetune import HeartLang_finetune_base
        from model.HeartLang.utils.utils import load_state_dict, freeze_except_prefix
        from timm.models import create_model
        
        self.model = create_model(
            "HeartLang_finetune_base",
            pretrained=False,
            num_classes=10
        )
        pth = '/home/work/jslee/model/HeartLang/checkpoints/heartlang_base/checkpoint-200.pth'
        checkpoint = torch.load(pth, map_location="cpu", weights_only=False)
        checkpoint_model = None
        for model_key in "model|module".split("|"):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
            
        load_state_dict(self.model, checkpoint_model, prefix="")
        freeze_except_prefix(self.model, "mlp_head")
        
    def setup_hooks(self):
        """Setup forward hooks for feature extraction."""
        if self.args.model == 'ECGFounder' and self.args.layer != 'encoder':
            self._setup_ecg_founder_hooks()
        elif self.args.model == 'hubert-ecg' and self.args.layer != 'encoder':
            self._setup_hubert_ecg_hooks()
        elif self.args.model == 'ECG-FM' and self.args.layer != 'encoder':
            self._setup_ecg_fm_hooks()
        elif self.args.model == 'MERL' and self.args.layer != 'encoder':
            self._setup_merl_hooks()
        elif self.args.model == 'HeartLang':
            self._setup_heartlang_hooks()
    
    def _get_activation(self, name):
        """Create activation hook function."""
        def hook(model, input, output):
            self.activations[name] = output
        return hook
    
    def _setup_ecg_founder_hooks(self):
        """Setup hooks for ECGFounder model."""
        if self.args.layer == 'encoder':
            return
        
        if self.args.layer == 'first_conv':
            target_stage_idx = -1
            target_block_idx = -1
        else:
            match = re.match(r'stage(\d+)_block(\d+)', self.args.layer)
            if not match:
                raise ValueError("Invalid layer format. Use 'conv1' or 'stage{stage_idx}_block{block_idx}'")
            target_stage_idx = int(match.group(1))
            target_block_idx = int(match.group(2))
        
        self.model.first_conv.register_forward_hook(self._get_activation('conv1'))
        self.model.dense = nn.Identity()
        
        if self.args.layer == 'first_conv':
            for i in range(len(self.model.stage_list)):
                for j in range(len(self.model.stage_list[i].block_list)):
                    self.model.stage_list[i].block_list[j] = nn.Identity()
        else:
            truncated = False
            for i in range(len(self.model.stage_list)):
                for j in range(len(self.model.stage_list[i].block_list)):
                    if truncated:
                        self.model.stage_list[i].block_list[j] = nn.Identity()
                    else:
                        self.model.stage_list[i].block_list[j].register_forward_hook(
                            self._get_activation(f'stage{i}_block{j}')
                        )
                        if i == target_stage_idx and j == target_block_idx:
                            truncated = True
    
    def _setup_hubert_ecg_hooks(self):
        """Setup hooks for HuBERT-ECG model."""
        target_type = None
        target_index = -1
        conv_match = re.match(r'conv_layers(\d+)', self.args.layer)
        encoder_match = re.match(r'encoder_layer(\d+)', self.args.layer)
        
        class FlexibleIdentity(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, hidden_states, *args, **kwargs):
                return (hidden_states,)
        if self.args.layer == 'encoder':
            return
        if conv_match:
            target_type = 'conv'
            target_index = int(conv_match.group(1))
        elif encoder_match:
            target_type = 'encoder'
            target_index = int(encoder_match.group(1))
        else:
            raise ValueError("Invalid layer format. Use 'conv_layer{index}' or 'encoder_layer{index}'")
        
        truncated = False
        conv_layers_list = self.model.feature_extractor.conv_layers
        for i in range(len(conv_layers_list)):
            if truncated:
                conv_layers_list[i] = nn.Identity()
            else:
                conv_layers_list[i].register_forward_hook(self._get_activation(f'conv{i}'))
                if target_type == 'conv' and i == target_index:
                    truncated = True
        
        encoder_layers_list = self.model.encoder.layers
        for i in range(len(encoder_layers_list)):
            if truncated:
                encoder_layers_list[i] = FlexibleIdentity()
            else:
                encoder_layers_list[i].register_forward_hook(self._get_activation(f'encoder_layer{i}'))
                if target_type == 'encoder' and i == target_index:
                    truncated = True
    
    def _setup_ecg_fm_hooks(self):
        """Setup hooks for ECG-FM model."""
        target_type = None
        target_index = -1
        conv_match = re.match(r'conv_layers(\d+)', self.args.layer)
        encoder_match = re.match(r'encoder_layer(\d+)', self.args.layer)
        
        class FlexibleIdentity(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, hidden_states, *args, **kwargs):
                return hidden_states, None
            
        if self.args.layer == 'encoder':
            return
        
        if conv_match:
            target_type = 'conv'
            target_index = int(conv_match.group(1))
        elif encoder_match:
            target_type = 'encoder'
            target_index = int(encoder_match.group(1))
        else:
            raise ValueError("Invalid layer format. Use 'conv_layer{index}' or 'encoder_layer{index}'")
        
        truncated = False
        
        conv_layers_list = self.model.encoder.feature_extractor.conv_layers
        for i in range(len(conv_layers_list)):
            if truncated:
                conv_layers_list[i] = nn.Identity()
            else:
                conv_layers_list[i].register_forward_hook(self._get_activation(f'conv{i}'))
                if target_type == 'conv' and i == target_index:
                    truncated = True
        
        encoder_layers_list = self.model.encoder.encoder.layers
        for i in range(len(encoder_layers_list)):
            if truncated:
                encoder_layers_list[i] = FlexibleIdentity()
            else:
                encoder_layers_list[i].register_forward_hook(self._get_activation(f'encoder_layer{i}'))
                if target_type == 'encoder' and i == target_index:
                    truncated = True
    
    def _setup_merl_hooks(self):
        """Setup hooks for MERL model."""
        layer_names = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4']
        target_layer = self.args.layer
        
        if target_layer not in layer_names:
            raise ValueError(f"Invalid layer for MERL. Use one of: {layer_names}")
        
        if target_layer == 'encoder':
                return
            
        setattr(self.model.ecg_encoder, 'linear', nn.Identity())
        truncated = False
        
        for layer_name in layer_names:
            if truncated:
                setattr(self.model.ecg_encoder, layer_name, nn.Identity())
            else:
                try:
                    layer_module = getattr(self.model.ecg_encoder, layer_name)
                except AttributeError:
                    raise AttributeError(f"Model 'MERL' does not have 'ecg_encoder.{layer_name}'")
                
                layer_module.register_forward_hook(self._get_activation(layer_name))
                if target_layer == layer_name:
                    truncated = True
                    
    def _setup_heartlang_hooks(self):
        """Setup hooks for HeartLang model."""
        # self.model.backbone.token_embed.register_forward_hook(self.get_activation('token_embed'))
        for i in range(12):
            self.model.backbone.transformer.layers[i][1].register_forward_hook(
                self._get_activation(f"layer{i}_ff")
            )