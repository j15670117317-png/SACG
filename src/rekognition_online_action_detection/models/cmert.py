import torch
import torch.nn as nn
import torch.nn.functional as F
from . import transformer as tr

from .models import META_ARCHITECTURES as registry, META_ARCHITECTURES
from .normalized_linear import NormalizedLinear
from .feature_head import BaseFeatureHead_mat
from ..utils.ek_utils import (action_to_noun_map, action_to_verb_map)



def _cfg_get(node, name, default):
    return getattr(node, name) if hasattr(node, name) else default


def _is_cfg_override(value):
    """Return True when a config value should override its fallback.

    Empty strings/lists are used as sentinels for optional stage-level configs.
    """
    return value is not None and value != '' and value != []


def _cfg_get_override(node, name, default):
    value = _cfg_get(node, name, None)
    return value if _is_cfg_override(value) else default


def _resolve_attention_type(lstr_cfg, module_prefix, attention_kind, default, stage_idx=None):
    """Resolve attention type for one site, optionally at a long-compress stage.

    Precedence for stage_idx=None:
      1. <MODULE>_<SELF|CROSS>_ATTN_TYPE
      2. default

    Precedence for stage_idx=i:
      1. <MODULE>_STAGE{i}_<SELF|CROSS>_ATTN_TYPE
      2. <MODULE>_STAGE{i}_ATTN_TYPE
      3. <MODULE>_<SELF|CROSS>_ATTN_TYPE
      4. default
    """
    base = _cfg_get_override(lstr_cfg, f'{module_prefix}_{attention_kind}_ATTN_TYPE', default)
    if stage_idx is None:
        return base
    stage_base = _cfg_get_override(lstr_cfg, f'{module_prefix}_STAGE{stage_idx}_ATTN_TYPE', base)
    return _cfg_get_override(lstr_cfg, f'{module_prefix}_STAGE{stage_idx}_{attention_kind}_ATTN_TYPE', stage_base)


def _resolve_assa_init_w(lstr_cfg, module_prefix, attention_kind, global_default, stage_idx=None):
    """Resolve ASSA [dense, sparse] initialization for one attention site.

    Precedence for stage_idx=None:
      1. <MODULE>_<SELF|CROSS>_ASSA_INIT_W
      2. <MODULE>_ASSA_INIT_W
      3. ASSA_INIT_W

    Precedence for stage_idx=i:
      1. <MODULE>_STAGE{i}_<SELF|CROSS>_ASSA_INIT_W
      2. <MODULE>_STAGE{i}_ASSA_INIT_W
      3. <MODULE>_<SELF|CROSS>_ASSA_INIT_W
      4. <MODULE>_ASSA_INIT_W
      5. ASSA_INIT_W
    """
    module_default = _cfg_get_override(lstr_cfg, f'{module_prefix}_ASSA_INIT_W', global_default)
    module_attention_default = _cfg_get_override(
        lstr_cfg, f'{module_prefix}_{attention_kind}_ASSA_INIT_W', module_default)
    if stage_idx is None:
        return module_attention_default
    stage_default = _cfg_get_override(
        lstr_cfg, f'{module_prefix}_STAGE{stage_idx}_ASSA_INIT_W', module_attention_default)
    return _cfg_get_override(
        lstr_cfg, f'{module_prefix}_STAGE{stage_idx}_{attention_kind}_ASSA_INIT_W', stage_default)


def _use_mq_moe_cross(enabled, cross_attention_type):
    """Enable MQ-MoE only on cross-attention sites that are not ASSA."""
    return bool(enabled) and str(cross_attention_type).lower() != 'assa'


def _read_mq_moe_cfg(lstr_cfg, prefix):
    """Read one module's MQ-MoE config group."""
    return {
        'enabled': _cfg_get(lstr_cfg, f'{prefix}_MQ_MOE', False),
        'experts': _cfg_get(lstr_cfg, f'{prefix}_MQ_MOE_EXPERTS', ['global', 'recent', 'boundary', 'action']),
        'recent_k': _cfg_get(lstr_cfg, f'{prefix}_MQ_MOE_RECENT_K', 32),
        'gate_temperature': _cfg_get(lstr_cfg, f'{prefix}_MQ_MOE_GATE_TEMPERATURE', 1.0),
        'boundary_abs': _cfg_get(lstr_cfg, f'{prefix}_MQ_MOE_BOUNDARY_ABS', False),
    }


@registry.register('CMeRT')
class CMeRT(nn.Module):
    # contextual enhancement
    def __init__(self, cfg):
        super(CMeRT, self).__init__()

        self.cfg = cfg
        # Build long feature heads
        self.long_memory_num_samples = cfg.MODEL.LSTR.LONG_MEMORY_NUM_SAMPLES
        self.long_enabled = self.long_memory_num_samples > 0
        if self.long_enabled:
            # BaseFeatureHead_mat 是一个神经网络模块，用于处理长时特征：将原始特征转换为适合Transformer处理的格式
            self.feature_head_long = BaseFeatureHead_mat(cfg)

        # Build work feature head
        self.work_memory_num_samples = cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES
        self.work_enabled = self.work_memory_num_samples > 0
        if self.work_enabled:
            self.feature_head_work = BaseFeatureHead_mat(cfg)

        self.d_model = self.feature_head_work.d_model
        self.num_heads = cfg.MODEL.LSTR.NUM_HEADS
        self.dim_feedforward = cfg.MODEL.LSTR.DIM_FEEDFORWARD
        self.dropout = cfg.MODEL.LSTR.DROPOUT
        self.activation = cfg.MODEL.LSTR.ACTIVATION
        self.num_classes = cfg.DATA.NUM_CLASSES

        # Build position encoding
        self.pos_encoding = tr.PositionalEncoding(self.d_model, self.dropout)
        self.anticipation_length = cfg.MODEL.LSTR.ANTICIPATION_LENGTH
        self.anticipation_sample_rate = cfg.MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE
        self.anticipation_num_samples = self.anticipation_length // self.anticipation_sample_rate
        self.future_num_samples = cfg.MODEL.LSTR.FUTURE_NUM_SAMPLES
        self.future_enabled = self.future_num_samples > 0
        self.context_num_samples = int(cfg.MODEL.LSTR.CONTEXT_SECONDS * cfg.DATA.FPS) // cfg.MODEL.LSTR.CONTEXT_MEMORY_SAMPLE_RATE

        self.long_memory_use_pe = cfg.MODEL.LSTR.LONG_MEMORY_USE_PE
        self.work_memory_use_pe = cfg.MODEL.LSTR.WORK_MEMORY_USE_PE
        self.include_work = cfg.MODEL.LSTR.LONG_MEMORY_INCLUDE_WORK

        # Attention switches for the ASSA ablation. Defaults keep the original CMeRT behavior.
        legacy_long_cross = _cfg_get(cfg.MODEL.LSTR, 'LONG_COMPRESS_ATTN_TYPE',
                                     _cfg_get(cfg.MODEL.LSTR, 'LONG_COMPRESS_M0_ATTENTION_TYPE', 'dotproduct'))
        self.assa_tau = _cfg_get(cfg.MODEL.LSTR, 'ASSA_TAU', 1.0)
        self.assa_init_w = _cfg_get(cfg.MODEL.LSTR, 'ASSA_INIT_W', [1.0, 1.0])
        self.long_compress_self_attention_type = _cfg_get(cfg.MODEL.LSTR, 'LONG_COMPRESS_SELF_ATTN_TYPE', 'dotproduct')
        self.long_compress_cross_attention_type = _cfg_get(cfg.MODEL.LSTR, 'LONG_COMPRESS_CROSS_ATTN_TYPE', legacy_long_cross)
        self.dec_self_attention_type = _cfg_get(cfg.MODEL.LSTR, 'DEC_SELF_ATTN_TYPE', 'dotproduct')
        self.dec_cross_attention_type = _cfg_get(cfg.MODEL.LSTR, 'DEC_CROSS_ATTN_TYPE', 'dotproduct')
        self.gen_self_attention_type = _cfg_get(cfg.MODEL.LSTR, 'GEN_SELF_ATTN_TYPE', 'dotproduct')
        self.gen_cross_attention_type = _cfg_get(cfg.MODEL.LSTR, 'GEN_CROSS_ATTN_TYPE', 'dotproduct')
        self.refine_self_attention_type = _cfg_get(cfg.MODEL.LSTR, 'REFINE_SELF_ATTN_TYPE', 'dotproduct')
        self.refine_cross_attention_type = _cfg_get(cfg.MODEL.LSTR, 'REFINE_CROSS_ATTN_TYPE', 'dotproduct')

        # ASSA branch-logit initialization can now differ by CMeRT location.
        # Values are [dense_logit, sparse_logit].
        self.long_compress_self_assa_init_w = _resolve_assa_init_w(cfg.MODEL.LSTR, 'LONG_COMPRESS', 'SELF', self.assa_init_w)
        self.long_compress_cross_assa_init_w = _resolve_assa_init_w(cfg.MODEL.LSTR, 'LONG_COMPRESS', 'CROSS', self.assa_init_w)
        self.dec_self_assa_init_w = _resolve_assa_init_w(cfg.MODEL.LSTR, 'DEC', 'SELF', self.assa_init_w)
        self.dec_cross_assa_init_w = _resolve_assa_init_w(cfg.MODEL.LSTR, 'DEC', 'CROSS', self.assa_init_w)
        self.gen_self_assa_init_w = _resolve_assa_init_w(cfg.MODEL.LSTR, 'GEN', 'SELF', self.assa_init_w)
        self.gen_cross_assa_init_w = _resolve_assa_init_w(cfg.MODEL.LSTR, 'GEN', 'CROSS', self.assa_init_w)
        self.refine_self_assa_init_w = _resolve_assa_init_w(cfg.MODEL.LSTR, 'REFINE', 'SELF', self.assa_init_w)
        self.refine_cross_assa_init_w = _resolve_assa_init_w(cfg.MODEL.LSTR, 'REFINE', 'CROSS', self.assa_init_w)

        # ContraNorm is kept local to the near-future generator to avoid changing
        # CMeRT's causal short-term/refinement path.
        self.gen_contranorm_enabled = _cfg_get(cfg.MODEL.LSTR, 'GEN_CONTRANORM', False)
        self.gen_contranorm_pos = _cfg_get(cfg.MODEL.LSTR, 'GEN_CONTRANORM_POS', 'none')
        if self.gen_contranorm_enabled and self.gen_contranorm_pos in ('', 'none'):
            self.gen_contranorm_pos = 'cross'
        self.gen_contranorm_scale = _cfg_get(cfg.MODEL.LSTR, 'GEN_CONTRANORM_SCALE', 0.1)
        self.gen_contranorm_temp = _cfg_get(cfg.MODEL.LSTR, 'GEN_CONTRANORM_TEMP', 1.0)
        self.gen_contranorm_dual_norm = _cfg_get(cfg.MODEL.LSTR, 'GEN_CONTRANORM_DUAL_NORM', False)
        self.gen_contranorm_learnable = _cfg_get(cfg.MODEL.LSTR, 'GEN_CONTRANORM_LEARNABLE', False)
        self.gen_contranorm_positive = _cfg_get(cfg.MODEL.LSTR, 'GEN_CONTRANORM_POSITIVE', False)
        self.gen_contranorm_identity = _cfg_get(cfg.MODEL.LSTR, 'GEN_CONTRANORM_IDENTITY', False)
        self.gen_contranorm_pre_norm = _cfg_get(cfg.MODEL.LSTR, 'GEN_CONTRANORM_PRE_NORM', False)

        # Multi-Query MoE is intentionally moved away from GEN. GEN can keep the
        # previous ASSA + ContraNorm path, while MQ-MoE is applied to non-ASSA
        # cross-attention sites such as DEC, REFINE, and LONG_COMPRESS.
        self.long_compress_mq_moe_cfg = _read_mq_moe_cfg(cfg.MODEL.LSTR, 'LONG_COMPRESS')
        self.dec_mq_moe_cfg = _read_mq_moe_cfg(cfg.MODEL.LSTR, 'DEC')
        self.refine_mq_moe_cfg = _read_mq_moe_cfg(cfg.MODEL.LSTR, 'REFINE')
        # Build LSTR encoder：compressed long-term
        if self.long_enabled:
            ## 1. ModuleList 中的模块会被自动注册到父模块中
            # 2. 参数可以被优化器识别和更新
            # 3. 可以使用 model.to(device) 自动移动所有模块
            self.enc_queries = nn.ModuleList()# enc_queries: 负责生成注意力机制中的查询向量
            self.enc_modules = nn.ModuleList()
            for i, param in enumerate(cfg.MODEL.LSTR.ENC_MODULE):#双阶段压缩
                if param[0] != -1:#使用decoder
                    long_stage_self_attention_type = _resolve_attention_type(
                        cfg.MODEL.LSTR, 'LONG_COMPRESS', 'SELF',
                        self.long_compress_self_attention_type, stage_idx=i)
                    long_stage_cross_attention_type = _resolve_attention_type(
                        cfg.MODEL.LSTR, 'LONG_COMPRESS', 'CROSS',
                        self.long_compress_cross_attention_type, stage_idx=i)
                    long_stage_self_assa_init_w = _resolve_assa_init_w(
                        cfg.MODEL.LSTR, 'LONG_COMPRESS', 'SELF', self.assa_init_w, stage_idx=i)
                    long_stage_cross_assa_init_w = _resolve_assa_init_w(
                        cfg.MODEL.LSTR, 'LONG_COMPRESS', 'CROSS', self.assa_init_w, stage_idx=i)
                    ## 创建一个可学习的查询向量矩阵，形状为 [param[0], d_model]   Ql
                    self.enc_queries.append(nn.Embedding(param[0], self.d_model))
                    enc_layer = tr.TransformerDecoderLayer(
                        self.d_model, self.num_heads, self.dim_feedforward,
                        self.dropout, self.activation,
                        self_attention_type=long_stage_self_attention_type,
                        cross_attention_type=long_stage_cross_attention_type,
                        assa_tau=self.assa_tau,
                        assa_init_w=self.assa_init_w,
                        self_assa_init_w=long_stage_self_assa_init_w,
                        cross_assa_init_w=long_stage_cross_assa_init_w,
                        mq_moe_cross=_use_mq_moe_cross(
                            self.long_compress_mq_moe_cfg['enabled'], long_stage_cross_attention_type),
                        mq_moe_experts=self.long_compress_mq_moe_cfg['experts'],
                        mq_moe_recent_k=self.long_compress_mq_moe_cfg['recent_k'],
                        mq_moe_gate_temperature=self.long_compress_mq_moe_cfg['gate_temperature'],
                        mq_moe_boundary_abs=self.long_compress_mq_moe_cfg['boundary_abs'])
                    ## 创建包含 param[1] 层的 TransformerDecoder
                    # param[2] 决定是否添加 LayerNorm
                    self.enc_modules.append(tr.TransformerDecoder(
                        enc_layer, param[1], tr.layer_norm(self.d_model, param[2])))
                else:
                    #使用encoder
                    self.enc_queries.append(None)
                    enc_layer = tr.TransformerEncoderLayer(
                        self.d_model, self.num_heads, self.dim_feedforward,
                        self.dropout, self.activation)  # ,attention_type=self.encoder_attention_type
                    self.enc_modules.append(tr.TransformerEncoder(
                        enc_layer, param[1], tr.layer_norm(self.d_model, param[2])))
        else:
            self.register_parameter('enc_queries', None)
            self.register_parameter('enc_modules', None)

        # Build LSTR decoder
        param = cfg.MODEL.LSTR.DEC_MODULE
        dec_layer = tr.TransformerDecoderLayer(
                self.d_model, self.num_heads, self.dim_feedforward,
                self.dropout, self.activation,
                self_attention_type=self.dec_self_attention_type,
                cross_attention_type=self.dec_cross_attention_type,
                assa_tau=self.assa_tau,
                assa_init_w=self.assa_init_w,
                self_assa_init_w=self.dec_self_assa_init_w,
                cross_assa_init_w=self.dec_cross_assa_init_w,
                mq_moe_cross=_use_mq_moe_cross(
                    self.dec_mq_moe_cfg['enabled'], self.dec_cross_attention_type),
                mq_moe_experts=self.dec_mq_moe_cfg['experts'],
                mq_moe_recent_k=self.dec_mq_moe_cfg['recent_k'],
                mq_moe_gate_temperature=self.dec_mq_moe_cfg['gate_temperature'],
                mq_moe_boundary_abs=self.dec_mq_moe_cfg['boundary_abs'])
        self.dec_modules = tr.TransformerDecoder(
                dec_layer, param[1], tr.layer_norm(self.d_model, param[2]))

        # Build Future Generation
        if self.future_enabled:
            param = cfg.MODEL.LSTR.GEN_MODULE
            # near-future queries
            self.gen_query = nn.Embedding(param[0], self.d_model)

            gen_layer = tr.TransformerDecoderLayer(
                self.d_model, self.num_heads, self.dim_feedforward,
                self.dropout, self.activation,
                self_attention_type=self.gen_self_attention_type,
                cross_attention_type=self.gen_cross_attention_type,
                assa_tau=self.assa_tau,
                assa_init_w=self.assa_init_w,
                self_assa_init_w=self.gen_self_assa_init_w,
                cross_assa_init_w=self.gen_cross_assa_init_w,
                contranorm_pos=self.gen_contranorm_pos,
                contranorm_scale=self.gen_contranorm_scale,
                contranorm_dual_norm=self.gen_contranorm_dual_norm,
                contranorm_temp=self.gen_contranorm_temp,
                contranorm_learnable=self.gen_contranorm_learnable,
                contranorm_positive=self.gen_contranorm_positive,
                contranorm_identity=self.gen_contranorm_identity,
                contranorm_pre_norm=self.gen_contranorm_pre_norm,
                # Future query 的 self-attention。
                attention_type='dotproduct',
                assa_v2_hidden=cfg.MODEL.LSTR.GEN_CROSS_ASSAV2_HIDDEN,
                assa_v2_bound=cfg.MODEL.LSTR.GEN_CROSS_ASSAV2_BOUND,
            )
            self.gen_layer = tr.TransformerDecoder(
                gen_layer, param[1], tr.layer_norm(self.d_model, param[2])
            )
            #anticipation queries
            self.final_query = nn.Embedding(cfg.MODEL.LSTR.FUT_MODULE[0][0], self.d_model)

            #refinement模块
            self.work_fusions = nn.ModuleList()
            for i in range(cfg.MODEL.LSTR.REFINE_TIMES):
                work_enc_layer = tr.TransformerDecoderLayer(
                    self.d_model, self.num_heads, self.dim_feedforward,
                    self.dropout, self.activation,
                    self_attention_type=self.refine_self_attention_type,
                    cross_attention_type=self.refine_cross_attention_type,
                    assa_tau=self.assa_tau,
                    assa_init_w=self.assa_init_w,
                    self_assa_init_w=self.refine_self_assa_init_w,
                    cross_assa_init_w=self.refine_cross_assa_init_w,
                    mq_moe_cross=_use_mq_moe_cross(
                        self.refine_mq_moe_cfg['enabled'], self.refine_cross_attention_type),
                    mq_moe_experts=self.refine_mq_moe_cfg['experts'],
                    mq_moe_recent_k=self.refine_mq_moe_cfg['recent_k'],
                    mq_moe_gate_temperature=self.refine_mq_moe_cfg['gate_temperature'],
                    mq_moe_boundary_abs=self.refine_mq_moe_cfg['boundary_abs'])
                self.work_fusions.append(tr.TransformerDecoder(
                    work_enc_layer, 1, tr.layer_norm(self.d_model, True)))

        else:
            if self.anticipation_num_samples > 0 :
                #EPIC-Kitchens 数据集中的每个动作标签包含三个层次
                self.final_query = nn.Embedding(self.anticipation_num_samples*3, self.d_model)
        # Build classifier
        if cfg.MODEL.LSTR.DROPOUT_CLS > 0:
            self.dropout_cls = nn.Dropout(cfg.MODEL.LSTR.DROPOUT_CLS)
        else:
            self.dropout_cls = None
        if cfg.MODEL.LSTR.FC_NORM:
            self.classifier = NormalizedLinear(self.d_model, self.num_classes)
        else:
            self.classifier = nn.Linear(self.d_model, self.num_classes)
        if self.cfg.DATA.DATA_NAME == 'EK100':
            self.classifier_verb = nn.Linear(self.d_model, 98)
            self.classifier_noun = nn.Linear(self.d_model, 301)
    #memory-refine
    def refine(self, memory, output, mask):
        #compressed long-term
        his_memory = memory

        enc_query = self.gen_query.weight.unsqueeze(1).repeat(1, his_memory.shape[1], 1)
        #提取near-future信息
        future = self.gen_layer(enc_query, his_memory)  # , knn=True

        future_rep = [future]
        #Msa
        short_rep = [output]
        #REFINE_TIMES
        for i in range(self.cfg.MODEL.LSTR.REFINE_TIMES):
            mask1 = torch.zeros((output.shape[0], memory.shape[0])).to(output.device)
            mask2 = torch.zeros((output.shape[0], future.shape[0])).to(output.device)
            the_mask = torch.cat((mask1, mask, mask2), dim=-1)
            total_memory = torch.cat([memory, output, future])
            output = self.work_fusions[i](output, total_memory, tgt_mask=mask, memory_mask=the_mask)  # , knn=True
            short_rep.append(output)
        #返回near-future，以及Msa:{精炼前，精炼后}
        return short_rep, future_rep

    def forward(self, visual_inputs, motion_inputs, memory_key_padding_mask=None):
        if self.long_enabled:
            # Compute long memories
            if self.long_memory_use_pe:
                long_memories = self.pos_encoding(self.feature_head_long(
                    visual_inputs[:, :self.long_memory_num_samples],
                    motion_inputs[:, :self.long_memory_num_samples]
                ).transpose(0, 1))
                long_memories = long_memories.transpose(0, 1)  ## transpose back
            else:
                long_memories = self.feature_head_long(
                    visual_inputs[:, :self.long_memory_num_samples],
                    motion_inputs[:, :self.long_memory_num_samples]
                )
                #transformer标准输入[seq_len,批次大小，维度]
            batch_size = long_memories.shape[0]
            long_memories = long_memories.view(batch_size, -1, self.d_model).transpose(0, 1)
            #long-term features提取完成
            #开始压缩long-term
            if len(self.enc_modules) > 0:
                #准备query
                enc_queries = [
                    #query有两个，一个是16，一个是32，shape：[16，dimension]，unsqueeze添加一个批次维度，使用repeat沿批次复制
                    enc_query.weight.unsqueeze(1).repeat(1, long_memories.shape[1], 1)
                    if enc_query is not None else None
                    for enc_query in self.enc_queries
                ]

                # Encode long memories
                if enc_queries[0] is not None:
                    long_memories = self.enc_modules[0](enc_queries[0], long_memories,
                                                        memory_key_padding_mask=memory_key_padding_mask)
                else:
                    long_memories = self.enc_modules[0](long_memories)
                for enc_query, enc_module in zip(enc_queries[1:], self.enc_modules[1:]):
                    if enc_query is not None:
                        long_memories = enc_module(enc_query, long_memories)
                    else:
                        long_memories = enc_module(long_memories)
        #压缩long-term完成，存入long_memories
        # Concatenate memories
        if self.long_enabled:
            memory = long_memories

        # Compute work memories：短时记忆
        visual_inputs_avg = visual_inputs[:, self.long_memory_num_samples:]
        motion_inputs_avg = motion_inputs[:, self.long_memory_num_samples:]
        work_memories_no_pe = self.feature_head_work(
            visual_inputs_avg,
            motion_inputs_avg,
        ).transpose(0, 1)
        if self.work_memory_use_pe:
            work_memories = self.pos_encoding(work_memories_no_pe,
                                              padding=self.long_memory_num_samples if self.long_memory_use_pe else 0)
        else:
            work_memories = work_memories_no_pe

        #构造anticipation queries
        if self.anticipation_num_samples > 0:
            anticipate_memories = self.pos_encoding(
                self.final_query.weight[
                :self.cfg.MODEL.LSTR.ANTICIPATION_LENGTH:self.cfg.MODEL.LSTR.ANTICIPATION_SAMPLE_RATE, ...]. \
                    unsqueeze(1).repeat(1, work_memories.shape[1], 1),
                padding=self.long_memory_num_samples + self.work_memory_num_samples if self.long_memory_use_pe else self.work_memory_num_samples)
            work_memories = torch.cat((work_memories, anticipate_memories), dim=0)

        # Build mask
        mask = tr.generate_square_subsequent_mask(work_memories.shape[0]).to(work_memories.device)
        if self.anticipation_num_samples > 0:
            #使anticipation可以互相交互
            mask1 = torch.zeros((self.anticipation_num_samples, self.anticipation_num_samples)).to(work_memories.device)
            mask[-self.anticipation_num_samples:, -self.anticipation_num_samples:] = mask1
        # Compute output：目前长时记忆已压缩，短时记忆和anticipation已拼接
        if self.long_enabled:
            #include_work控制是否将短时记忆作为训练样本的inmediate context
            if self.include_work:
                #work_memories=short-term+anticipation
                extended_memory = torch.cat((memory, work_memories), dim=0)
                extended_mask = F.pad(mask,
                                      (memory.shape[0], 0),
                                      'constant', 0)
                extended_mask = extended_mask.to(extended_memory.device)
                ##context-enhanced
                output = self.dec_modules(
                    work_memories,
                    memory=extended_memory,
                    tgt_mask=mask,
                    memory_mask=extended_mask,
                )
            else:
                output = self.dec_modules(
                    work_memories,
                    memory=memory,
                    tgt_mask=mask,
                )
        else:
            output = self.dec_modules(
                work_memories,
                src_mask=mask,
            )

        if self.future_enabled:
            #流向refine模块
            #只需要long-term，去除near-past的short-term
            #内部会生成near-future，并进行拼接
            works, futs = self.refine(memory, output[self.context_num_samples:], mask[self.context_num_samples:, self.context_num_samples:])
            work_scores = []
            fut_scores = []
            for i, work in enumerate(works):
                #精炼前和精炼后都需要计算损失
                if i == len(works) - 1 and self.cfg.DATA.DATA_NAME == 'EK100':
                    noun_score = self.classifier_noun(work).transpose(0, 1)
                    verb_score = self.classifier_verb(work).transpose(0, 1)
                    work_scores.append(self.classifier(self.dropout_cls(work)).transpose(0, 1))
                else:
                    work_scores.append(self.classifier(work).transpose(0, 1))
            for i, fut in enumerate(futs):
                if i == 0:
                    if i == len(futs) - 1 and self.cfg.DATA.DATA_NAME == 'EK100':
                        fut_noun_score = self.classifier_noun(fut).transpose(0, 1)
                        fut_verb_score = self.classifier_verb(fut).transpose(0, 1)
                    fut_scores.append(self.classifier(
                        F.interpolate(fut.permute(1, 2, 0), size=self.future_num_samples).permute(2, 0, 1)).transpose(0,
                                                                                                                      1))
                else:
                    fut_scores.append(self.classifier(fut).transpose(0, 1))
                    if i == len(futs) - 1 and self.cfg.DATA.DATA_NAME == 'EK100':
                        fut_noun_score = self.classifier_noun(fut).transpose(0, 1)
                        fut_verb_score = self.classifier_verb(fut).transpose(0, 1)
            return (work_scores, fut_scores) if self.cfg.DATA.DATA_NAME != 'EK100' else (work_scores, fut_scores, noun_score, fut_noun_score, verb_score, fut_verb_score)

        else:
            work_scores = []
            fut_scores = []
            work_scores.append(self.classifier(output[self.context_num_samples:]).transpose(0, 1))
            if self.cfg.DATA.DATA_NAME == 'EK100':
                # 方案 A: 对序列中每个点都分类 (如果 Loss 支持序列标签)
                noun_score = self.classifier_noun(output[self.context_num_samples:]).transpose(0, 1)
                verb_score = self.classifier_verb(output[self.context_num_samples:]).transpose(0, 1)

                # 方案 B: 如果只需要最后一个时刻的预测 (常见于视频级标签)
                # noun_score = self.classifier_noun(output[self.context_num_samples:][-1:]).transpose(0, 1)
                # verb_score = self.classifier_verb(output[self.context_num_samples:][-1:]).transpose(0, 1)
                return work_scores, fut_scores, noun_score, None, verb_score, None
            else:
                return work_scores, fut_scores
@META_ARCHITECTURES.register('CMert')
class FROAD_CTX(nn.Module):
    # provide context with directional  mask, no leakage and anticipate short
    def __init__(self, cfg):
        super(FROAD_CTX, self).__init__()

        self.cfg = cfg
        # Build long feature heads
        self.long_memory_num_samples = cfg.MODEL.LSTR.LONG_MEMORY_NUM_SAMPLES
        self.long_enabled = self.long_memory_num_samples > 0
        if self.long_enabled:
            self.feature_head_long = BaseFeatureHead_mat(cfg)

        # Build work feature head
        self.work_memory_num_samples = cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES
        self.work_enabled = self.work_memory_num_samples > 0
        if self.work_enabled:
            self.feature_head_work = BaseFeatureHead_mat(cfg)

        self.context_num_samples = int(cfg.MODEL.LSTR.CONTEXT_SECONDS * cfg.DATA.FPS) // cfg.MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE
        self.anticipation_num_samples = cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES
        self.future_num_samples = cfg.MODEL.LSTR.FUTURE_NUM_SAMPLES
        self.future_enabled = self.future_num_samples > 0

        self.d_model = self.feature_head_work.d_model
        self.num_heads = cfg.MODEL.LSTR.NUM_HEADS
        self.dim_feedforward = cfg.MODEL.LSTR.DIM_FEEDFORWARD
        self.dropout = cfg.MODEL.LSTR.DROPOUT
        self.activation = cfg.MODEL.LSTR.ACTIVATION
        self.num_classes = cfg.DATA.NUM_CLASSES

        self.use_v_n = cfg.MODEL.LSTR.V_N_CLASSIFIER

        self.knn = False  # cfg.MODEL.LSTR.ATT_KNN
        self.ratio = 0.0  # cfg.MODEL.LSTR.ATT_KNN_RATIO

        # Build position encoding
        self.pos_encoding = tr.PositionalEncoding(self.d_model, self.dropout)

        self.long_memory_use_pe = cfg.MODEL.LSTR.LONG_MEMORY_USE_PE
        self.work_memory_use_pe = cfg.MODEL.LSTR.WORK_MEMORY_USE_PE
        self.include_work = cfg.MODEL.LSTR.LONG_MEMORY_INCLUDE_WORK

        # EK100 uses the registry name ``CMert`` rather than ``CMeRT``.
        # Mirror the THUMOS CMeRT generator switches here so the same
        # command-line overrides work on EK100:
        #
        #   MODEL.LSTR.GEN_CROSS_ATTN_TYPE assa
        #   MODEL.LSTR.GEN_CROSS_ASSA_INIT_W '[0.0, 0.0]'
        #   MODEL.LSTR.GEN_CONTRANORM True
        #   MODEL.LSTR.GEN_CONTRANORM_POS cross
        #   MODEL.LSTR.GEN_CONTRANORM_SCALE 0.05
        #
        # Only the near-future generator is changed. Long compression,
        # the short-memory decoder, and refinement keep dot-product attention.
        self.assa_tau = _cfg_get(cfg.MODEL.LSTR, 'ASSA_TAU', 1.0)
        self.assa_init_w = _cfg_get(
            cfg.MODEL.LSTR, 'ASSA_INIT_W', [1.0, 1.0])

        self.gen_self_attention_type = _cfg_get(
            cfg.MODEL.LSTR, 'GEN_SELF_ATTN_TYPE', 'dotproduct')
        self.gen_cross_attention_type = _cfg_get(
            cfg.MODEL.LSTR, 'GEN_CROSS_ATTN_TYPE', 'dotproduct')

        self.gen_self_assa_init_w = _resolve_assa_init_w(
            cfg.MODEL.LSTR, 'GEN', 'SELF', self.assa_init_w)
        self.gen_cross_assa_init_w = _resolve_assa_init_w(
            cfg.MODEL.LSTR, 'GEN', 'CROSS', self.assa_init_w)

        self.gen_contranorm_enabled = bool(_cfg_get(
            cfg.MODEL.LSTR, 'GEN_CONTRANORM', False))
        requested_contranorm_pos = str(_cfg_get(
            cfg.MODEL.LSTR, 'GEN_CONTRANORM_POS', 'cross')).lower()

        if requested_contranorm_pos not in ('none', 'cross'):
            raise ValueError(
                "GEN_CONTRANORM_POS for CMert/EK100 must be 'none' or 'cross'")

        # The current TransformerDecoderLayer API enables ContraNorm through
        # contranorm_pos. Force it to 'none' when GEN_CONTRANORM is False.
        self.gen_contranorm_pos = (
            requested_contranorm_pos
            if self.gen_contranorm_enabled
            else 'none'
        )
        self.gen_contranorm_scale = _cfg_get(
            cfg.MODEL.LSTR, 'GEN_CONTRANORM_SCALE', 0.05)
        self.gen_contranorm_temp = _cfg_get(
            cfg.MODEL.LSTR, 'GEN_CONTRANORM_TEMP', 1.0)
        self.gen_contranorm_dual_norm = _cfg_get(
            cfg.MODEL.LSTR, 'GEN_CONTRANORM_DUAL_NORM', False)
        self.gen_contranorm_learnable = _cfg_get(
            cfg.MODEL.LSTR, 'GEN_CONTRANORM_LEARNABLE', False)
        self.gen_contranorm_positive = _cfg_get(
            cfg.MODEL.LSTR, 'GEN_CONTRANORM_POSITIVE', False)
        self.gen_contranorm_identity = _cfg_get(
            cfg.MODEL.LSTR, 'GEN_CONTRANORM_IDENTITY', False)
        self.gen_contranorm_pre_norm = _cfg_get(
            cfg.MODEL.LSTR, 'GEN_CONTRANORM_PRE_NORM', False)

        # Build LSTR encoder
        if self.long_enabled:
            self.enc_queries = nn.ModuleList()
            self.enc_modules = nn.ModuleList()
            for i, param in enumerate(cfg.MODEL.LSTR.ENC_MODULE):
                if param[0] != -1:
                    self.enc_queries.append(nn.Embedding(param[0], self.d_model))
                    enc_layer = tr.TransformerDecoderLayer(
                        self.d_model, self.num_heads, self.dim_feedforward,
                        self.dropout, self.activation)
                    self.enc_modules.append(tr.TransformerDecoder(
                        enc_layer, param[1], tr.layer_norm(self.d_model, param[2])))
                else:
                    self.enc_queries.append(None)
                    enc_layer = tr.TransformerEncoderLayer(
                        self.d_model, self.num_heads, self.dim_feedforward,
                        self.dropout, self.activation)
                    self.enc_modules.append(tr.TransformerEncoder(
                        enc_layer, param[1], tr.layer_norm(self.d_model, param[2])))
        else:
            self.register_parameter('enc_queries', None)
            self.register_parameter('enc_modules', None)

        # Build LSTR decoder
        param = cfg.MODEL.LSTR.DEC_MODULE
        dec_layer = tr.TransformerDecoderLayer(
                self.d_model, self.num_heads, self.dim_feedforward,
                self.dropout, self.activation)
        self.dec_modules = tr.TransformerDecoder(
            dec_layer, param[1], tr.layer_norm(self.d_model, param[2]))

        # Build future Generation
        if self.future_enabled:
            param = cfg.MODEL.LSTR.GEN_MODULE
            self.gen_query = nn.Embedding(param[0], self.d_model)
            gen_layer = tr.TransformerDecoderLayer(
                self.d_model, self.num_heads, self.dim_feedforward,
                self.dropout, self.activation,
                self_attention_type=self.gen_self_attention_type,
                cross_attention_type=self.gen_cross_attention_type,
                assa_tau=self.assa_tau,
                assa_init_w=self.assa_init_w,
                self_assa_init_w=self.gen_self_assa_init_w,
                cross_assa_init_w=self.gen_cross_assa_init_w,
                contranorm_pos=self.gen_contranorm_pos,
                contranorm_scale=self.gen_contranorm_scale,
                contranorm_dual_norm=self.gen_contranorm_dual_norm,
                contranorm_temp=self.gen_contranorm_temp,
                contranorm_learnable=self.gen_contranorm_learnable,
                contranorm_positive=self.gen_contranorm_positive,
                contranorm_identity=self.gen_contranorm_identity,
                contranorm_pre_norm=self.gen_contranorm_pre_norm,
                attention_type='dotproduct',
                assa_v2_hidden=_cfg_get(
                    cfg.MODEL.LSTR, 'GEN_CROSS_ASSAV2_HIDDEN', 64),
                assa_v2_bound=_cfg_get(
                    cfg.MODEL.LSTR, 'GEN_CROSS_ASSAV2_BOUND', 1.0),
            )
            self.gen_layer = tr.TransformerDecoder(
                gen_layer, param[1], tr.layer_norm(self.d_model, param[2])
            )
            self.final_query = nn.Embedding(cfg.MODEL.LSTR.FUT_MODULE[0][0], self.d_model)

            self.work_fusions = nn.ModuleList()
            for i in range(cfg.MODEL.LSTR.REFINE_TIMES):
                work_enc_layer = tr.TransformerDecoderLayer(
                    self.d_model, self.num_heads, self.dim_feedforward,
                    self.dropout, self.activation)
                self.work_fusions.append(tr.TransformerDecoder(
                    work_enc_layer, 1, tr.layer_norm(self.d_model, True)))

        # Build classifier
        self.dropout_cls = None
        if cfg.MODEL.LSTR.DROPOUT_CLS > 0:
            self.dropout_cls = nn.Dropout(cfg.MODEL.LSTR.DROPOUT_CLS)

        self.classifier = nn.Linear(self.d_model, self.num_classes)

        self.dropout_vn = None
        if self.use_v_n:
            self.classifier_verb = nn.Linear(self.d_model, 98)
            self.classifier_noun = nn.Linear(self.d_model, 301)

            if cfg.MODEL.LSTR.DROPOUT_VN > 0:
                self.dropout_vn = nn.Dropout(cfg.MODEL.LSTR.DROPOUT_VN)

    def refine(self, memory, output, mask):
        his_memory = memory
        enc_query = self.gen_query.weight.unsqueeze(1).repeat(1, his_memory.shape[1], 1)
        future = self.gen_layer(enc_query, his_memory, knn=self.knn)  # , knn=True

        future_rep = [future]
        short_rep = [output]
        for i in range(self.cfg.MODEL.LSTR.REFINE_TIMES):
            mask1 = torch.zeros((output.shape[0], memory.shape[0])).to(output.device)
            mask2 = torch.zeros((output.shape[0], future.shape[0])).to(output.device)
            the_mask = torch.cat((mask1, mask, mask2), dim=-1)
            total_memory = torch.cat([memory, output, future])
            output = self.work_fusions[i](output, total_memory, tgt_mask=mask, memory_mask=the_mask,
                                          knn=self.knn)  # , knn=True
            short_rep.append(output)


        return short_rep, future_rep

    def forward(self, visual_inputs, motion_inputs, memory_key_padding_mask=None):
        if self.long_enabled:
            # Compute long memories
            if self.long_memory_use_pe:
                long_memories = self.pos_encoding(self.feature_head_long(
                    visual_inputs[:, :self.long_memory_num_samples],
                    motion_inputs[:, :self.long_memory_num_samples],
                ).transpose(0, 1))
                long_memories = long_memories.transpose(0, 1)  ## transpose back
            else:
                long_memories = self.feature_head_long(
                    visual_inputs[:, :self.long_memory_num_samples],
                    motion_inputs[:, :self.long_memory_num_samples],
                )
            batch_size = long_memories.shape[0]
            long_memories = long_memories.view(batch_size, -1, self.d_model).transpose(0, 1)

            if len(self.enc_modules) > 0:
                enc_queries = [
                    enc_query.weight.unsqueeze(1).repeat(1, long_memories.shape[1], 1)
                    if enc_query is not None else None
                    for enc_query in self.enc_queries
                ]

                # Encode long memories
                if enc_queries[0] is not None:
                    long_memories = self.enc_modules[0](enc_queries[0], long_memories, memory_key_padding_mask= \
                        memory_key_padding_mask, knn=self.knn)

                else:
                    long_memories = self.enc_modules[0](long_memories)
                for enc_query, enc_module in zip(enc_queries[1:], self.enc_modules[1:]):
                    if enc_query is not None:
                        long_memories = enc_module(enc_query, long_memories, knn=self.knn)
                    else:
                        long_memories = enc_module(long_memories, knn=self.knn)

        # Concatenate memories
        if self.long_enabled:
            memory = long_memories

        if self.work_enabled:
            # Compute work memories
            visual_inputs_avg = visual_inputs[:, self.long_memory_num_samples - self.context_num_samples:]
            motion_inputs_avg = motion_inputs[:, self.long_memory_num_samples - self.context_num_samples:]
            work_memories_no_pe = self.feature_head_work(
                visual_inputs_avg,
                motion_inputs_avg,
            ).transpose(0, 1)
            if self.work_memory_use_pe:
                work_memories = self.pos_encoding(work_memories_no_pe,
                                                  padding=self.long_memory_num_samples if self.long_memory_use_pe else 0)
            else:
                work_memories = work_memories_no_pe

            if self.anticipation_num_samples > 0:
                anticipate_memories = self.pos_encoding(
                    self.final_query.weight[
                    :self.cfg.MODEL.LSTR.ANTICIPATION_LENGTH:self.cfg.MODEL.LSTR.ANTICIPATION_SAMPLE_RATE, ...]. \
                        unsqueeze(1).repeat(1, work_memories.shape[1], 1),
                    padding=self.long_memory_num_samples + work_memories.shape[0] if self.long_memory_use_pe else work_memories.shape[0])
                work_memories = torch.cat((work_memories, anticipate_memories), dim=0)

            # Build mask
            mask = tr.generate_square_subsequent_mask(work_memories.shape[0]).to(work_memories.device)
            if self.anticipation_num_samples > 0:
                mask1 = torch.zeros((self.anticipation_num_samples, self.anticipation_num_samples)).to(work_memories.device)
                mask[-self.anticipation_num_samples:, -self.anticipation_num_samples:] = mask1
            # Compute output
            if self.long_enabled:
                if self.include_work:
                    extended_memory = torch.cat((memory, work_memories), dim=0)
                    extended_mask = F.pad(mask, (memory.shape[0], 0), 'constant', 0)
                    extended_mask = extended_mask.to(extended_memory.device)
                    output = self.dec_modules(
                        work_memories,
                        memory=extended_memory,
                        tgt_mask=mask,
                        memory_mask=extended_mask,
                        knn=self.knn
                    )
                else:
                    output = self.dec_modules(
                        work_memories,
                        memory=memory,
                        tgt_mask=mask,
                        knn=self.knn
                    )
            else:
                output = self.dec_modules(
                    work_memories,
                    src_mask=mask,
                    knn=self.knn
                )

        if self.future_enabled:
            works, futs = self.refine(memory, output[self.context_num_samples:], mask[self.context_num_samples:, self.context_num_samples:])
            work_scores = []
            fut_scores = []
            for i, work in enumerate(works):
                if i == len(works) - 1:
                    if self.dropout_cls is not None:
                        output_a = self.dropout_cls(work)
                    else:
                        output_a = work
                    if self.dropout_vn is not None:
                        output_v = self.dropout_vn(work)
                        output_n = self.dropout_vn(work)
                    else:
                        output_v = work
                        output_n = work
                    noun_score = self.classifier_noun(output_n).transpose(0, 1)
                    verb_score = self.classifier_verb(output_v).transpose(0, 1)
                    work_scores.append(self.classifier(output_a).transpose(0, 1))
                else:
                    work_scores.append(self.classifier(work).transpose(0, 1))

            for i, fut in enumerate(futs):
                fut = F.interpolate(fut.permute(1, 2, 0), size=self.future_num_samples).permute(2, 0, 1)
                fut_scores.append(self.classifier(fut).transpose(0, 1))
                if i == len(futs) - 1:
                    fut_noun_score = self.classifier_noun(fut).transpose(0, 1)
                    fut_verb_score = self.classifier_verb(fut).transpose(0, 1)

            return (work_scores, fut_scores, noun_score, fut_noun_score, verb_score, fut_verb_score)

        # Compute classification score
        if self.dropout_cls is not None:
            output_a = self.dropout_cls(output)
        else:
            output_a = output
        score = self.classifier(output_a)
        score = score.transpose(0, 1)

        if self.use_v_n:
            if self.dropout_vn is not None:
                output_v = self.dropout_vn(output)
                output_n = self.dropout_vn(output)
            else:
                output_v = output
                output_n = output
            score_verb = self.classifier_verb(output_v).transpose(0, 1)
            score_noun = self.classifier_noun(output_n).transpose(0, 1)
            score = (score, score_verb, score_noun)

        return score