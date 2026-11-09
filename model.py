"""TempoLS：固定使用单层 1×1 奇偶耦合，默认空间交互宽度为 128。

删除选层配置、中间交互宽度、时间卷积填充及旧权重名称转换逻辑。
保留空间瓶颈、残差、预测头与原 checkpoint 配置。
模型使用 model.to(device) 放置设备。训练参数中不再传入选层参数。
同配置的原 second_only 模型参数名称和形状保留，可直接严格加载其 state_dict。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class PeriodicTemporalEmbedding(nn.Module):
    """根据最后时刻生成日内时间和星期嵌入。"""

    def __init__(self, time_steps_per_day, embedding_dim):
        super().__init__()
        self.time_steps_per_day = time_steps_per_day
        self.time_of_day_embedding_table = nn.Parameter(torch.empty(time_steps_per_day, embedding_dim))
        nn.init.xavier_uniform_(self.time_of_day_embedding_table)
        self.day_of_week_embedding_table = nn.Parameter(torch.empty(7, embedding_dim))
        nn.init.xavier_uniform_(self.day_of_week_embedding_table)

    def forward(self, history_features):
        time_of_day_values = history_features[..., 1]
        time_of_day_indices = (time_of_day_values[:, -1, :] * self.time_steps_per_day).long()
        time_of_day_features = self.time_of_day_embedding_table[time_of_day_indices]
        time_of_day_features = time_of_day_features.transpose(1, 2).unsqueeze(-1)
        day_of_week_values = history_features[..., 2]
        day_of_week_indices = day_of_week_values[:, -1, :].long()
        day_of_week_features = self.day_of_week_embedding_table[day_of_week_indices]
        day_of_week_features = day_of_week_features.transpose(1, 2).unsqueeze(-1)
        return time_of_day_features + day_of_week_features

class LearnableLatentSpaceInteraction(nn.Module):
    """节点→潜在空间→节点的低秩空间传播。"""

    def __init__(self, channels, num_nodes, latent_rank=32):
        super().__init__()
        if not isinstance(latent_rank, int) or isinstance(latent_rank, bool) or latent_rank < 1:
            raise ValueError('latent_rank must be a positive integer')
        self.node_feature_projection = nn.Conv2d(channels, channels, 1)
        self.spatial_output_projection = nn.Conv2d(channels, channels, 1)
        self.node_to_latent_logits = nn.Parameter(torch.empty(latent_rank, num_nodes))
        self.latent_to_node_logits = nn.Parameter(torch.empty(latent_rank, num_nodes))
        nn.init.xavier_uniform_(self.node_to_latent_logits)
        nn.init.xavier_uniform_(self.latent_to_node_logits)
        self.latent_relation_logits = nn.Parameter(torch.ones(latent_rank, latent_rank))

    def forward(self, node_features):
        node_to_latent_weights = F.softmax(self.node_to_latent_logits, dim=1)
        latent_to_node_weights = F.softmax(self.latent_to_node_logits, dim=0)
        latent_relation_weights = F.softmax(self.latent_relation_logits, dim=-1)
        projected_node_features = self.node_feature_projection(node_features)
        batch_size, channels, num_nodes, sequence_length = projected_node_features.shape
        flattened_node_features = projected_node_features.permute(0, 2, 1, 3).contiguous().view(batch_size, num_nodes, -1)
        node_to_latent_weights = node_to_latent_weights.to(dtype=flattened_node_features.dtype)
        latent_to_node_weights = latent_to_node_weights.to(dtype=flattened_node_features.dtype)
        latent_relation_weights = latent_relation_weights.to(dtype=flattened_node_features.dtype)
        latent_features = torch.matmul(node_to_latent_weights, flattened_node_features)
        propagated_latent_features = torch.matmul(latent_relation_weights, latent_features)
        reconstructed_node_features = torch.matmul(latent_to_node_weights.transpose(0, 1), propagated_latent_features)
        reconstructed_node_features = reconstructed_node_features.view(batch_size, num_nodes, channels, sequence_length).permute(0, 2, 1, 3).contiguous()
        return self.spatial_output_projection(reconstructed_node_features)

class NodeSpecificChannelInteraction(nn.Module):
    """每个节点、时间位置的动态 2×2 通道混合。"""

    def __init__(self, channels: int, num_nodes: int, dropout: float=0.1, reduction: int=2) -> None:
        super().__init__()
        if channels < 2 or channels % 2 != 0:
            raise ValueError('channels 必须为不小于 2 的偶数。')
        if num_nodes < 1 or reduction < 1:
            raise ValueError('num_nodes 和 reduction 必须为正。')
        self.channel_dim = channels
        self.num_nodes = num_nodes
        self.num_channel_groups = channels // 2
        hidden = max(1, channels // reduction)
        self.channel_value_projection = nn.Conv2d(channels, channels, kernel_size=1)
        self.channel_interaction_generator = nn.Sequential(nn.Conv2d(channels, hidden, kernel_size=1), nn.ReLU(), nn.Conv2d(hidden, 2 * channels, kernel_size=1))
        node_channel_response = torch.eye(2).reshape(1, 1, 1, 1, 2, 2)
        node_channel_response = node_channel_response.repeat(1, num_nodes, 1, self.num_channel_groups, 1, 1)
        self.node_channel_response_matrices = nn.Parameter(node_channel_response)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features):
        B, C, N, T = features.shape
        if C != self.channel_dim or N != self.num_nodes:
            raise ValueError('输入通道数或节点数与初始化配置不一致。')
        G = self.num_channel_groups
        grouped_channel_values = self.channel_value_projection(features).reshape(B, G, 2, N, T)
        input_dependent_channel_matrices = self.channel_interaction_generator(features).reshape(B, G, 2, 2, N, T)
        node_channel_response = self.node_channel_response_matrices[0, :, 0].to(dtype=input_dependent_channel_matrices.dtype)
        adapted_channel_matrices = torch.einsum('bgiknt,ngkj->bgijnt', input_dependent_channel_matrices, node_channel_response)
        channel_interaction_matrices = torch.tanh(adapted_channel_matrices) / 2 ** 0.5
        mixed_channel_features = torch.einsum('bgijnt,bgjnt->bgint', channel_interaction_matrices, grouped_channel_values)
        mixed_channel_features = mixed_channel_features.reshape(B, C, N, T).contiguous()
        return self.dropout(mixed_channel_features)

class SpatialInteractionModule(nn.Module):
    """降维→空间传播和通道交互→升维。"""

    def __init__(self, channels, num_nodes, dropout=0.1, latent_rank=32, interaction_channels=None):
        super().__init__()
        width = channels //2
        self.interaction_input_projection = nn.Conv2d(channels, width, kernel_size=1) if width != channels else nn.Identity()
        self.learnable_latent_space_interaction = LearnableLatentSpaceInteraction(width, num_nodes, latent_rank=latent_rank)
        self.spatial_interaction_dropout = nn.Dropout(p=dropout)
        self.node_specific_channel_interaction = NodeSpecificChannelInteraction(width, num_nodes)
        self.interaction_output_projection = nn.Conv2d(width, channels, kernel_size=1) if width != channels else nn.Identity()

    def forward(self, node_features):
        reduced_features = self.interaction_input_projection(node_features)
        spatially_propagated_features = reduced_features + torch.relu(self.learnable_latent_space_interaction(reduced_features))
        spatially_propagated_features = self.spatial_interaction_dropout(spatially_propagated_features)
        interacted_features = self.node_specific_channel_interaction(spatially_propagated_features)
        return self.interaction_output_projection(interacted_features)

class CoupledTemporalTransform(nn.Module):
    """单层 1×1 奇偶耦合：共享自身/跨分支权重。"""

    def __init__(self, channels, out_channels):
        super().__init__()
        if channels < 1 or out_channels < 1:
            raise ValueError('channels and out_channels must be positive')
        self.same_branch_projection = nn.Conv2d(channels, out_channels, 1, bias=False)
        self.cross_branch_projection = nn.Conv2d(channels, out_channels, 1, bias=False)
        self.even_bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        self.odd_bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

    def forward(self, even_features, odd_features):
        wr = self.same_branch_projection.weight
        wi = self.cross_branch_projection.weight
        block_weight = torch.cat((torch.cat((wr, -wi), dim=1), torch.cat((wi, wr), dim=1)), dim=0)
        block_bias = torch.cat((self.even_bias.flatten(), self.odd_bias.flatten()))
        paired_features = torch.cat((even_features, odd_features), dim=1)
        mixed = F.conv2d(paired_features, block_weight, block_bias)
        return F.relu(mixed).chunk(2, dim=1)

class BidirectionalEvenOddInteraction(nn.Module):
    """逐位置奇偶耦合、分支残差和输出融合。"""

    def __init__(self, channels, out_channels):
        super().__init__()
        self.channels = channels
        self.coupled_output_transform = CoupledTemporalTransform(channels, out_channels)
        self.even_residual_projection = nn.Conv2d(channels, out_channels, kernel_size=1) if channels != out_channels else nn.Identity()
        self.odd_residual_projection = nn.Conv2d(channels, out_channels, kernel_size=1) if channels != out_channels else nn.Identity()
        self.branch_fusion_projection = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)

    def forward(self, even_features, odd_features):
        if even_features.ndim != 4 or odd_features.ndim != 4 or even_features.shape != odd_features.shape or (even_features.shape[1] != self.channels) or (even_features.shape[2] < 1) or (even_features.shape[3] < 1):
            raise ValueError('even_features and odd_features must have the same nonempty [B, channels, N, T] shape with the configured input channels')
        mixed_even, mixed_odd = self.coupled_output_transform(even_features, odd_features)
        updated_even_features = self.even_residual_projection(even_features) + mixed_even
        updated_odd_features = self.odd_residual_projection(odd_features) + mixed_odd
        concatenated_features = torch.cat((updated_even_features, updated_odd_features), dim=1)
        return self.branch_fusion_projection(concatenated_features)

def split_even_odd(features):
    even, odd = (features[..., ::2], features[..., 1::2])
    if odd.size(-1) == 0:
        raise ValueError('Temporal splitting requires at least two time steps')
    if even.size(-1) > odd.size(-1):
        odd = F.pad(odd, (0, 1, 0, 0), mode='replicate')
    return (even, odd)

class ParityInputStage(nn.Module):
    """奇偶拆分、共享输入投影、逐位置耦合。"""

    def __init__(self, input_dim=3, channels=16):
        super().__init__()
        self.projection = nn.Conv2d(input_dim, channels, kernel_size=1)
        self.interaction = BidirectionalEvenOddInteraction(channels=channels, out_channels=channels)

    def forward(self, history_data):
        even, odd = split_even_odd(history_data)
        even = self.projection(even)
        odd = self.projection(odd)
        return self.interaction(even, odd)

class RecursiveTemporalAggregationLayer(nn.Module):
    """通过奇偶聚合将时间长度减半。"""

    def __init__(self, channels, out_channels):
        super().__init__()
        self.bidirectional_interaction = BidirectionalEvenOddInteraction(channels=channels, out_channels=out_channels)

    def forward(self, temporal_features):
        even_features, odd_features = split_even_odd(temporal_features)
        return self.bidirectional_interaction(even_features, odd_features)

class ProgressiveTemporalAggregation(nn.Module):
    """递归时间聚合、最终时间压缩和末步残差。"""

    def __init__(
        self, features=128, input_channels=16, length=12,
        dropout=0.1, max_blocks=2, checkpoint_blocks=True,
    ):
        super().__init__()
        if length < 1:
            raise ValueError('length must be >= 1')
        if input_channels < 1 or features < input_channels:
            raise ValueError('Require 1 <= input_channels <= features')
        if not isinstance(max_blocks, int) or isinstance(max_blocks, bool) or max_blocks < 0:
            raise ValueError('max_blocks must be a non-negative integer')
        self.input_length = length
        self.input_channels = input_channels
        self.checkpoint_blocks = checkpoint_blocks
        current_length = length
        current_channels = input_channels
        blocks = []
        while current_length > 3 and len(blocks) < max_blocks:
            next_channels = min(features, current_channels * 2)
            blocks.append(RecursiveTemporalAggregationLayer(channels=current_channels, out_channels=next_channels))
            current_length = (current_length + 1) // 2
            current_channels = next_channels
        self.recursive_temporal_layers = nn.ModuleList(blocks)
        self.final_temporal_aggregation = nn.Sequential(
            nn.Conv2d(current_channels, features, kernel_size=(1, current_length)),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.last_step_projection = nn.Conv2d(input_channels, features, kernel_size=1) if input_channels != features else nn.Identity()

    def forward(self, temporal_features):
        if temporal_features.shape[1] != self.input_channels:
            raise ValueError(f'Expected {self.input_channels} input channels, got {temporal_features.shape[1]}')
        if temporal_features.shape[-1] != self.input_length:
            raise ValueError(f'Expected time length {self.input_length}, got {temporal_features.shape[-1]}')
        last_time_step = self.last_step_projection(temporal_features[..., -1:])
        compressed_features = temporal_features
        if self.training and torch.is_grad_enabled() and self.checkpoint_blocks and (len(self.recursive_temporal_layers) > 0):

            def run_ifm_blocks(x):
                for block in self.recursive_temporal_layers:
                    x = block(x)
                return x
            compressed_features = checkpoint(run_ifm_blocks, compressed_features, use_reentrant=False)
        else:
            for block in self.recursive_temporal_layers:
                compressed_features = block(compressed_features)
        compressed_features = self.final_temporal_aggregation(compressed_features)
        return compressed_features + last_time_step

class TempoLS(nn.Module):
    """固定单层逐位置耦合的 TempoLS；输入 [B,D,N,T]，输出 [B,H,N,1]。"""

    def __init__(
        self,
        input_dim=3,
        num_nodes=170,
        input_len=12,
        output_len=12,
        dropout=0.1,
        latent_rank=32,
        checkpoint_temporal=False,
        checkpoint_spatial=False,
        spatial_interaction_channels=128,
        checkpoint_local=True,
    ):
        super().__init__()
        if input_len < 2:
            raise ValueError('input_len must be >= 2')
        channels = 128
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.input_len = input_len
        self.checkpoint_temporal = checkpoint_temporal
        self.checkpoint_spatial = checkpoint_spatial
        self.checkpoint_local = checkpoint_local
        if num_nodes in (170, 307, 358, 883):
            time_steps_per_day = 288
        elif num_nodes in (250, 266):
            time_steps_per_day = 48
        elif num_nodes > 200:
            time_steps_per_day = 96
        else:
            raise ValueError(f'Unsupported num_nodes={num_nodes}: cannot determine time_steps_per_day.')
        self.periodic_temporal_embedding = PeriodicTemporalEmbedding(time_steps_per_day, channels)
        fused_channels = channels * 2
        self.spatial_interaction = SpatialInteractionModule(
            channels=fused_channels,
            num_nodes=num_nodes,
            dropout=dropout,
            latent_rank=latent_rank,
            interaction_channels=spatial_interaction_channels,
        )
        self.parity_input_stage = ParityInputStage(input_dim=input_dim, channels=16)
        self.temporal_aggregation = ProgressiveTemporalAggregation(
            features=channels,
            input_channels=16,
            length=(input_len + 1) // 2,
            dropout=dropout,
            max_blocks=2,
            checkpoint_blocks=checkpoint_local and not checkpoint_temporal,
        )
        self.global_skip_projection = nn.Sequential(
            nn.Conv2d(fused_channels, fused_channels, kernel_size=1),

        )
        self.global_skip_gate_projection = nn.Sequential(
            nn.Conv2d(fused_channels, fused_channels, kernel_size=1),

        )
        self.forecasting_head = nn.Conv2d(fused_channels, output_len, 1)

    def param_num(self):
        return sum((p.numel() for p in self.parameters()))

    def forward(self, history_data):
        if (
            history_data.ndim != 4
            or history_data.shape[1] != self.input_dim
            or self.input_dim < 3
            or history_data.shape[2] != self.num_nodes
            or history_data.shape[-1] != self.input_len
        ):
            raise ValueError('history_data must be [B,input_dim,num_nodes,input_len], with flow/time-of-day/day-of-week channels')
        if self.training and torch.is_grad_enabled() and self.checkpoint_temporal:
            compressed_temporal_features = checkpoint(self._encode_temporal, history_data, use_reentrant=False)
        elif self.training and torch.is_grad_enabled() and self.checkpoint_local:
            parity_features = checkpoint(self.parity_input_stage, history_data, use_reentrant=False)
            compressed_temporal_features = self.temporal_aggregation(parity_features)
        else:
            compressed_temporal_features = self._encode_temporal(history_data)
        temporal_features = self.periodic_temporal_embedding(history_data.permute(0, 3, 2, 1))
        spatiotemporal_features = torch.cat((compressed_temporal_features, temporal_features), dim=1)
        if self.training and torch.is_grad_enabled() and self.checkpoint_spatial:
            return checkpoint(self._forecast_from_features, spatiotemporal_features, use_reentrant=False)
        return self._forecast_from_features(spatiotemporal_features)

    def _encode_temporal(self, history_data):
        return self.temporal_aggregation(self.parity_input_stage(history_data))

    def _forecast_from_features(self, spatiotemporal_features):
        encoded_features = self.spatial_interaction(spatiotemporal_features)
        residual_features = self.global_skip_projection(spatiotemporal_features)
        residual_gate = torch.sigmoid(self.global_skip_gate_projection(spatiotemporal_features))
        fused_features = encoded_features + residual_features * residual_gate
        return self.forecasting_head(fused_features)

