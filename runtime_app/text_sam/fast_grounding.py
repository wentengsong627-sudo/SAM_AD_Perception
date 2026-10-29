"""The pinned HF 4.46 deformable-attention math with one metadata read per call.

The stock fallback repeatedly converts CUDA shape scalars to Python integers.
Keep those shapes on the host once per detector invocation. Weights, image
resolution and precision are unchanged. The optional ARM fused sampler changes
the reduction order; its floating-point tolerance is checked against the
reference implementation.
"""
import types
import torch
import torch.nn.functional as F


def enable_fast_shapes(model):
    from transformers.models.grounding_dino.modeling_grounding_dino import GroundingDinoMultiscaleDeformableAttention
    cache = {}
    import os
    sampler = None
    pass  # Board-only path omitted from this CUDA release.

    def forward(self, hidden_states, attention_mask=None, encoder_hidden_states=None,
                encoder_attention_mask=None, position_embeddings=None, reference_points=None,
                spatial_shapes=None, spatial_shapes_list=None, level_start_index=None,
                output_attentions=False):
        if position_embeddings is not None:
            hidden_states = hidden_states + position_embeddings
        batch_size, queries, _ = hidden_states.shape
        _, sequence, _ = encoder_hidden_states.shape
        key = spatial_shapes.data_ptr()
        if key not in cache:
            shapes = spatial_shapes.cpu().tolist()
            cache[key] = [(int(h), int(w)) for h, w in shapes]
        shapes = cache[key]
        if sum(h*w for h,w in shapes) != sequence:
            raise ValueError("Deformable attention spatial shapes do not match sequence length")
        value = self.value_proj(encoder_hidden_states)
        if attention_mask is not None:
            value = value.masked_fill(~attention_mask[..., None], 0.)
        value = value.view(batch_size, sequence, self.n_heads, self.d_model // self.n_heads)
        offsets = self.sampling_offsets(hidden_states).view(batch_size, queries, self.n_heads, self.n_levels, self.n_points, 2)
        weights = self.attention_weights(hidden_states).view(batch_size, queries, self.n_heads, self.n_levels*self.n_points)
        weights = F.softmax(weights, -1).view(batch_size, queries, self.n_heads, self.n_levels, self.n_points)
        original_weights = weights
        if reference_points.shape[-1] == 2:
            normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            locations = reference_points[:, :, None, :, None, :] + offsets / normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            locations = reference_points[:, :, None, :, None, :2] + offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * .5
        else:
            raise ValueError("Reference points need 2 or 4 coordinates")
        heads, channels = value.shape[-2:]
        if sampler is not None:
            fused = sampler(value, shapes, locations, weights)
            if fused is not None:
                return self.output_proj(fused), original_weights
        values = value.split([h*w for h,w in shapes], dim=1)
        grids = 2 * locations - 1
        samples = []
        for level, (h,w) in enumerate(shapes):
            level_value = values[level].flatten(2).transpose(1,2).reshape(batch_size*heads, channels, h, w)
            grid = grids[:, :, :, level].transpose(1,2).flatten(0,1)
            samples.append(F.grid_sample(level_value, grid, mode="bilinear", padding_mode="zeros", align_corners=False))
        weights = weights.transpose(1,2).reshape(batch_size*heads, 1, queries, self.n_levels*self.n_points)
        output = (torch.stack(samples, dim=-2).flatten(-2)*weights).sum(-1).view(batch_size, heads*channels, queries)
        return self.output_proj(output.transpose(1,2).contiguous()), original_weights

    count = 0
    for module in model.modules():
        if isinstance(module, GroundingDinoMultiscaleDeformableAttention):
            module.forward = types.MethodType(forward, module)
            count += 1
    return cache, count
