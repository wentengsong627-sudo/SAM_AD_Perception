"""Cache only invariant encoder metadata; keep upstream sources unchanged."""
import torch


def cache_encoder_metadata(encoder):
    original=encoder._prepare_multilevel_features
    cache={}

    def prepare(srcs,masks,pos_embeds):
        # Masked inputs need frame-dependent valid ratios. Use the native path.
        if masks is None or masks[0] is not None:
            return original(srcs,masks,pos_embeds)
        assert len(srcs)==encoder.num_feature_levels
        key=(tuple(tuple(s.shape) for s in srcs),srcs[0].device)
        if key not in cache:
            shapes=torch.tensor([s.shape[-2:] for s in srcs],dtype=torch.long,device=srcs[0].device)
            starts=torch.cat((shapes.new_zeros((1,)),shapes.prod(1).cumsum(0)[:-1]))
            ratios=torch.ones((srcs[0].shape[0],encoder.num_feature_levels,2),device=srcs[0].device)
            cache.clear();cache[key]=(starts,ratios,shapes)
        features=[];positions=[]
        for lvl,(src,pos) in enumerate(zip(srcs,pos_embeds)):
            features.append(src.flatten(2).transpose(1,2))
            pos=pos.flatten(2).transpose(1,2)
            if encoder.level_embed is not None:pos=pos+encoder.level_embed[lvl].view(1,1,-1)
            positions.append(pos)
        starts,ratios,shapes=cache[key]
        return torch.cat(features,1),None,torch.cat(positions,1),starts,ratios,shapes

    encoder._prepare_multilevel_features=prepare
    return original
