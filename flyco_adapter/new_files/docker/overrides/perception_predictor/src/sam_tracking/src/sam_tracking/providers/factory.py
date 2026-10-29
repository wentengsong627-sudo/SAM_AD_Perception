"""Select a provider without importing modern model code into ROS Noetic."""


def make_camera_provider(backend, runtime_params, endpoint):
    backend = str(backend).strip().upper()
    if backend in ("A", "D"):
        from .http_text_sam_provider import HTTPTextSAMProvider
        return HTTPTextSAMProvider(endpoint, backend)
    if backend not in ("NATIVE", "SAM2"):
        raise ValueError("segmentation_backend must be native, A or D")
    from .sam2_camera_provider import Sam2CameraProvider, EVFTextInitConfig, SAM2RuntimeConfig
    return Sam2CameraProvider(
        evf_text_init_config=EVFTextInitConfig(
            enabled=runtime_params.evf_text_init_enabled,
            evf_root=runtime_params.evf_root,
            model_version=runtime_params.evf_model_version,
            model_type=runtime_params.evf_model_type,
            precision=runtime_params.evf_precision,
            device=runtime_params.evf_device,
            image_size=runtime_params.evf_image_size,
            semantic_level=runtime_params.evf_semantic_level,
            min_mask_area=runtime_params.evf_min_mask_area,
            max_init_points=runtime_params.evf_max_init_points,
        ),
        sam2_runtime_config=SAM2RuntimeConfig(
            samurai_mode=runtime_params.sam2_samurai_mode,
            depth_score_weight=runtime_params.sam2_depth_score_weight,
            kf_score_weight=runtime_params.sam2_kf_score_weight,
            memory_bank_geo_score_threshold=runtime_params.sam2_memory_bank_geo_score_threshold,
        ),
    )
