ALL_ID_DATASET = [
     # far task
    'cifar10', 'cifar100', 'bird200', 'car196', 'food101', 'pet37', 'ImageNet', 'ImageNet_sketch',
    # near task
    'ImageNet10', 'ImageNet20', 
    # fine_grained task
    'cub100_ID', 'car98_ID', 'food50_ID', 'pet18_ID', 
    # explore the robustness
    'ImageNet_C_blur_defocus_blur', 'ImageNet_C_blur_glass_blur', 'ImageNet_C_blur_motion_blur', 'ImageNet_C_blur_zoom_blur', 
    'ImageNet_C_digital_contrast', 'ImageNet_C_digital_elastic_transform', 'ImageNet_C_digital_jpeg_compression', 'ImageNet_C_digital_pixelate',  
    'ImageNet_C_extra_gaussian_blur', 'ImageNet_C_extra_saturate', 'ImageNet_C_extra_spatter', 'ImageNet_C_extra_speckle_noise',  
    'ImageNet_C_noise_gaussian_noise', 'ImageNet_C_noise_impulse_noise', 'ImageNet_C_noise_shot_noise',  
    'ImageNet_C_weather_brightness', 'ImageNet_C_weather_fog', 'ImageNet_C_weather_frost', 'ImageNet_C_weather_snow']


ALL_OOD_TASK = [
    # main results
    'far', 'near', 'fine_grained',
    # general prompt (Limitation II in paper)
    'general',
    # below is the ablation studies for LLM prompts
    'fine_grained_irrelevant', 'fine_grained_dissimilar',
    'near_irrelevant', 'near_dissimilar',
    'far_irrelevant', 'far_dissimilar']


ALL_LLM = [
    'gpt-3.5-turbo', 'gpt-3.5-turbo-16k', 'gpt-4', 'gpt-4-1106-preview', 'gpt-4-0125-preview',
    'Claude-2', 'Claude-2-100k', 'Claude-3-Haiku',
    # NOTE: Llama's responses may not adhere strictly to the predefined JSON format, thus we manually input the output of llama into JSON in the ablation experiment.
    'Llama-2-7b', 'Llama-2-13b', 'Llama-2-70b',
    'Mixtral-8x7B-Chat', 'Gemma-7b-FW', 'Gemini-Pro']


dataset_mappings = {
    # far ood
    'bird200': ['iNaturalist', 'SUN', 'places365', 'dtd'],
    'car196': ['iNaturalist', 'SUN', 'places365', 'dtd'],
    'food101': ['iNaturalist', 'SUN', 'places365', 'dtd'],
    'pet37': ['iNaturalist', 'SUN', 'places365', 'dtd'],
    'ImageNet_sketch': ['iNaturalist', 'SUN', 'places365', 'dtd'],
    'cifar10': ['svhn', 'lsun', 'dtd', 'places365'],
    'cifar100': ['svhn', 'lsun', 'dtd', 'places365'],
    # near ood
    'ImageNet10': ['ImageNet20'],
    'ImageNet20': ['ImageNet10'],
    # fine-grained ood
    'cub100_ID': ['cub100_OOD'],
    'car98_ID': ['car98_OOD'],
    'food50_ID': ['food50_OOD'],
    'pet18_ID': ['pet18_OOD'],
}