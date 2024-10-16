"""
OpenCLIP model properties

This file contains properties for OpenCLIP models. It is intended to be used in conjunction with the model registry
and should not be used in isolation.
"""

from typing import Dict, List, Optional

from marqo_commons.model_registry.model_properties_object import (
    Modality,
    ModelProperties,
    ModelType,
    VectorNumericType,
)
from marqo_commons.model_registry.utils import convert_model_properties_to_dict


class OpenClipModelProperties(ModelProperties):
    vector_numeric_type: VectorNumericType = VectorNumericType.float32
    default_memory_size: float = 1.0
    modality: List[Modality] = [Modality.text, Modality.image]
    type: ModelType = ModelType.open_clip
    pretrained: Optional[str] = None
    notes: str = ""

    @classmethod
    def get_all_model_properties_objects(cls) -> Dict[str, "OpenClipModelProperties"]:
        # use this link to find all the model_configs
        # https://github.com/mlfoundations/open_clip/tree/main/src/open_clip/model_configs
        return {
            "open_clip/RN50/openai": OpenClipModelProperties(
                name="open_clip/RN50/openai",
                dimensions=1024,
                notes="open_clip models",
                pretrained="openai",
            ),
            "open_clip/RN50/yfcc15m": OpenClipModelProperties(
                name="open_clip/RN50/yfcc15m",
                dimensions=1024,
                notes="open_clip models",
                pretrained="yfcc15m",
            ),
            "open_clip/RN50/cc12m": OpenClipModelProperties(
                name="open_clip/RN50/cc12m",
                dimensions=1024,
                notes="open_clip models",
                pretrained="cc12m",
            ),
            "open_clip/RN50-quickgelu/openai": OpenClipModelProperties(
                name="open_clip/RN50-quickgelu/openai",
                dimensions=1024,
                notes="open_clip models",
                pretrained="openai",
            ),
            "open_clip/RN50-quickgelu/yfcc15m": OpenClipModelProperties(
                name="open_clip/RN50-quickgelu/yfcc15m",
                dimensions=1024,
                notes="open_clip models",
                pretrained="yfcc15m",
            ),
            "open_clip/RN50-quickgelu/cc12m": OpenClipModelProperties(
                name="open_clip/RN50-quickgelu/cc12m",
                dimensions=1024,
                notes="open_clip models",
                pretrained="cc12m",
            ),
            "open_clip/RN101/openai": OpenClipModelProperties(
                name="open_clip/RN101/openai",
                dimensions=512,
                notes="open_clip models",
                pretrained="openai",
            ),
            "open_clip/RN101/yfcc15m": OpenClipModelProperties(
                name="open_clip/RN101/yfcc15m",
                dimensions=512,
                notes="open_clip models",
                pretrained="yfcc15m",
            ),
            "open_clip/RN101-quickgelu/openai": OpenClipModelProperties(
                name="open_clip/RN101-quickgelu/openai",
                dimensions=512,
                notes="open_clip models",
                pretrained="openai",
            ),
            "open_clip/RN101-quickgelu/yfcc15m": OpenClipModelProperties(
                name="open_clip/RN101-quickgelu/yfcc15m",
                dimensions=512,
                notes="open_clip models",
                pretrained="yfcc15m",
            ),
            "open_clip/RN50x4/openai": OpenClipModelProperties(
                name="open_clip/RN50x4/openai",
                dimensions=640,
                notes="open_clip models",
                pretrained="openai",
            ),
            "open_clip/RN50x16/openai": OpenClipModelProperties(
                name="open_clip/RN50x16/openai",
                dimensions=768,
                notes="open_clip models",
                pretrained="openai",
            ),
            "open_clip/RN50x64/openai": OpenClipModelProperties(
                name="open_clip/RN50x64/openai",
                dimensions=1024,
                notes="open_clip models",
                pretrained="openai",
            ),
            "open_clip/ViT-B-32/openai": OpenClipModelProperties(
                name="open_clip/ViT-B-32/openai",
                dimensions=512,
                notes="open_clip models",
                pretrained="openai",
            ),
            "open_clip/ViT-B-32/laion400m_e31": OpenClipModelProperties(
                name="open_clip/ViT-B-32/laion400m_e31",
                dimensions=512,
                notes="open_clip models",
                pretrained="laion400m_e31",
            ),
            "open_clip/ViT-B-32/laion400m_e32": OpenClipModelProperties(
                name="open_clip/ViT-B-32/laion400m_e32",
                dimensions=512,
                notes="open_clip models",
                pretrained="laion400m_e32",
            ),
            "open_clip/ViT-B-32/laion2b_e16": OpenClipModelProperties(
                name="open_clip/ViT-B-32/laion2b_e16",
                dimensions=512,
                notes="open_clip models",
                pretrained="laion2b_e16",
            ),
            "open_clip/ViT-B-32/laion2b_s34b_b79k": OpenClipModelProperties(
                name="open_clip/ViT-B-32/laion2b_s34b_b79k",
                dimensions=512,
                notes="open_clip models",
                pretrained="laion2b_s34b_b79k",
            ),
            "open_clip/ViT-B-32-quickgelu/openai": OpenClipModelProperties(
                name="open_clip/ViT-B-32-quickgelu/openai",
                dimensions=512,
                notes="open_clip models",
                pretrained="openai",
            ),
            "open_clip/ViT-B-32-quickgelu/laion400m_e31": OpenClipModelProperties(
                name="open_clip/ViT-B-32-quickgelu/laion400m_e31",
                dimensions=512,
                notes="open_clip models",
                pretrained="laion400m_e31",
            ),
            "open_clip/ViT-B-32-quickgelu/laion400m_e32": OpenClipModelProperties(
                name="open_clip/ViT-B-32-quickgelu/laion400m_e32",
                dimensions=512,
                notes="open_clip models",
                pretrained="laion400m_e32",
            ),
            "open_clip/ViT-B-16/openai": OpenClipModelProperties(
                name="open_clip/ViT-B-16/openai",
                dimensions=512,
                notes="open_clip models",
                pretrained="openai",
            ),
            "open_clip/ViT-B-16/laion400m_e31": OpenClipModelProperties(
                name="open_clip/ViT-B-16/laion400m_e31",
                dimensions=512,
                notes="open_clip models",
                pretrained="laion400m_e31",
            ),
            "open_clip/ViT-B-16/laion400m_e32": OpenClipModelProperties(
                name="open_clip/ViT-B-16/laion400m_e32",
                dimensions=512,
                notes="open_clip models",
                pretrained="laion400m_e32",
            ),
            "open_clip/ViT-B-16/laion2b_s34b_b88k": OpenClipModelProperties(
                name="open_clip/ViT-B-16/laion2b_s34b_b88k",
                dimensions=512,
                notes="open_clip models",
                pretrained="laion2b_s34b_b88k",
            ),
            "open_clip/ViT-B-16-plus-240/laion400m_e31": OpenClipModelProperties(
                name="open_clip/ViT-B-16-plus-240/laion400m_e31",
                dimensions=640,
                notes="open_clip models",
                pretrained="laion400m_e31",
            ),
            "open_clip/ViT-B-16-plus-240/laion400m_e32": OpenClipModelProperties(
                name="open_clip/ViT-B-16-plus-240/laion400m_e32",
                dimensions=640,
                notes="open_clip models",
                pretrained="laion400m_e32",
            ),
            "open_clip/ViT-L-14/openai": OpenClipModelProperties(
                name="open_clip/ViT-L-14/openai",
                dimensions=768,
                notes="open_clip models",
                pretrained="openai",
            ),
            "open_clip/ViT-L-14/laion400m_e31": OpenClipModelProperties(
                name="open_clip/ViT-L-14/laion400m_e31",
                dimensions=768,
                notes="open_clip models",
                pretrained="laion400m_e31",
            ),
            "open_clip/ViT-L-14/laion400m_e32": OpenClipModelProperties(
                name="open_clip/ViT-L-14/laion400m_e32",
                dimensions=768,
                notes="open_clip models",
                pretrained="laion400m_e32",
            ),
            "open_clip/ViT-L-14/laion2b_s32b_b82k": OpenClipModelProperties(
                name="open_clip/ViT-L-14/laion2b_s32b_b82k",
                dimensions=768,
                notes="open_clip models",
                pretrained="laion2b_s32b_b82k",
            ),
            "open_clip/ViT-L-14-336/openai": OpenClipModelProperties(
                name="open_clip/ViT-L-14-336/openai",
                dimensions=768,
                notes="open_clip models",
                pretrained="openai",
            ),
            "open_clip/ViT-H-14/laion2b_s32b_b79k": OpenClipModelProperties(
                name="open_clip/ViT-H-14/laion2b_s32b_b79k",
                dimensions=1024,
                notes="open_clip models",
                type=ModelType.open_clip,
                pretrained="laion2b_s32b_b79k",
            ),
            "open_clip/ViT-g-14/laion2b_s12b_b42k": OpenClipModelProperties(
                name="open_clip/ViT-g-14/laion2b_s12b_b42k",
                dimensions=1024,
                notes="open_clip models",
                pretrained="laion2b_s12b_b42k",
            ),
            "open_clip/ViT-g-14/laion2b_s34b_b88k": OpenClipModelProperties(
                name="open_clip/ViT-g-14/laion2b_s34b_b88k",
                dimensions=1024,
                notes="open_clip models",
                pretrained="laion2b_s34b_b88k",
            ),
            "open_clip/ViT-bigG-14/laion2b_s39b_b160k": OpenClipModelProperties(
                name="open_clip/ViT-bigG-14/laion2b_s39b_b160k",
                dimensions=1280,
                notes="open_clip models",
                pretrained="laion2b_s39b_b160k",
            ),
            "open_clip/roberta-ViT-B-32/laion2b_s12b_b32k": OpenClipModelProperties(
                name="open_clip/roberta-ViT-B-32/laion2b_s12b_b32k",
                dimensions=512,
                notes="open_clip models",
                pretrained="laion2b_s12b_b32k",
            ),
            "open_clip/xlm-roberta-base-ViT-B-32/laion5b_s13b_b90k": OpenClipModelProperties(
                name="open_clip/xlm-roberta-base-ViT-B-32/laion5b_s13b_b90k",
                dimensions=512,
                notes="open_clip models",
                pretrained="laion5b_s13b_b90k",
            ),
            "open_clip/xlm-roberta-large-ViT-H-14/frozen_laion5b_s13b_b90k": OpenClipModelProperties(
                name="open_clip/xlm-roberta-large-ViT-H-14/frozen_laion5b_s13b_b90k",
                dimensions=1024,
                notes="open_clip models",
                pretrained="frozen_laion5b_s13b_b90k",
            ),
            "open_clip/convnext_base/laion400m_s13b_b51k": OpenClipModelProperties(
                name="open_clip/convnext_base/laion400m_s13b_b51k",
                dimensions=512,
                notes="open_clip models",
                pretrained="laion400m_s13b_b51k",
            ),
            "open_clip/convnext_base_w/laion2b_s13b_b82k": OpenClipModelProperties(
                name="open_clip/convnext_base_w/laion2b_s13b_b82k",
                dimensions=640,
                notes="open_clip models",
                pretrained="laion2b_s13b_b82k",
            ),
            "open_clip/convnext_base_w/laion2b_s13b_b82k_augreg": OpenClipModelProperties(
                name="open_clip/convnext_base_w/laion2b_s13b_b82k_augreg",
                dimensions=640,
                notes="open_clip models",
                pretrained="laion2b_s13b_b82k_augreg",
            ),
            "open_clip/convnext_base_w/laion_aesthetic_s13b_b82k": OpenClipModelProperties(
                name="open_clip/convnext_base_w/laion_aesthetic_s13b_b82k",
                dimensions=640,
                notes="open_clip models",
                pretrained="laion_aesthetic_s13b_b82k",
            ),
            "open_clip/convnext_base_w_320/laion_aesthetic_s13b_b82k": OpenClipModelProperties(
                name="open_clip/convnext_base_w_320/laion_aesthetic_s13b_b82k",
                dimensions=640,
                notes="open_clip models",
                pretrained="laion_aesthetic_s13b_b82k",
            ),
            "open_clip/convnext_base_w_320/laion_aesthetic_s13b_b82k_augreg": OpenClipModelProperties(
                name="open_clip/convnext_base_w_320/laion_aesthetic_s13b_b82k_augreg",
                dimensions=640,
                notes="open_clip models",
                pretrained="laion_aesthetic_s13b_b82k_augreg",
            ),
            "open_clip/convnext_large_d/laion2b_s26b_b102k_augreg": OpenClipModelProperties(
                name="open_clip/convnext_large_d/laion2b_s26b_b102k_augreg",
                dimensions=768,
                notes="open_clip models",
                type=ModelType.open_clip,
                pretrained="laion2b_s26b_b102k_augreg",
            ),
            "open_clip/convnext_large_d_320/laion2b_s29b_b131k_ft": OpenClipModelProperties(
                name="open_clip/convnext_large_d_320/laion2b_s29b_b131k_ft",
                dimensions=768,
                notes="open_clip models",
                pretrained="laion2b_s29b_b131k_ft",
            ),
            "open_clip/convnext_large_d_320/laion2b_s29b_b131k_ft_soup": OpenClipModelProperties(
                name="open_clip/convnext_large_d_320/laion2b_s29b_b131k_ft_soup",
                dimensions=768,
                notes="open_clip models",
                pretrained="laion2b_s29b_b131k_ft_soup",
            ),
            "open_clip/convnext_xxlarge/laion2b_s34b_b82k_augreg": OpenClipModelProperties(
                name="open_clip/convnext_xxlarge/laion2b_s34b_b82k_augreg",
                dimensions=1024,
                notes="open_clip models",
                pretrained="laion2b_s34b_b82k_augreg",
            ),
            "open_clip/convnext_xxlarge/laion2b_s34b_b82k_augreg_rewind": OpenClipModelProperties(
                name="open_clip/convnext_xxlarge/laion2b_s34b_b82k_augreg_rewind",
                dimensions=1024,
                notes="open_clip models",
                pretrained="laion2b_s34b_b82k_augreg_rewind",
            ),
            "open_clip/convnext_xxlarge/laion2b_s34b_b82k_augreg_soup": OpenClipModelProperties(
                name="open_clip/convnext_xxlarge/laion2b_s34b_b82k_augreg_soup",
                dimensions=1024,
                notes="open_clip models",
                pretrained="laion2b_s34b_b82k_augreg_soup",
            ),
            "open_clip/coca_ViT-B-32/laion2b_s13b_b90k": OpenClipModelProperties(
                name="open_clip/coca_ViT-B-32/laion2b_s13b_b90k",
                dimensions=512,
                notes="open_clip models",
                pretrained="laion2b_s13b_b90k",
            ),
            "open_clip/coca_ViT-B-32/mscoco_finetuned_laion2b_s13b_b90k": OpenClipModelProperties(
                name="open_clip/coca_ViT-B-32/mscoco_finetuned_laion2b_s13b_b90k",
                dimensions=512,
                notes="open_clip models",
                pretrained="mscoco_finetuned_laion2b_s13b_b90k",
            ),
            "open_clip/coca_ViT-L-14/laion2b_s13b_b90k": OpenClipModelProperties(
                name="open_clip/coca_ViT-L-14/laion2b_s13b_b90k",
                dimensions=768,
                notes="open_clip models",
                pretrained="laion2b_s13b_b90k",
            ),
            "open_clip/coca_ViT-L-14/mscoco_finetuned_laion2b_s13b_b90k": OpenClipModelProperties(
                name="open_clip/coca_ViT-L-14/mscoco_finetuned_laion2b_s13b_b90k",
                dimensions=768,
                notes="open_clip models",
                pretrained="mscoco_finetuned_laion2b_s13b_b90k",
            ),
            # New models as of Marqo 2.7.0
            # Added for: Open CLIP 2.24.0
            "open_clip/ViT-SO400M-14-SigLIP-384/webli": OpenClipModelProperties(
                name="open_clip/ViT-SO400M-14-SigLIP-384/webli",
                dimensions=1152,
                notes="open_clip model: ViT-SO400M-14-SigLIP-384/webli",
                pretrained="webli",
            ),
            "open_clip/ViT-H-14-378-quickgelu/dfn5b": OpenClipModelProperties(
                name="open_clip/ViT-H-14-378-quickgelu/dfn5b",
                dimensions=1024,
                notes="open_clip model: ViT-H-14-378-quickgelu/dfn5b",
                pretrained="dfn5b",
            ),
            "open_clip/ViT-L-16-SigLIP-384/webli": OpenClipModelProperties(
                name="open_clip/ViT-L-16-SigLIP-384/webli",
                dimensions=1024,
                notes="open_clip model: ViT-L-16-SigLIP-384/webli",
                pretrained="webli",
            ),
            "open_clip/ViT-H-14-quickgelu/dfn5b": OpenClipModelProperties(
                name="open_clip/ViT-H-14-quickgelu/dfn5b",
                dimensions=1024,
                notes="open_clip model: ViT-H-14-quickgelu/dfn5b",
                pretrained="dfn5b",
            ),
            "open_clip/ViT-L-16-SigLIP-256/webli": OpenClipModelProperties(
                name="open_clip/ViT-L-16-SigLIP-256/webli",
                dimensions=1024,
                notes="open_clip model: ViT-L-16-SigLIP-256/webli",
                pretrained="webli",
            ),
            "open_clip/ViT-B-16-SigLIP-512/webli": OpenClipModelProperties(
                name="open_clip/ViT-B-16-SigLIP-512/webli",
                dimensions=768,
                notes="open_clip model: ViT-B-16-SigLIP-512/webli",
                pretrained="webli",
            ),
            "open_clip/ViT-B-16-SigLIP-384/webli": OpenClipModelProperties(
                name="open_clip/ViT-B-16-SigLIP-384/webli",
                dimensions=768,
                notes="open_clip model: ViT-B-16-SigLIP-384/webli",
                pretrained="webli",
            ),
            "open_clip/ViT-B-16-SigLIP-256/webli": OpenClipModelProperties(
                name="open_clip/ViT-B-16-SigLIP-256/webli",
                dimensions=768,
                notes="open_clip model: ViT-B-16-SigLIP-256/webli",
                pretrained="webli",
            ),
            "open_clip/ViT-B-16-SigLIP/webli": OpenClipModelProperties(
                name="open_clip/ViT-B-16-SigLIP/webli",
                dimensions=768,
                notes="open_clip model: ViT-B-16-SigLIP/webli",
                pretrained="webli",
            ),
            "open_clip/ViT-L-14-quickgelu/dfn2b": OpenClipModelProperties(
                name="open_clip/ViT-L-14-quickgelu/dfn2b",
                dimensions=768,
                notes="open_clip model: ViT-L-14-quickgelu/dfn2b",
                pretrained="dfn2b",
            ),
            "open_clip/EVA02-L-14-336/merged2b_s6b_b61k": OpenClipModelProperties(
                name="open_clip/EVA02-L-14-336/merged2b_s6b_b61k",
                dimensions=768,
                notes="open_clip model: EVA02-L-14-336/merged2b_s6b_b61k",
                pretrained="merged2b_s6b_b61k",
            ),
            "open_clip/EVA02-B-16/merged2b_s8b_b131k": OpenClipModelProperties(
                name="open_clip/EVA02-B-16/merged2b_s8b_b131k",
                dimensions=512,
                notes="open_clip model: EVA02-B-16/merged2b_s8b_b131k",
                pretrained="merged2b_s8b_b131k",
            ),
            "open_clip/EVA02-L-14/merged2b_s4b_b131k": OpenClipModelProperties(
                name="open_clip/EVA02-L-14/merged2b_s4b_b131k",
                dimensions=768,
                notes="open_clip model: EVA02-L-14/merged2b_s4b_b131k",
                pretrained="merged2b_s4b_b131k",
            ),
            "open_clip/ViT-B-16-quickgelu/metaclip_fullcc": OpenClipModelProperties(
                name="open_clip/ViT-B-16-quickgelu/metaclip_fullcc",
                dimensions=512,
                notes="open_clip model: ViT-B-16-quickgelu/metaclip_fullcc",
                pretrained="metaclip_fullcc",
            ),
            "open_clip/ViT-L-14-CLIPA-336/datacomp1b": OpenClipModelProperties(
                name="open_clip/ViT-L-14-CLIPA-336/datacomp1b",
                dimensions=768,
                notes="open_clip model: ViT-L-14-CLIPA-336/datacomp1b",
                pretrained="datacomp1b",
            ),
            "open_clip/ViT-B-32-256/datacomp_s34b_b86k": OpenClipModelProperties(
                name="open_clip/ViT-B-32-256/datacomp_s34b_b86k",
                dimensions=512,
                notes="open_clip model: ViT-B-32-256/datacomp_s34b_b86k",
                pretrained="datacomp_s34b_b86k",
            ),
            "Marqo/marqo-fashionCLIP": OpenClipModelProperties(
                name="hf-hub:Marqo/marqo-fashionCLIP",
                dimensions=512,
                note="Marqo's fashionCLIP model",
            ),
            "Marqo/marqo-fashionSigLIP": OpenClipModelProperties(
                name="hf-hub:Marqo/marqo-fashionSigLIP",
                dimensions=768,
                note="Marqo's fashionSigLIP model",
            ),
            "visheratin/nllb-clip-base-siglip": OpenClipModelProperties(
                name="hf-hub:visheratin/nllb-clip-base-siglip",
                dimensions=768,
                note="A multilingual CLIP model",
            ),
            "visheratin/nllb-siglip-mrl-base": OpenClipModelProperties(
                name="hf-hub:visheratin/nllb-siglip-mrl-base",
                dimensions=768,
                note="A multilingual CLIP model",
            ),
            "visheratin/nllb-clip-large-siglip": OpenClipModelProperties(
                name="hf-hub:visheratin/nllb-clip-large-siglip",
                dimensions=1152,
                note="A multilingual CLIP model",
            ),
            "visheratin/nllb-siglip-mrl-large": OpenClipModelProperties(
                name="hf-hub:visheratin/nllb-siglip-mrl-large",
                dimensions=1152,
                note="A multilingual CLIP model",
            ),
        }


@convert_model_properties_to_dict
def _get_open_clip_properties() -> Dict:
    return OpenClipModelProperties.get_all_model_properties_objects()
