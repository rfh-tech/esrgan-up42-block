import cv2
import numpy as np
import torch
import arch as arch

from blockutils.logging import get_logger

from blockutils.common import ensure_data_directories_exist

from blockutils.datapath import set_data_path, get_in_out_feature_names_and_paths
from blockutils.windows import WindowsUtil

from blockutils.blocks import ProcessingBlock


logger = get_logger(__name__)

class ESRGAN(ProcessingBlock):
    def __init__(self):
        super().__init__()

    def super_resolve(
        self, input_file_path: Union[str, Path], output_file_path: Union[str, Path]
    ) -> None:
        model_path = 'models/RRDB_ESRGAN_x4.pth'
        device = torch.device('cpu')

        model = arch.RRDBNet(3, 3, 64, 23, gc=32)
        model.load_state_dict(torch.load(model_path), strict=True)
        model.eval()
        model = model.to(device)

        img = cv2.imread(input_file_path, cv2.IMREAD_COLOR)
        img = img * 1.0 / 255
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_LR = img.unsqueeze(0)
        img_LR = img_LR.to(device)

        with torch.no_grad():
            output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round()
        cv2.imwrite(output_file_path, output)

    def process(self, input_fc: FeatureCollection) -> FeatureCollection:
        """
        Args:
            metadata: A GeoJSON FeatureCollection describing all input datasets

        Returns:
            A GeoJSON FeatureCollection describing all output datasets
        """

        ensure_data_directories_exist()

        results: List[Feature] = []
        for in_feature in input_fc.features:
            (
                in_feature_name,
                out_feature_name,
                in_feature_path,
                out_feature_path,
            ) = get_in_out_feature_names_and_paths(in_feature)

            logger.debug("Input file: %s", in_feature_name)
            logger.debug("Output file: %s", out_feature_name)

            self.super_resolve(in_feature_path, out_feature_path)

            out_feature = Feature(
                geometry=in_feature["geometry"], bbox=in_feature["bbox"]
            )
            out_feature["properties"] = self.get_metadata(in_feature)
            set_data_path(out_feature, out_feature_name)
            results.append(out_feature)

            logger.debug("File %s was super resolved", out_feature_name)
        logger.debug("DONE!")
        return FeatureCollection(results)

    @classmethod
    def get_metadata(cls, feature: Feature) -> dict:
        """
        Extracts metadata elements that need to be propagated to the output tif
        """
        prop_dict = feature["properties"]
        meta_dict = {
            k: v
            for k, v in prop_dict.items()
            if not (k.startswith("up42.") or k.startswith("custom."))
        }
        return meta_dict