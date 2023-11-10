from collections import defaultdict

import yaml

from savant.deepstream.drawfunc import NvDsDrawFunc
from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.utils.artist import Artist, Position

from .utils import RandColorIterator


class Overlay(NvDsDrawFunc):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.obj_colors = defaultdict(lambda: next(RandColorIterator()))
        self.config = {}

        # Load configuration from a YAML file if provided
        if hasattr(self, "config_path") and self.config_path:
            with open(self.config_path, "rt") as f:
                self.config = yaml.safe_load(f)

        # Set debug mode based on configuration
        self.debug = self.config.get("debug", True)

        # Extract ROI and ALPR area configurations from sources
        self.roi_cfgs = {}
        self.alpr_cfgs = {}
        if "sources" in self.config:
            for source_id, source_cfg in self.config["sources"].items():
                if "roi_area" in source_cfg:
                    self.roi_cfgs[source_id] = source_cfg["roi_area"]

                if "alpr_area" in source_cfg:
                    self.alpr_cfgs[source_id] = source_cfg["alpr_area"]

    def draw_on_frame(self, frame_meta: NvDsFrameMeta, artist: Artist):
        if not self.debug:
            return

        source_id = frame_meta.source_id

        roi_area = None
        if source_id in self.roi_cfgs:
            roi_area = self.roi_cfgs[source_id]

        alpr_area = None
        if source_id in self.alpr_cfgs:
            alpr_area = self.alpr_cfgs[source_id]

        # Determine ROI and ALPR areas for the current frame's source
        for i, obj_meta in enumerate(frame_meta.objects):
            # Process non-primary objects inside ROI
            is_not_primary = not obj_meta.is_primary
            is_inside_roi = obj_meta.get_attr_meta("global_roi", "is_inside")
            is_inside_roi = (
                is_inside_roi.value if is_inside_roi is not None else False
            )
            is_inside_roi = True

            if is_not_primary and is_inside_roi:
                is_inside_alpr_roi = obj_meta.get_attr_meta(
                    "alpr_roi", "is_inside"
                )
                is_inside_alpr_roi = (
                    is_inside_alpr_roi.value
                    if is_inside_alpr_roi is not None
                    else False
                )
                is_inside_alpr_roi = True

                # Draw bounding box
                if obj_meta.label == "vehicle" or (
                    obj_meta.label == "lp" and is_inside_alpr_roi
                ):
                    color = self.obj_colors[
                        (frame_meta.source_id, obj_meta.track_id)
                    ]
                    xc, yc, w, h = obj_meta.bbox.as_xcycwh_int()
                    if (
                        xc + w / 2 < 1920
                        and yc + h / 2 < 1080
                        and w > 0
                        and h > 0
                    ):
                        artist.add_bbox(
                            obj_meta.bbox, border_width=4, border_color=color
                        )

                    # Draw license plate
                    if obj_meta.label == "lp":
                        lp_result = obj_meta.get_attr_meta(
                            "LPRecognizer", "lp_result"
                        )
                        lp_result = (
                            lp_result.value if lp_result is not None else ""
                        )

                        artist.add_text(
                            lp_result,
                            (int(obj_meta.bbox.xc), int(obj_meta.bbox.yc)),
                            font_scale=2.0,
                            font_thickness=3,
                            bg_color=(0, 0, 0, 0),
                            anchor_point_type=Position.LEFT_TOP,
                        )

        # Draw ROI area
        roi_area = roi_area if roi_area is not None else []
        roi_area = [roi_area[i : i + 2] for i in range(0, len(roi_area), 2)]
        if roi_area:
            x, y = roi_area[0]
            artist.add_polygon(
                vertices=roi_area,
                line_width=3,
                line_color=(255, 255, 255, 255),
            )
            artist.add_text(
                "Global RoI",
                (x, y),
                anchor_point_type=Position.LEFT_TOP,
            )

        # Draw ALPR area
        alpr_area = alpr_area if alpr_area is not None else []
        if alpr_area:
            alpr_area = [
                alpr_area[i : i + 2] for i in range(0, len(alpr_area), 2)
            ]
            x, y = alpr_area[0]
            artist.add_polygon(
                vertices=alpr_area,
                line_width=3,
                line_color=(255, 255, 255, 255),
            )
            artist.add_text(
                "ALPR RoI",
                (x, y),
                anchor_point_type=Position.LEFT_TOP,
            )