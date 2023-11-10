import time
from collections import defaultdict
from typing import List

import yaml
from savant_rs.primitives.geometry import Point, PolygonalArea

from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.deepstream.pyfunc import NvDsPyFuncPlugin
from savant.gstreamer import Gst

from .utils import VehicleLPTracker


class ConditionalDetectorSkip(NvDsPyFuncPlugin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config = {}
        if self.config_path:
            with open(self.config_path, "r", encoding="utf8") as stream:
                self.config = yaml.safe_load(stream)

    def process_frame(self, buffer: Gst.Buffer, frame_meta: NvDsFrameMeta):
        primary_meta_object = None
        for obj_meta in frame_meta.objects:
            if obj_meta.is_primary:
                primary_meta_object = obj_meta
                break

        # if the boundary lines are not configured for this source
        # then disable detector inference entirely by removing the primary object
        # Note:
        # In order to enable use cases such as conditional inference skip
        # or user-defined ROI, Savant configures all Deepstream models to run
        # in 'secondary' mode and inserts a primary 'frame' object into the DS meta
        if (
            primary_meta_object is not None
            and "sources" in self.config
            and frame_meta.source_id not in self.config["sources"]
        ):
            frame_meta.remove_obj_meta(primary_meta_object)


class ObjectDetectorSkip(NvDsPyFuncPlugin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config = {}

        # Load the configuration from a YAML file if provided
        if hasattr(self, "config_path") and self.config_path:
            with open(self.config_path, "r", encoding="utf8") as stream:
                self.config = yaml.safe_load(stream)

        # Initialize configurations
        self.initialize_configurations()

    def initialize_configurations(self):
        # Initialize Region of Interest (ROI) configurations
        self.roi_cfgs = {}
        sources = self.config.get("sources", {})
        for source_id, source_cfg in sources.items():
            self.roi_cfgs[source_id] = self.extract_roi_config(
                source_id, source_cfg
            )

    def extract_roi_config(self, source_id, source_cfg):
        # Extract and store ROI configurations for each source
        roi_area = None
        if "roi_area" in source_cfg:
            area = source_cfg["roi_area"]
            if area:
                assert len(area) % 2 == 0
                vetices = []
                for i in range(0, len(area), 2):
                    vetices.append(Point(area[i], area[i + 1]))

                roi_area = PolygonalArea(vetices)
        if roi_area is None:
            print(
                f"WARNING: RoI area is None, entire frame would be used. Source {source_id}"
            )
        return roi_area

    def process_frame(self, buffer: Gst.Buffer, frame_meta: NvDsFrameMeta):
        source_id = frame_meta.source_id
        if "sources" in self.config and source_id in self.config["sources"]:
            # if objects are outside of the RoI
            obj_metas = list(frame_meta.objects)
            for obj_meta in obj_metas:
                if not obj_meta.is_primary:
                    point = Point(obj_meta.bbox.xc, obj_meta.bbox.yc)

                    is_roi_area_none = self.roi_cfgs.get(source_id) is None
                    if not is_roi_area_none:
                        is_inside_roi_area = self.roi_cfgs[source_id].contains(
                            point
                        )
                    else:
                        is_inside_roi_area = True

                    if is_inside_roi_area:
                        obj_meta.add_attr_meta("global_roi", "is_inside", True)
                    else:
                        frame_meta.remove_obj_meta(obj_meta)


class LicensePlateDetectorSkip(NvDsPyFuncPlugin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config = {}

        # Load the configuration from a YAML file if provided
        if hasattr(self, "config_path") and self.config_path:
            with open(self.config_path, "r", encoding="utf8") as stream:
                self.config = yaml.safe_load(stream)

        # Initialize configurations
        self.initialize_configurations()

    def initialize_configurations(self):
        # Initialize ALPR Region of Interest (ROI) configurations
        self.roi_cfgs = {}
        self.alpr_cfgs = {}
        sources = self.config.get("sources", {})
        for source_id, source_cfg in sources.items():
            self.roi_cfgs[source_id] = self.extract_roi_config(
                source_id, source_cfg
            )
            self.alpr_cfgs[source_id] = self.extract_alpr_config(
                source_id, source_cfg
            )

    def extract_roi_config(self, source_id, source_cfg):
        # Extract and store ROI configurations for each source
        roi_area = None
        if "roi_area" in source_cfg:
            area = source_cfg["roi_area"]
            if area:
                assert len(area) % 2 == 0
                vetices = []
                for i in range(0, len(area), 2):
                    vetices.append(Point(area[i], area[i + 1]))

                roi_area = PolygonalArea(vetices)
        if roi_area is None:
            print(
                f"WARNING: RoI area is None, entire frame would be used. Source {source_id}"
            )
        return roi_area

    def extract_alpr_config(self, source_id, source_cfg):
        # Extract and store ALPR ROI configurations for each source
        alpr_area = None
        if "alpr_area" in source_cfg:
            area = source_cfg["alpr_area"]
            if area:
                assert len(area) % 2 == 0
                vetices = []
                for i in range(0, len(area), 2):
                    vetices.append(Point(area[i], area[i + 1]))

                alpr_area = PolygonalArea(vetices)
        if alpr_area is None:
            print(f"WARNING: ALPR area is None. Source {source_id}")

        return alpr_area

    def process_frame(self, buffer: Gst.Buffer, frame_meta: NvDsFrameMeta):
        source_id = frame_meta.source_id
        if "sources" in self.config and source_id in self.config["sources"]:
            # if objects are outside of the License Plate RoI
            for obj_meta in frame_meta.objects:
                if not obj_meta.is_primary:
                    point = Point(obj_meta.bbox.xc, obj_meta.bbox.yc)

                    is_roi_area_none = self.roi_cfgs.get(source_id) is None
                    if not is_roi_area_none:
                        is_inside_roi_area = self.roi_cfgs[source_id].contains(
                            point
                        )
                    else:
                        is_inside_roi_area = True

                    if is_inside_roi_area:
                        obj_meta.add_attr_meta("global_roi", "is_inside", True)

                    if obj_meta.label == "lp":
                        is_alpr_area_none = (
                            self.alpr_cfgs.get(source_id) is None
                        )
                        if not is_alpr_area_none:
                            is_inside_alpr_area = self.alpr_cfgs[
                                source_id
                            ].contains(point)
                        else:
                            is_inside_alpr_area = True

                        if is_inside_alpr_area:
                            obj_meta.add_attr_meta(
                                "alpr_roi", "is_inside", True
                            )
                        else:
                            frame_meta.remove_obj_meta(obj_meta)


class PostProcessor(NvDsPyFuncPlugin):
    def __init__(
        self,
        min_text_len: int = 9,
        max_text_len: int = 10,
        private_lp_state_codes: List[str] = [],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.min_text_len = min_text_len
        self.max_text_len = max_text_len
        self.private_lp_state_codes = private_lp_state_codes
        self.config = {}

        if hasattr(self, "config_path") and self.config_path:
            with open(self.config_path, "r", encoding="utf8") as stream:
                self.config = yaml.safe_load(stream)

        self.debug = self.config.get("debug", True)
        self.initialize_configurations()

        # Extract configuration details from the provided sources

    def initialize_configurations(self):
        self.vehicle_tracker_cfgs = {}

        if "sources" in self.config:
            for source_id, source_cfg in self.config["sources"].items():
                self.vehicle_tracker_cfgs[
                    source_id
                ] = self.extract_vehicle_tracker_config(source_id, source_cfg)

        self.lp_trackers = {}
        self.track_last_frame_num = defaultdict(lambda: defaultdict(int))

    def extract_vehicle_tracker_config(self, source_id, source_cfg):
        if "vehicle_tracker" in source_cfg:
            return source_cfg["vehicle_tracker"]

    def on_source_eos(self, source_id: str):
        """On source EOS event callback."""
        # Clean up tracking data for the source when the stream ends
        if source_id in self.lp_trackers:
            del self.lp_trackers[source_id]
        if source_id in self.track_last_frame_num:
            del self.track_last_frame_num[source_id]

    def process_frame(self, buffer: Gst.Buffer, frame_meta: NvDsFrameMeta):
        """Process frame metadata.

        :param buffer: Gstreamer buffer with this frame's data.
        :param frame_meta: This frame's metadata.
        """
        # Locate the primary meta object for the frame
        primary_meta_object = None
        for obj_meta in frame_meta.objects:
            if obj_meta.is_primary:
                primary_meta_object = obj_meta
                break

        source_id = frame_meta.source_id

        # Check if there's a primary object and its source ID is in the line configurations.
        if (
            primary_meta_object is not None
            and source_id in self.config["sources"]
        ):
            if source_id not in self.lp_trackers:
                self.lp_trackers[source_id] = {}

            # Update LP trackers and attributes
            lp_found = self.update_lp_trackers_and_attributes(
                frame_meta, source_id
            )

            # If there is any license plate in the frame, update primary meta object
            if lp_found:
                primary_meta_object.add_attr_meta(
                    "alpr_meta", "timestamp", time.time()
                )

            # Handle skip case based on update status
            skip = not lp_found
            self.handle_skip_case(skip, frame_meta)
        else:
            # Handle skip case if no primary object or source ID not in line configurations
            self.handle_skip_case(True, frame_meta)

    def update_lp_trackers_and_attributes(
        self,
        frame_meta: NvDsFrameMeta,
        source_id: str,
    ):
        """Update license plate trackers and associated attributes for vehicles.

        Args:
            frame_meta (NvDsFrameMeta): The frame's metadata.
            source_id (str): The unique identifier for the source.

        Returns:
            bool: True if updates were made to license plate attributes, False otherwise.
        """
        # Initialize a flag to indicate if any license plate was found
        lp_found = False

        # Iterate over vehicle object IDs and their associated license plate object IDs
        for obj_meta in frame_meta.objects:
            if (
                not obj_meta.is_primary
                and obj_meta.label == "lp"
                and obj_meta.parent is not None
            ):
                vehicle_obj_meta = obj_meta.parent
                track_id = vehicle_obj_meta.track_id

                # Create a license plate tracker if it doesn't exist for the current track
                if track_id not in self.lp_trackers[source_id]:
                    vehicle_tracker_cfg = self.vehicle_tracker_cfgs.get(
                        source_id
                    )
                    # TODO: print a warning message here

                    if vehicle_tracker_cfg is not None:
                        self.lp_trackers[source_id][
                            track_id
                        ] = VehicleLPTracker(
                            min_candidates=vehicle_tracker_cfg.get(
                                "min_candidates", 5
                            )
                        )
                    else:
                        self.lp_trackers[source_id][track_id] = None

                lp_tracker = self.lp_trackers[source_id][track_id]
                if lp_tracker is None:
                    continue

                text_value = obj_meta.get_attr_meta(
                    "LPRecognizer", "lp_result"
                )
                text = text_value.value if text_value is not None else ""
                conf = text_value.confidence if text_value is not None else 0

                is_valid_text_len = (
                    self.min_text_len <= len(text) <= self.max_text_len
                )
                is_valid_prefix = (
                    not self.private_lp_state_codes
                    or text[:2].upper() in self.private_lp_state_codes
                )

                # If text is valid, update the license plate tracker
                if text and is_valid_text_len and is_valid_prefix:
                    bbox = obj_meta.bbox.as_xcycwh_int()
                    box_area = bbox[2] * bbox[3]
                    frame_area = frame_meta.roi.width * frame_meta.roi.height
                    lp_tracker.update(
                        type="lp",
                        value=text,
                        confidence=conf,
                        text_area_ratio=box_area / frame_area,
                    )

                # Update vehicle attributes from metadata
                car_make = vehicle_obj_meta.get_attr_meta(
                    "Secondary_CarMake", "car_make"
                )
                if car_make is not None and car_make.value:
                    lp_tracker.update(
                        type="car_make",
                        value=car_make.value,
                        confidence=car_make.confidence,
                    )

                car_type = vehicle_obj_meta.get_attr_meta(
                    "Secondary_CarType", "car_type"
                )
                if car_type is not None and car_type.value:
                    lp_tracker.update(
                        type="car_type",
                        value=car_type.value,
                        confidence=car_type.confidence,
                    )

                # Get the best candidates from the license plate tracker
                best_candidates = lp_tracker.get_best_candidates()

                # If a license plate is found and the tracker is not disabled, update attributes
                if best_candidates["lp"] and not lp_tracker.disabled:
                    # if self.debug:
                    car_make = best_candidates["car_make"]
                    car_type = best_candidates["car_type"]
                    lp_value = best_candidates["lp"]
                    print(
                        f"car_make: {car_make} car_type {car_type} lp {lp_value}"
                    )

                    # Disable the tracker if not in debug mode
                    if not self.debug:
                        lp_tracker.disabled = True

                    # Add the license plate result to vehicle object metadata
                    vehicle_obj_meta.add_attr_meta(
                        "LPRecognizer", "lp_result", best_candidates["lp"]
                    )
                    lp_found = True

        return lp_found

    def handle_skip_case(
        self,
        skip: bool,
        frame_meta: NvDsFrameMeta,
    ):
        if skip:
            frame_meta.set_tag("skip", True)