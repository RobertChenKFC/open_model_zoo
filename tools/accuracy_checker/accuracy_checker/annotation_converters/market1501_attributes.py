import os

from mat4py import loadmat

from edgetpu_pass.definitions import CONFIG
from .format_converter import DirectoryBasedAnnotationConverter, ConverterReturn
from ..representation import MultiLabelRecognitionAnnotation


class Market1501AttributesConverter(DirectoryBasedAnnotationConverter):
    __provider__ = "market1501_attributes"
    annotation_types = (MultiLabelRecognitionAnnotation,)

    def configure(self):
        super().configure()
        self.market_attributes_path = CONFIG.get(
            "market1501_attributes", "path"
        )

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        dataset_attrs = loadmat(
            self.market_attributes_path
        )["market_attribute"]["test"]
        person_id_to_index = {
            person_id: index
            for index, person_id in enumerate(dataset_attrs["image_index"])
        }

        # These are the labels used in the model
        model_attr_name_to_index = {
            "is_male": 0,
            "has_bag": 1,
            "has_backpack": 2,
            "has_hat": 3,
            "has_longsleeves": 4,
            "has_longpants": 5,
            "has_longhair": 6
        }
        num_attributes = len(model_attr_name_to_index)

        # These are how the dataset labels are converted to the model labels
        def matches(val: int):
            def f(original: bool, cur: int):
                return cur == val
            return f

        def matches_or(val: int):
            def f(original: bool, cur: int):
                return original or (cur == val)
            return f

        male = 1
        yes = 2
        long_sleeves = 1
        long_pants = 1
        long_hair = 2
        dataset_and_model_attrs = [
            # The keys represent the attribute names used in the dataset,
            # and the value is of the form (name, conversion), where name
            # is the attribute name used in the model, and conversion is the
            # function to convert the dataset's attribute to the model's
            # attribute.
            ("gender", ("is_male", matches(male))),
            ("bag", ("has_bag", matches_or(yes))),
            ("handbag", ("has_bag", matches_or(yes))),
            ("backpack", ("has_backpack", matches(yes))),
            ("hat", ("has_hat", matches(yes))),
            ("up", ("has_longsleeves", matches(long_sleeves))),
            ("down", ("has_longpants", matches(long_pants))),
            ("hair", ("has_longhair", matches(long_hair)))
        ]

        annotations = []
        gallery_dir = os.path.join(self.data_dir, "bounding_box_test")
        for img_filename in os.listdir(gallery_dir):
            # This dataset doesn't use distractor and junk images
            if (
                img_filename.startswith("-") or
                img_filename.startswith("0000") or
                not img_filename.endswith(".jpg")
            ):
                continue

            person_id = img_filename[:4]
            person_index = person_id_to_index[person_id]
            identifier = os.path.abspath(
                os.path.join(gallery_dir, img_filename)
            )
            attrs = [False] * num_attributes
            for (
                dataset_attr_name, (model_attr_name, conversion)
            ) in dataset_and_model_attrs:
                attr_index = model_attr_name_to_index[model_attr_name]
                attrs[attr_index] = conversion(
                    attrs[attr_index],
                    dataset_attrs[dataset_attr_name][person_index]
                )
            attrs = [int(attr) for attr in attrs]
            attrs_annotation = MultiLabelRecognitionAnnotation(
                identifier, attrs
            )
            annotations.append(attrs_annotation)

        content_errors = None if not check_content else []
        return ConverterReturn(
            annotations, self.generate_meta(model_attr_name_to_index),
            content_errors
        )

    @staticmethod
    def generate_meta(attribute_values_mapping):
        return {
            'label_map': {
                value: key for key, value in attribute_values_mapping.items()
            }
        }
