# coding=utf-8
# Copyright 2021 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" VoxLingua107 Dataset"""


import datasets
import glob
from pathlib import Path


_DATA_URL = "http://bark.phon.ioc.ee/voxlingua107/{}.zip"

_CITATION = """\
@inproceedings{valk2021slt,
  title={{VoxLingua107}: a Dataset for Spoken Language Recognition},
  author={J{\"o}rgen Valk and Tanel Alum{\"a}e},
  booktitle={Proc. IEEE SLT Workshop},
  year={2021},
}
"""

_DESCRIPTION = """\
VoxLingua107 contains data for 107 languages. The total amount of speech in the training set is 6628 hours. 
The average amount of data per language is 62 hours. However, the real amount per language varies a lot. 
There is also a seperate development set containing 1609 speech segments from 33 languages, validated by 
at least two volunteers to really contain the given language.
"""

_HOMEPAGE = "http://bark.phon.ioc.ee/voxlingua107/"

_LICENSE = "Creative Commons Attribution 4.0 International License"

_LANGUAGES = {
    'ab': {'language': 'Abkhazian', 'duration_hours': '10', 'size': '980M'},
    'af': {'language': 'Afrikaans', 'duration_hours': '108', 'size': '10G'},
    'am': {'language': 'Amharic', 'duration_hours': '81', 'size': '7.7G'},
    'ar': {'language': 'Arabic', 'duration_hours': '59', 'size': '5.5G'},
    'as': {'language': 'Assamese', 'duration_hours': '155', 'size': '15G'},
    'az': {'language': 'Azerbaijani', 'duration_hours': '58', 'size': '5.6G'},
    'ba': {'language': 'Bashkir', 'duration_hours': '58', 'size': '5.5G'},
    'be': {'language': 'Belarusian', 'duration_hours': '133', 'size': '13G'},
    'bg': {'language': 'Bulgarian', 'duration_hours': '50', 'size': '4.7G'},
    'bn': {'language': 'Bengali', 'duration_hours': '55', 'size': '5.4G'},
    'bo': {'language': 'Tibetan', 'duration_hours': '101', 'size': '9.3G'},
    'br': {'language': 'Breton', 'duration_hours': '44', 'size': '4.2G'},
    'bs': {'language': 'Bosnian', 'duration_hours': '105', 'size': '9.7G'},
    'ca': {'language': 'Catalan', 'duration_hours': '88', 'size': '8.1G'},
    'ceb': {'language': 'Cebuano', 'duration_hours': '6', 'size': '589M'},
    'cs': {'language': 'Czech', 'duration_hours': '67', 'size': '6.3G'},
    'cy': {'language': 'Welsh', 'duration_hours': '76', 'size': '6.6G'},
    'da': {'language': 'Danish', 'duration_hours': '28', 'size': '2.6G'},
    'de': {'language': 'German', 'duration_hours': '39', 'size': '3.7G'},
    'el': {'language': 'Greek', 'duration_hours': '66', 'size': '6.2G'},
    'en': {'language': 'English', 'duration_hours': '49', 'size': '4.6G'},
    'eo': {'language': 'Esperanto', 'duration_hours': '10', 'size': '916M'},
    'es': {'language': 'Spanish', 'duration_hours': '39', 'size': '3.7G'},
    'et': {'language': 'Estonian', 'duration_hours': '38', 'size': '3.5G'},
    'eu': {'language': 'Basque', 'duration_hours': '29', 'size': '2.8G'},
    'fa': {'language': 'Persian', 'duration_hours': '56', 'size': '5.2G'},
    'fi': {'language': 'Finnish', 'duration_hours': '33', 'size': '3.1G'},
    'fo': {'language': 'Faroese', 'duration_hours': '67', 'size': '6.0G'},
    'fr': {'language': 'French', 'duration_hours': '67', 'size': '6.2G'},
    'gl': {'language': 'Galician', 'duration_hours': '72', 'size': '6.7G'},
    'gn': {'language': 'Guarani', 'duration_hours': '2', 'size': '250M'},
    'gu': {'language': 'Gujarati', 'duration_hours': '46', 'size': '4.5G'},
    'gv': {'language': 'Manx', 'duration_hours': '4', 'size': '374M'},
    'ha': {'language': 'Hausa', 'duration_hours': '106', 'size': '10G'},
    'haw': {'language': 'Hawaiian', 'duration_hours': '12', 'size': '1.2G'},
    'hi': {'language': 'Hindi', 'duration_hours': '81', 'size': '7.7G'},
    'hr': {'language': 'Croatian', 'duration_hours': '118', 'size': '11G'},
    'ht': {'language': 'Haitian', 'duration_hours': '96', 'size': '9.2G'},
    'hu': {'language': 'Hungarian', 'duration_hours': '73', 'size': '6.9G'},
    'hy': {'language': 'Armenian', 'duration_hours': '69', 'size': '6.6G'},
    'ia': {'language': 'Interlingua', 'duration_hours': '3', 'size': '241M'},
    'id': {'language': 'Indonesian', 'duration_hours': '40', 'size': '3.8G'},
    'is': {'language': 'Icelandic', 'duration_hours': '92', 'size': '8.4G'},
    'it': {'language': 'Italian', 'duration_hours': '51', 'size': '4.8G'},
    'iw': {'language': 'Hebrew', 'duration_hours': '96', 'size': '8.9G'},
    'ja': {'language': 'Japanese', 'duration_hours': '56', 'size': '5.1G'},
    'jw': {'language': 'Javanese', 'duration_hours': '53', 'size': '5.0G'},
    'ka': {'language': 'Georgian', 'duration_hours': '98', 'size': '9.2G'},
    'kk': {'language': 'Kazakh', 'duration_hours': '78', 'size': '7.3G'},
    'km': {'language': 'Central Khmer', 'duration_hours': '41', 'size': '4.0G'},
    'kn': {'language': 'Kannada', 'duration_hours': '46', 'size': '4.4G'},
    'ko': {'language': 'Korean', 'duration_hours': '77', 'size': '7.1G'},
    'la': {'language': 'Latin', 'duration_hours': '67', 'size': '6.0G'},
    'lb': {'language': 'Luxembourgish', 'duration_hours': '75', 'size': '7.1G'},
    'ln': {'language': 'Lingala', 'duration_hours': '90', 'size': '8.7G'},
    'lo': {'language': 'Lao', 'duration_hours': '42', 'size': '4.0G'},
    'lt': {'language': 'Lithuanian', 'duration_hours': '82', 'size': '7.7G'},
    'lv': {'language': 'Latvian', 'duration_hours': '42', 'size': '4.0G'},
    'mg': {'language': 'Malagasy', 'duration_hours': '109', 'size': '11G'},
    'mi': {'language': 'Maori', 'duration_hours': '34', 'size': '3.2G'},
    'mk': {'language': 'Macedonian', 'duration_hours': '112', 'size': '11G'},
    'ml': {'language': 'Malayalam', 'duration_hours': '47', 'size': '4.6G'},
    'mn': {'language': 'Mongolian', 'duration_hours': '71', 'size': '6.4G'},
    'mr': {'language': 'Marathi', 'duration_hours': '85', 'size': '8.1G'},
    'ms': {'language': 'Malay', 'duration_hours': '83', 'size': '7.8G'},
    'mt': {'language': 'Maltese', 'duration_hours': '66', 'size': '6.1G'},
    'my': {'language': 'Burmese', 'duration_hours': '41', 'size': '4.0G'},
    'ne': {'language': 'Nepali', 'duration_hours': '72', 'size': '7.1G'},
    'nl': {'language': 'Dutch', 'duration_hours': '40', 'size': '3.8G'},
    'nn': {'language': 'Norwegian Nynorsk', 'duration_hours': '57', 'size': '4.8G'},
    'no': {'language': 'Norwegian', 'duration_hours': '107', 'size': '9.7G'},
    'oc': {'language': 'Occitan', 'duration_hours': '15', 'size': '1.5G'},
    'pa': {'language': 'Panjabi', 'duration_hours': '54', 'size': '5.2G'},
    'pl': {'language': 'Polish', 'duration_hours': '80', 'size': '7.6G'},
    'ps': {'language': 'Pushto', 'duration_hours': '47', 'size': '4.5G'},
    'pt': {'language': 'Portuguese', 'duration_hours': '64', 'size': '6.1G'},
    'ro': {'language': 'Romanian', 'duration_hours': '65', 'size': '6.1G'},
    'ru': {'language': 'Russian', 'duration_hours': '73', 'size': '6.9G'},
    'sa': {'language': 'Sanskrit', 'duration_hours': '15', 'size': '1.6G'},
    'sco': {'language': 'Scots', 'duration_hours': '3', 'size': '269M'},
    'sd': {'language': 'Sindhi', 'duration_hours': '84', 'size': '8.3G'},
    'si': {'language': 'Sinhala', 'duration_hours': '67', 'size': '6.4G'},
    'sk': {'language': 'Slovak', 'duration_hours': '40', 'size': '3.7G'},
    'sl': {'language': 'Slovenian', 'duration_hours': '121', 'size': '12G'},
    'sn': {'language': 'Shona', 'duration_hours': '30', 'size': '2.9G'},
    'so': {'language': 'Somali', 'duration_hours': '103', 'size': '9.9G'},
    'sq': {'language': 'Albanian', 'duration_hours': '71', 'size': '6.6G'},
    'sr': {'language': 'Serbian', 'duration_hours': '50', 'size': '4.7G'},
    'su': {'language': 'Sundanese', 'duration_hours': '64', 'size': '6.2G'},
    'sv': {'language': 'Swedish', 'duration_hours': '34', 'size': '3.1G'},
    'sw': {'language': 'Swahili', 'duration_hours': '64', 'size': '6.1G'},
    'ta': {'language': 'Tamil', 'duration_hours': '51', 'size': '5.0G'},
    'te': {'language': 'Telugu', 'duration_hours': '77', 'size': '7.5G'},
    'tg': {'language': 'Tajik', 'duration_hours': '64', 'size': '6.1G'},
    'th': {'language': 'Thai', 'duration_hours': '61', 'size': '5.8G'},
    'tk': {'language': 'Turkmen', 'duration_hours': '85', 'size': '8.1G'},
    'tl': {'language': 'Tagalog', 'duration_hours': '93', 'size': '8.7G'},
    'tr': {'language': 'Turkish', 'duration_hours': '59', 'size': '5.7G'},
    'tt': {'language': 'Tatar', 'duration_hours': '103', 'size': '9.6G'},
    'uk': {'language': 'Ukrainian', 'duration_hours': '52', 'size': '4.9G'},
    'ur': {'language': 'Urdu', 'duration_hours': '42', 'size': '4.1G'},
    'uz': {'language': 'Uzbek', 'duration_hours': '45', 'size': '4.3G'},
    'vi': {'language': 'Vietnamese', 'duration_hours': '64', 'size': '6.1G'},
    'war': {'language': 'Waray', 'duration_hours': '11', 'size': '1.1G'},
    'yi': {'language': 'Yiddish', 'duration_hours': '46', 'size': '4.4G'},
    'yo': {'language': 'Yoruba', 'duration_hours': '94', 'size': '9.1G'},
    'zh': {'language': 'Mandarin Chinese', 'duration_hours': '44','size': '4.1G'}
}

class VoxLingua107Config(datasets.BuilderConfig):
    """BuilderConfig for VoxLingua107."""

    def __init__(self, name, **kwargs):
        """
        Args:
          data_dir: `string`, the path to the folder containing the files in the
            downloaded .tar
          citation: `string`, citation for the data set
          url: `string`, url for information about the data set
          **kwargs: keyword arguments forwarded to super.
        """
        # self.sub_version = sub_version
        self.language = kwargs.pop("language", None)
        self.size = kwargs.pop("size", None)
        self.duration_hours = kwargs.pop("duration_hours", None)
        description = f"VoxLingua107 dataset in {self.language}. The dataset comprises {self.duration_hours} of speech data. The dataset has a size of {self.size}"
        super().__init__(name=name, description=description, **kwargs)


class VoxLingua107(datasets.GeneratorBasedBuilder):

    DEFAULT_WRITER_BATCH_SIZE = 1000
    BUILDER_CONFIGS = [
        VoxLingua107Config(
            name=lang_id,
            language=_LANGUAGES[lang_id]["language"],
            size=_LANGUAGES[lang_id]["size"],
            duration_hours=_LANGUAGES[lang_id]["duration_hours"]
        )
        for lang_id in _LANGUAGES.keys()
    ]

    def _info(self):
        features = datasets.Features(
            {
                "path": datasets.Value("string"),
                "audio": datasets.Audio(sampling_rate=16_000),
            }
        )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        downloaded_dir = dl_manager.download_and_extract(_DATA_URL.format(self.config.name))
        # filepath = f"{downloaded_dir}/{self.name}"
        filepath = Path(downloaded_dir)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": filepath,
                },
            ),
            # datasets.SplitGenerator(
            #     name=datasets.Split.TEST,
            #     gen_kwargs={
            #         "files": filepath,
            #     },
            # )
        ]

    def _generate_examples(self, filepath):
        """Yields examples."""
        data_fields = list(self._info().features.keys())

        # audio is not a header of the csv files
        data_fields.remove("audio")
        path_idx = data_fields.index("path")
        _i = 0
        for path in Path(filepath).rglob("*.wav"):
            path = path.as_posix()
            yield _i, {
                "path": path,
                "audio": {"path":path, "bytes": open(path, "rb").read()}
            }
            _i += 1
