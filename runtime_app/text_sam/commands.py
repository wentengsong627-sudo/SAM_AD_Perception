"""A deliberately explicit vocabulary; unknown Chinese is never silently discarded."""
from dataclasses import asdict, dataclass
import re


@dataclass
class Command:
    original: str
    query: str
    selector: str = "best"
    ordinal: int | None = None
    action: str = "track"
    quantity: str = "one"

    def to_dict(self):
        return asdict(self)


VOCABULARY = {
    "自行车": "bicycle", "摩托车": "motorcycle", "公交车": "bus",
    "公共汽车": "bus", "卡车": "truck", "货车": "truck", "汽车": "car",
    "小车": "car", "轿车": "car", "车辆": "car", "行人": "person",
    "人物": "person", "男人": "man", "女人": "woman", "男孩": "boy",
    "女孩": "girl", "小孩": "child", "人": "person", "斑马": "zebra",
    "河马": "hippopotamus", "狗": "dog", "猫": "cat", "飞机": "airplane",
    "船": "boat", "建筑物": "building", "建筑": "building", "房子": "house",
    "桥梁": "bridge", "桥": "bridge", "树木": "tree", "树": "tree",
    "杯子": "cup", "球": "ball", "红色": "red", "蓝色": "blue",
    "白色": "white", "黑色": "black", "黄色": "yellow", "绿色": "green",
}


def parse_command(text: str) -> Command:
    original = text.strip()
    if not original:
        raise ValueError("请输入目标描述，例如：左边的汽车、行人、car。")
    if original.lower() in {"停止", "停止跟踪", "结束", "stop", "stop tracking"}:
        return Command(original, "", action="stop")
    value = original.lower()
    quantity = "one"
    # Quantifiers belong to the application, not the detector's category phrase.
    quantifiers = r"(?<![a-z])(all|every|each)(?![a-z])|所有|全部|每一[辆个只]?|每个|每辆|每只"
    if re.search(quantifiers, value):
        quantity = "all"
        value = re.sub(quantifiers, " ", value)
    selector = "best"
    for phrases, kind in [
        (("左边", "左侧", "最左", "leftmost", "on the left"), "left"),
        (("右边", "右侧", "最右", "rightmost", "on the right"), "right"),
        (("最大的", "最大", "largest", "biggest"), "largest"),
        (("最小的", "最小", "smallest"), "smallest"),
    ]:
        for phrase in phrases:
            if phrase in value:
                if selector != "best" and selector != kind:
                    raise ValueError("本版本请使用一种位置/尺寸条件，例如“左边的汽车”。")
                selector = kind
                value = value.replace(phrase, " ")
    ordinal = None
    match = re.search(r"第?\s*(\d+)\s*号", value)
    if match:
        ordinal = int(match.group(1))
        if ordinal < 1:
            raise ValueError("目标编号从 1 开始。")
        value = value[:match.start()] + value[match.end():]
    if quantity == "all" and (ordinal is not None or selector != "best"):
        raise ValueError("请区分“所有汽车”和“左边的汽车 / 2号汽车”；暂不组合全部与单目标条件。")
    for phrase in ("请帮我", "帮我", "请", "给我", "跟踪", "追踪", "分割", "识别", "切换到", "画面中", "画面里", "画面", "目标", "这个", "那个"):
        value = value.replace(phrase, " ")
    for source in sorted(VOCABULARY, key=len, reverse=True):
        value = value.replace(source, " " + VOCABULARY[source] + " ")
    value = value.replace("的", " ").replace("一辆", " ").replace("一只", " ")
    if re.search(r"[\u4e00-\u9fff]", value):
        raise ValueError("这句中文包含词表外的描述。请改用简短中文，或直接输入英文目标描述。")
    value = re.sub(r"^(please\s+)?(track|segment|follow|find)\s+", "", value.strip())
    value = re.sub(r"\s+", " ", value).strip(" .,!，。！？;；")
    if quantity == "all":
        plurals = {"cars": "car", "people": "person", "persons": "person",
                   "dogs": "dog", "cats": "cat", "cups": "cup", "balls": "ball", "buses": "bus",
                   "trucks": "truck", "bicycles": "bicycle", "motorcycles": "motorcycle"}
        value = re.sub(r"\b(" + "|".join(plurals) + r")\b", lambda m: plurals[m[0]], value)
    if not value:
        raise ValueError("请同时写明目标类别，例如“二号汽车”。")
    return Command(original, value + ".", selector, ordinal, quantity=quantity)

