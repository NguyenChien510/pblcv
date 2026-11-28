"""
Vietnamese person descriptions -> the English the search pipeline understands.

Everything behind the search box is English-only: the DSCA text encoder (BERT) and CLIP were trained on English
captions, and `engine.query_attributes` reads gender, garment colours, carried items, actions and directions with
English word lists. A general vi->en translation model (Helsinki-NLP/opus-mt-vi-en) was tried first and lost exactly
the words that matter here ("kéo vali" -> "case puller", "người" -> "man", colours moved to the wrong garment) and
turned unaccented input into nonsense ("nguoi dan ong deo balo" -> "Bees die").

This module is a rule-based translator for the narrow domain of describing a person:
1. Vietnamese is detected by its letters, or, for text typed without diacritics, by at least two Vietnamese words.
   Anything else is returned unchanged, so English queries behave exactly as before.
2. The text is cut into phrases from a glossary (longest match first). Each input word must match the glossary word
   exactly, or, when the input word has no diacritics at all, match it with the diacritics removed ("ba lo" ->
   "ba lô"). Several readings of an unaccented word ("do" = đỏ red / đồ clothes; "cam" = cam orange / cầm holding)
   are resolved by position: right after a garment, item, colour or "màu" a modifier reading wins.
3. Vietnamese puts modifiers after the noun ("áo khoác xanh đậm"); English and `_parse_colors` need them before it,
   so each noun phrase is rebuilt as  [size/age] [sleeves] [pattern] [colours] [material] noun  ("a dark blue jacket").
   A garment directly after the person gets "wearing", an item or hair gets "with".
4. Words the glossary does not know are dropped from the English and returned, so the UI can show them.
"""
import re
import unicodedata
from typing import Dict, List, NamedTuple, Optional, Tuple

# chunk types
PERSON, GARMENT, ITEM, HAIR = "person", "garment", "item", "hair"                  # heads of noun phrases
COLOR, SHADE, PATTERN, MATERIAL, ADJ, AGE, SLEEVE = "color", "shade", "pattern", "material", "adj", "age", "sleeve"
VERB, ACT, PHRASE, PHRASE_C, CONJ, NUM, MAU, FUNC, SEP = ("verb", "act", "phrase", "phrase_c", "conj", "num", "mau",
                                                          "func", "sep")
HEADS = {PERSON, GARMENT, ITEM, HAIR}
MODS = {COLOR, SHADE, PATTERN, MATERIAL, ADJ, AGE, SLEEVE}
COLOR_CONTEXT = (HEADS - {PERSON}) | MODS | {MAU, PHRASE_C}   # after these an unaccented colour word reads as a colour

# english -> "vietnamese; variants" per type. Longer phrases win, so "áo khoác dài" beats "áo khoác" beats "áo".
LEXICON: Dict[str, Dict[str, str]] = {
    PERSON: {
        "man": "người đàn ông; đàn ông; nam giới; nam; người nam; bạn nam; anh; anh chàng; chàng trai; chàng; "
               "thanh niên; chú; ông; bác trai; anh trai; nam thanh niên; người con trai",
        "old man": "ông cụ; ông già; cụ ông; ông lão",
        "old person": "cụ; người già; người lớn tuổi",
        "boy": "cậu bé; bé trai; con trai; cậu; thằng bé; cậu nhóc",
        "woman": "người phụ nữ; phụ nữ; nữ giới; nữ; người nữ; bạn nữ; chị; cô; bà; bác gái; quý bà; chị gái; "
                 "cô nàng; nàng",
        "old woman": "bà cụ; bà già; cụ bà; bà lão",
        "girl": "cô gái; bé gái; con gái; cô bé; thiếu nữ",
        "person": "người; ai đó; người nào đó; đối tượng",
        "student": "học sinh; sinh viên",
    },
    GARMENT: {
        "t-shirt": "áo thun; áo phông; áo t-shirt; áo tshirt; áo pull; t-shirt; tshirt",
        "polo shirt": "áo polo; polo; áo thun có cổ",
        "shirt": "áo sơ mi; sơ mi; áo sơmi; sơmi; áo",
        "jacket": "áo khoác; áo khoác ngoài; áo jacket; jacket",
        "long coat": "áo khoác dài; áo măng tô; măng tô; áo choàng dài; áo dạ; áo bành tô; áo trench coat",
        "coat": "áo choàng",
        "sweater": "áo len",
        "sweatshirt": "áo nỉ",
        "hoodie": "áo hoodie; hoodie; áo có mũ; áo nỉ có mũ",
        "cardigan": "áo cardigan; cardigan; áo khoác len",
        "blazer": "áo blazer; blazer",
        "suit": "áo vest; vest; áo com lê; com lê; bộ vest; bộ com lê; áo veston",
        "tank top": "áo ba lỗ; áo ba-lỗ",
        "sleeveless top": "áo sát nách; áo không tay",
        "crop top": "áo croptop; croptop; áo crop top; crop top; áo lửng",
        "off-the-shoulder top": "áo trễ vai; áo hở vai",
        "blouse": "áo kiểu; áo blouse; blouse; áo cánh",
        "high-visibility vest": "áo phản quang; áo gile phản quang; áo bảo hộ",
        "vest": "áo gile; áo ghi lê; gile",
        "sports shirt": "áo thể thao; áo đá bóng; áo bóng đá",
        "uniform": "đồng phục; áo đồng phục",
        "long dress": "áo dài",
        "raincoat": "áo mưa",
        "turtleneck sweater": "áo len cổ lọ; áo cổ lọ; áo len cao cổ",
        "school uniform": "đồng phục học sinh; đồng phục trường",
        "sportswear": "đồ thể thao; quần áo thể thao; bộ thể thao; bộ đồ thể thao",
        "dress": "đầm; váy liền; váy đầm; đầm liền; dress",
        "maxi dress": "váy maxi; đầm maxi",
        "pinafore dress": "váy yếm; đầm yếm",
        "skirt": "váy; chân váy; skirt",
        "jeans": "quần jean; quần jeans; quần bò; quần denim; jeans",
        "shorts": "quần short; quần shorts; quần sọt; quần soóc; quần đùi; quần ngắn; quần cộc; shorts",
        "cargo shorts": "quần short túi hộp; quần đùi túi hộp",
        "cargo pants": "quần túi hộp",
        "cropped trousers": "quần lửng; quần ngố",
        "wide trousers": "quần ống rộng; quần suông",
        "trousers": "quần dài; quần tây; quần âu; quần vải",
        "khaki pants": "quần kaki; quần khaki",
        "leggings": "quần legging; quần leggings; legging; leggings; quần bó",
        "sweatpants": "quần thể thao; quần jogger; quần nỉ; jogger",
        "pants": "quần",
        "clothes": "đồ; quần áo; trang phục; bộ đồ",
    },
    ITEM: {
        "shoes": "giày; đôi giày; giầy",
        "sneakers": "giày thể thao; giày sneaker; sneaker; sneakers",
        "high heels": "giày cao gót; cao gót",
        "boots": "bốt; giày bốt; boots; ủng",
        "sandals": "dép; dép lê; sandal; xăng đan",
        "socks": "tất; vớ",
        "backpack": "ba lô; balo; ba-lô; balô; cặp sách; cặp đeo lưng; backpack",
        "hiking backpack": "ba lô leo núi; balo leo núi; ba lô du lịch; balo du lịch",
        "shoulder bag": "túi đeo vai; túi khoác vai; túi vai",
        "crossbody bag": "túi đeo chéo; túi chéo; túi bao tử",
        "plastic bag": "túi nilon; túi ni lông; bao nilon; túi bóng",
        "handbag": "túi xách; túi xách tay; giỏ xách; ví cầm tay",
        "bag": "túi; cặp; giỏ; túi vải; túi tote",
        "suitcase": "vali; va li; va-li; vali kéo; va li kéo; hành lý; hành lí; valy",
        "hat": "mũ; nón",
        "cap": "mũ lưỡi trai; nón lưỡi trai; mũ kết; nón kết",
        "sun hat": "mũ rộng vành; nón rộng vành; mũ vành; nón vành",
        "helmet": "mũ bảo hiểm; nón bảo hiểm",
        "beanie": "mũ len; nón len",
        "face mask": "khẩu trang",
        "glasses": "kính; mắt kính; kính cận; kính mắt",
        "sunglasses": "kính râm; kính mát",
        "headphones": "tai nghe; headphone",
        "scarf": "khăn; khăn quàng; khăn choàng; khăn quàng cổ",
        "umbrella": "ô; dù; cây dù",
        "watch": "đồng hồ",
        "phone": "điện thoại",
        "bicycle": "xe đạp",
        "stroller": "xe đẩy; xe nôi; xe đẩy em bé; xe đẩy trẻ em",
        "dog": "chó; con chó",
        "conical hat": "nón lá",
        "cane": "gậy; cây gậy; chống gậy",
        "child": "em bé; đứa bé; đứa trẻ; trẻ em; trẻ con",
    },
    HAIR: {
        "hair": "tóc; mái tóc",
        "blonde hair": "tóc vàng; tóc vàng hoe; tóc bạch kim",
        "grey hair": "tóc bạc; tóc muối tiêu; tóc hoa râm",
        "white hair": "tóc bạc trắng; tóc trắng",
        "ponytail": "tóc buộc; tóc đuôi ngựa; buộc tóc; cột tóc",
    },
    COLOR: {
        "black": "đen", "white": "trắng", "grey": "xám; ghi", "silver": "bạc",
        "red": "đỏ; đỏ tươi", "maroon": "đỏ đô; đỏ mận; đỏ rượu; đỏ rượu vang; mận",
        "orange": "cam", "yellow": "vàng; vàng chanh", "gold": "vàng kim; vàng đồng",
        "green": "xanh lá; xanh lá cây; xanh lục; xanh lá mạ", "olive": "xanh rêu; xanh quân đội; xanh ô liu",
        "blue": "xanh dương; xanh nước biển; xanh biển; xanh lam; xanh da trời; xanh trời; xanh",
        "navy": "xanh navy; xanh tím than; xanh than; xanh đen; navy",
        "turquoise": "xanh ngọc; xanh ngọc lam; xanh lơ",
        "purple": "tím", "pink": "hồng; hồng phấn", "brown": "nâu; nâu đất; cà phê",
        "beige": "be; màu be", "cream": "kem; màu kem", "khaki": "kaki; khaki", "ginger": "hung; đỏ hung",
    },
    SHADE: {
        "dark": "đậm; sẫm; thẫm; sậm; tối; tối màu; màu tối",
        "light": "nhạt; sáng; sáng màu; màu sáng; lợt; nhạt màu",
    },
    PATTERN: {
        "plaid": "kẻ caro; caro; ca rô; kẻ ca rô; kẻ ô; ô vuông; sọc caro",
        "striped": "kẻ sọc; sọc; kẻ; sọc ngang; sọc dọc; kẻ ngang",
        "floral": "hoa; họa tiết hoa; in hoa; hoa văn hoa; hoa nhí",
        "patterned": "họa tiết; có họa tiết; hoa văn; có hoa văn; in hình",
        "polka dot": "chấm bi",
        "colourful": "nhiều màu; sặc sỡ; nhiều màu sắc; rực rỡ",
    },
    MATERIAL: {"denim": "jean; jeans; bò; denim", "leather": "da", "plastic": "nilon; ni lông"},
    ADJ: {
        "long": "dài", "short": "ngắn; cộc", "large": "to; lớn; to bản; cỡ lớn; bự", "small": "nhỏ",
        "wide": "rộng", "tight": "bó sát; ôm sát", "curly": "xoăn", "straight": "thẳng",
        "bald": "hói; đầu hói; trọc; đầu trọc",
        "tall": "cao; cao lớn", "thin": "gầy; ốm; mảnh khảnh; gầy gò", "heavy": "béo; mập; to béo; mũm mĩm",
    },
    AGE: {
        "young": "trẻ; trẻ tuổi; trẻ trung", "older": "lớn tuổi; có tuổi; cao tuổi; đứng tuổi",
        "old": "già", "middle-aged": "trung niên",
    },
    SLEEVE: {
        "long-sleeved": "dài tay; tay dài", "short-sleeved": "ngắn tay; tay ngắn; cộc tay; tay cộc",
        "sleeveless": "sát nách; không tay",
    },
    VERB: {
        "wearing": "mặc; đội; mang; khoác; quàng; diện",
        "holding": "cầm; ôm; cầm theo; nắm",
        "carrying": "xách; mang theo; vác; xách theo; bế; bồng; ẵm; địu",
        "pulling": "kéo",
        "pushing": "đẩy; dắt",
    },
    ACT: {        # take no object: a garment or item right after them gets "wearing"/"with", as after the person
        "smoking": "hút thuốc",
        "reading": "đọc sách; đọc báo",
        "slowly": "chậm; chậm rãi; từ từ",
        "quickly": "nhanh; nhanh chóng; vội vã",
        "walking": "đi; đi bộ; bước; bước đi; đi dạo; dạo bước; tản bộ; tiến; tiến bước",
        "running": "chạy; chạy bộ; chạy nhanh",
        "standing": "đứng; đứng yên; đứng im; đứng một chỗ; đứng lại",
        "waiting": "đợi; chờ; đứng đợi; đứng chờ",
        "sitting": "ngồi",
    },
    PHRASE: {
        "riding a bicycle": "đi xe đạp; đạp xe",
        "pushing a bicycle": "dắt xe đạp; dắt xe",
        "pushing a stroller": "đẩy xe nôi; đẩy xe đẩy; đẩy xe em bé",
        "walking a dog": "dắt chó; dắt chó đi dạo; dắt cún",
        "together": "cùng nhau; với nhau",
        "on a bench": "trên ghế; trên ghế đá; ghế đá; băng ghế",
        "at the station": "ở nhà ga; tại nhà ga; nhà ga; sân ga",
        "talking on the phone": "nghe điện thoại; gọi điện; nói chuyện điện thoại",
        "looking at the phone": "xem điện thoại; bấm điện thoại; nhìn điện thoại",
        "to the left": "sang trái; qua trái; sang bên trái; qua bên trái; về bên trái; về phía bên trái; "
                       "về phía trái; rẽ trái; hướng trái; về hướng trái; sang hướng trái; bên trái; phía trái",
        "to the right": "sang phải; qua phải; sang bên phải; qua bên phải; về bên phải; về phía bên phải; "
                        "về phía phải; rẽ phải; hướng phải; về hướng phải; sang hướng phải; bên phải; phía phải",
        "from left to right": "từ trái sang phải; từ trái qua phải; từ bên trái sang bên phải; từ trái về phải",
        "from right to left": "từ phải sang trái; từ phải qua trái; từ bên phải sang bên trái; từ phải về trái",
        "towards the camera": "về phía camera; hướng về camera; hướng về phía camera; tiến về camera; "
                              "tiến về phía camera; lại gần camera; đến gần camera; tới gần camera; gần camera hơn; "
                              "về phía máy quay; lại gần máy quay; hướng vào camera; về phía ống kính; tiến lại gần; "
                              "lại gần; tới camera; đến camera; về camera; tới máy quay; về phía trước camera",
        "away from the camera": "ra xa camera; xa khỏi camera; rời xa camera; rời khỏi camera; xa camera; ra xa; "
                                "xa dần; xa khỏi máy quay; ra xa máy quay; quay lưng về camera; quay lưng lại camera; "
                                "quay lưng",
        "the camera": "camera; máy quay; ống kính",
        "with a british flag": "in cờ anh; hình cờ anh; cờ anh; in hình cờ anh",
    },
    PHRASE_C: {   # a colour follows ("mặc toàn đồ đen")
        "dressed all in": "mặc toàn đồ; mặc toàn màu; toàn đồ; toàn màu; mặc đồ toàn màu",
    },
    CONJ: {"and": "và", "or": "hoặc", "with": "đeo; có; với; cùng; cùng với; kèm", "over": "bên ngoài"},
    NUM: {"two": "hai; 2"},
    MAU: {"": "màu; có màu; màu sắc"},
    FUNC: {"": "một; cái; chiếc; con; đang; thì; là; này; đó; kia; ấy; của; bị; được; rất; khá; hơi; lắm; tìm; "
               "tìm kiếm; cho; tôi; mình; giúp; hãy; vừa; ở; trong; video; nào; đâu; mà; đều; cũng; như; loại; "
               "đôi; bộ; về; qua; lại; ra; gần; xa; từ; đến; tới; theo; trên; dưới; phía; bên; hướng; người ta; "
               "toàn thân; từ đầu đến chân; clip; hình; ảnh; vào; lúc; khi; rồi; nhé; ạ; nha; do; sang; tôi; "
               "đã; chỉ; thấy; ai"},
}
STRICT = {"ô"}   # matched only when typed with its diacritics (unaccented "o" is far more often "ở")
EN_TYPES = {     # English words kept as they are inside a Vietnamese query
    **{w: GARMENT for w in ("shirt", "top", "jacket", "coat", "sweater", "blouse", "suit", "jumper", "sweatshirt",
                            "parka", "overcoat", "pants", "trousers")},
    **{w: ITEM for w in ("bag", "handbag", "purse", "hat", "cap", "mask", "glasses", "sunglasses", "scarf",
                         "umbrella", "suitcase", "luggage", "helmet", "beanie")},
    **{w: COLOR for w in ("black", "white", "grey", "gray", "red", "orange", "yellow", "green", "blue", "purple",
                          "pink", "brown", "beige", "maroon", "turquoise", "olive")},
}

_VI_LETTERS = re.compile("[ăâđêôơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹĩũõ]")
_TONE_OLD = {"oà": "òa", "oá": "óa", "oả": "ỏa", "oã": "õa", "oạ": "ọa", "oè": "òe", "oé": "óe", "oẻ": "ỏe",
             "oẽ": "õe", "oẹ": "ọe", "uỳ": "ùy", "uý": "úy", "uỷ": "ủy", "uỹ": "ũy", "uỵ": "ụy"}
_TONE_RX = re.compile("(" + "|".join(_TONE_OLD) + r")(?![a-zà-ỹ])")
_TOKEN_RX = re.compile(r"[^\W_]+(?:-[^\W_]+)*|[,.;!?]")


def strip_marks(s: str) -> str:
    nfd = unicodedata.normalize("NFD", s.replace("đ", "d").replace("Đ", "D"))
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")


def _norm(text: str) -> str:
    """NFC, lower case, one tone placement for oa/oe/uy at the end of a syllable (hoà -> hòa)."""
    return _TONE_RX.sub(lambda m: _TONE_OLD[m.group(1)], unicodedata.normalize("NFC", text.lower()))


def _tokens(text: str) -> List[str]:
    return _TOKEN_RX.findall(_norm(text))


class _Entry(NamedTuple):
    words: Tuple[str, ...]
    out: Tuple[Tuple[str, str], ...]     # (english, type) chunks; FUNC/MAU have empty english


def _build():
    exact: Dict[Tuple[str, ...], List[_Entry]] = {}
    loose: Dict[Tuple[str, ...], List[_Entry]] = {}
    for typ, table in LEXICON.items():
        for en, variants in table.items():
            for vi in variants.split(";"):
                words = tuple(_tokens(vi))
                if not words:
                    continue
                e = _Entry(words, ((en, typ),))
                exact.setdefault(words, []).append(e)
                if vi.strip() not in STRICT:
                    loose.setdefault(tuple(strip_marks(w) for w in words), []).append(e)
    for footwear in ("giày", "giầy", "dép", "tất", "vớ", "bốt", "giày thể thao", "giày cao gót", "giày bốt"):
        words = ("đi",) + tuple(_tokens(footwear))     # "đi giày" = wearing shoes, not walking
        item = exact[tuple(_tokens(footwear))][0].out[0]
        e = _Entry(words, (("wearing", VERB), item))
        exact.setdefault(words, []).append(e)
        loose.setdefault(tuple(strip_marks(w) for w in words), []).append(e)
    return exact, loose, max(len(k) for k in exact)


_EXACT, _LOOSE, _MAX_N = _build()
# Unaccented Vietnamese words that are not also common English words, for detecting text typed without diacritics.
_EN_COMMON = {"a", "an", "the", "in", "on", "to", "of", "and", "or", "with", "is", "are", "man", "men", "woman",
              "women", "person", "people", "wearing", "walking", "carrying", "holding", "his", "her", "who", "at"}
_AMBIGUOUS = {"do", "den", "cam", "tim", "co", "ba", "la", "da", "be", "ve", "ra", "va", "di", "o", "bo", "cu", "mu",
              "non", "hai", "chi", "cho", "dung", "tay", "vai", "nu", "nam", "sang", "toi", "xa", "gan", "ke", "hung",
              "in", "ong", "long", "bac", "kem", "mang", "can"}
_VI_HINTS = {strip_marks(w) for k in _EXACT for w in k} - _EN_COMMON - _AMBIGUOUS - set(EN_TYPES) - {
    "short", "shorts", "jean", "jeans", "polo", "vest", "hoodie", "cardigan", "blazer", "camera", "sneaker",
    "sneakers", "sandal", "boots", "jogger", "legging", "leggings", "navy", "croptop", "crop", "top", "blouse",
    "jacket", "headphone", "denim", "khaki", "kaki", "tshirt", "t-shirt", "pull", "trench", "coat", "clip", "video",
    "dress", "skirt", "backpack", "2"}


def is_vietnamese(text: str) -> bool:
    t = _norm(text)
    if _VI_LETTERS.search(t):
        return True
    words = re.findall(r"[a-z]+", t)
    hits = sum(w in _VI_HINTS for w in words)
    return hits >= 2 and hits > sum(w in _EN_COMMON or w in EN_TYPES for w in words)


def _has_marks(w: str) -> bool:
    return strip_marks(w) != w


def _choose(cands: List[_Entry], prev: Optional[str], next_holdable: bool = False) -> _Entry:
    """One reading of an ambiguous phrase from what comes before and after it (exact matches are listed first)."""
    if len(cands) == 1:
        return cands[0]
    first = lambda e: e.out[0][1]
    if next_holdable:                       # "ao xanh cam tui" = cầm túi (holding a bag), not orange
        verbs = [e for e in cands if first(e) in (VERB, CONJ)]
        if verbs and any(first(e) in MODS for e in cands):
            return verbs[0]
    if prev in COLOR_CONTEXT:
        mods = [e for e in cands if first(e) in MODS]
        if mods:
            return mods[0]
    if prev == PERSON:                      # "nguoi dan ong hoi" = hói (bald), not hơi
        traits = [e for e in cands if first(e) in (AGE, ADJ)]
        if traits:
            return traits[0]
    if prev in (None, SEP):
        persons = [e for e in cands if first(e) == PERSON]
        if persons:
            return persons[0]
    rank = lambda e: (first(e) in MODS, first(e) == PERSON, first(e) in (FUNC, MAU), first(e) not in HEADS)
    return sorted(cands, key=rank)[0]


def _candidates(seg: Tuple[str, ...], accented: bool = False) -> List[_Entry]:
    """Glossary readings of a phrase. In a sentence typed with diacritics an unaccented word is taken literally
    ("áo cam" = orange; cầm would have been typed as such) unless it has no exact reading."""
    cands = list(_EXACT.get(seg, []))
    if accented and cands:
        return cands
    for e in _LOOSE.get(tuple(strip_marks(w) for w in seg), []):
        if e not in cands and all(w == p or (not _has_marks(w) and w == strip_marks(p)) for w, p in zip(seg, e.words)):
            cands.append(e)
    return cands


_HOLDABLE = {"bag", "handbag", "plastic bag", "umbrella", "phone", "cane", "shoulder bag", "child", "dog"}


def _starts_holdable(tokens: List[str], i: int, accented: bool) -> bool:
    """Whether an item one holds (bag, umbrella, phone...) starts at position i: "cam tui" = cầm túi."""
    for n in range(min(_MAX_N, len(tokens) - i), 0, -1):
        cands = _candidates(tuple(tokens[i:i + n]), accented)
        if cands:
            return any(e.out[0][0] in _HOLDABLE for e in cands)
    return False


def _chunks(tokens: List[str]) -> Tuple[List[Tuple[str, str]], List[str]]:
    out: List[Tuple[str, str]] = []
    unknown: List[str] = []
    prev: Optional[str] = None
    accented = any(_has_marks(w) for w in tokens)
    i = 0
    while i < len(tokens):
        if tokens[i] in ",.;!?":
            out.append((",", SEP))
            prev, i = SEP, i + 1
            continue
        hit = None
        for n in range(min(_MAX_N, len(tokens) - i), 0, -1):
            seg = tuple(tokens[i:i + n])
            if any(w in ",.;!?" for w in seg):
                continue
            cands = _candidates(seg, accented)
            if cands:
                hit = (n, _choose(cands, prev, len(cands) > 1 and _starts_holdable(tokens, i + n, accented)))
                break
        if hit is None:
            w = tokens[i]
            if w in EN_TYPES:
                out.append((w, EN_TYPES[w]))
                prev = EN_TYPES[w]
            elif not w.isdigit():
                unknown.append(w)
            i += 1
            continue
        n, e = hit
        for en, typ in e.out:
            if typ == FUNC:
                continue
            out.append((en, typ))
            prev = typ
        i += n
    return out, unknown


def _article(phrase: List[str]) -> List[str]:
    last = phrase[-1].split()[-1]
    plural = (last.endswith("s") and not last.endswith("ss")) or last in ("glasses", "sunglasses", "hair", "sportswear")
    if plural:
        return phrase
    vowel = phrase[0][0] in "aeiou" and not phrase[0].startswith(("uni", "use", "eu", "one"))
    return ["an" if vowel else "a"] + phrase


_PREFIX_ADJ = {"wide", "cropped", "long", "sleeveless", "off-the-shoulder"}   # go before the colours, not after


def _colours(mods: List[Tuple[str, str]]) -> Tuple[List[str], List[str]]:
    """(colours with their shades, shades without a colour)."""
    colours: List[str] = []
    lone_shades: List[str] = []
    for t, y in mods:
        if y == COLOR:
            colours.append(t)
        elif y == SHADE:          # Vietnamese puts the shade after its colour: "xanh đậm" -> "dark blue"
            if colours and colours[-1].split()[0] not in ("dark", "light"):
                colours[-1] = f"{t} {colours[-1]}"
            else:
                lone_shades.append(t)
    return colours, lone_shades


def _noun_phrase(head: str, typ: str, mods: List[Tuple[str, str]], plural: bool) -> List[str]:
    adjs = [t for t, y in mods if y in (AGE, ADJ)]
    first, _, rest = head.partition(" ")
    if rest and first in _PREFIX_ADJ:     # "wide trousers" + orange -> "wide orange trousers"
        adjs.append(first)
        head = rest
    sleeves = [t for t, y in mods if y == SLEEVE]
    patterns = [t for t, y in mods if y == PATTERN]
    materials = [t for t, y in mods if y == MATERIAL]
    colours, lone_shades = _colours(mods)
    if plural and not head.endswith("s"):
        head = head[:-6] + "people" if head.endswith("person") else head + "s"
    phrase = adjs + sleeves + patterns + lone_shades + ([" and ".join(colours)] if colours else []) + materials + [head]
    return phrase if plural else _article(phrase)


def _is_flowers(chunks: List[Tuple[str, str]], i: int) -> bool:
    return chunks[i] == ("floral", PATTERN) and i + 1 < len(chunks) and chunks[i + 1][1] == COLOR


def _assemble(chunks: List[Tuple[str, str]]) -> str:
    words: List[str] = []
    last: Optional[str] = None       # PERSON / "head" / other type of what was emitted last
    plural = False
    i = 0
    while i < len(chunks):
        text, typ = chunks[i]
        if typ in HEADS:
            allowed = {AGE, ADJ} if typ == PERSON else MODS | {MAU}
            mods: List[Tuple[str, str]] = []
            j = i + 1
            while j < len(chunks):
                t2, y2 = chunks[j]
                if _is_flowers(chunks, j) and any(y == COLOR for _, y in mods):
                    break                              # "đầm đen hoa đỏ": a black dress with red flowers
                if y2 in allowed:
                    mods.append(chunks[j])
                elif (y2 == CONJ and t2 in ("and", "or") and typ != PERSON and j + 1 < len(chunks)
                      and chunks[j + 1][1] == COLOR and any(y == COLOR for _, y in mods)):
                    pass                               # "trắng và đen": the colours are joined below
                else:
                    break
                j += 1
            if typ != PERSON and last in (PERSON, ACT, PHRASE):
                words.append("wearing" if typ == GARMENT else "with")
            elif typ != PERSON and last == "head":
                words.append("and")
            words += _noun_phrase(text, typ, [m for m in mods if m[1] != MAU], plural)
            last, plural, i = (PERSON if typ == PERSON else "head"), False, j
            continue
        if _is_flowers(chunks, i):
            j = i + 1
            while j < len(chunks) and chunks[j][1] in (COLOR, SHADE):
                j += 1
            colours, shades = _colours(chunks[i + 1:j])
            words += ["with"] + shades + [" and ".join(colours), "flowers"]
            last, i = "head", j
            continue
        if typ == NUM:
            plural = True
        if typ == PHRASE and text in words:
            i += 1
            continue
        if typ != MAU:
            words.append(text)
            last = typ
        i += 1
    s = " ".join(w for w in words if w)
    s = re.sub(r"\s+,", ",", s)
    s = re.sub(r"(,\s*)+", ", ", s).strip(" ,")
    return s


def translate(prompt: str) -> Tuple[str, str, List[str]]:
    """(english, language, unknown words). English input comes back unchanged with language "en"."""
    if not is_vietnamese(prompt):
        return prompt.strip(), "en", []
    chunks, unknown = _chunks(_tokens(prompt))
    return _assemble(chunks), "vi", unknown
