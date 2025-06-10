import re, string


def normalText(sent):
    #Chuẩn hóa tiếng Việt, xử lý emoj, chuẩn hóa tiếng Anh, thuật ngữ
    replace_list = {
        'òa': 'oà', 'óa': 'oá', 'ỏa': 'oả', 'õa': 'oã', 'ọa': 'oạ', 'òe': 'oè', 'óe': 'oé','ỏe': 'oẻ',
        'õe': 'oẽ', 'ọe': 'oẹ', 'ùy': 'uỳ', 'úy': 'uý', 'ủy': 'uỷ', 'ũy': 'uỹ','ụy': 'uỵ', 'uả': 'ủa',
        'ả': 'ả', 'ố': 'ố', 'u´': 'ố','ỗ': 'ỗ', 'ồ': 'ồ', 'ổ': 'ổ', 'ấ': 'ấ', 'ẫ': 'ẫ', 'ẩ': 'ẩ',
        'ầ': 'ầ', 'ỏ': 'ỏ', 'ề': 'ề','ễ': 'ễ', 'ắ': 'ắ', 'ủ': 'ủ', 'ế': 'ế', 'ở': 'ở', 'ỉ': 'ỉ',
        'ẻ': 'ẻ', 'àk': u' à ','aˋ': 'à', 'iˋ': 'ì', 'ă´': 'ắ','ử': 'ử', 'e˜': 'ẽ', 'y˜': 'ỹ', 'a´': 'á',
        #Quy các icon 
        # VUI VẺ/THÍCH THÚ
        "😊": "Vui vẻ", "😁": "Vui vẻ", "😃": "Vui vẻ", "😄": "Vui vẻ",
        "😆": "Vui vẻ", "🤣": "Vui vẻ", "😹": "Vui vẻ", "😍": "Vui vẻ",
        "😘": "Vui vẻ", "😙": "Vui vẻ", "😚": "Vui vẻ", "😋": "Vui vẻ",
        "😛": "Vui vẻ", "😜": "Vui vẻ", "🤪": "Vui vẻ", "🤗": "Vui vẻ",
        "😎": "Vui vẻ", "🙂": "Vui vẻ", "💃": "Vui vẻ", "🕺": "Vui vẻ",
        "💖": "Vui vẻ", "💞": "Vui vẻ", "💗": "Vui vẻ", "💕": "Vui vẻ",
        "💓": "Vui vẻ", "❤️": "Vui vẻ", "❤": "Vui vẻ", "♥": "Vui vẻ",
        "💜": "Vui vẻ", "💙": "Vui vẻ", "💚": "Vui vẻ", "💛": "Vui vẻ",
        "💘": "Vui vẻ", "✨": "Vui vẻ", "🎉": "Vui vẻ", "🌟": "Vui vẻ",
        "🌸": "Vui vẻ", "🌺": "Vui vẻ", "🌼": "Vui vẻ", "😇": "Vui vẻ",

        # BUỒN BÃ
        "😢": "Buồn bã", "😭": "Buồn bã", "😞": "Buồn bã", "😔": "Buồn bã",
        "😟": "Buồn bã", "😿": "Buồn bã", "😩": "Buồn bã", "😫": "Buồn bã",
        "😓": "Buồn bã", "😥": "Buồn bã", "☹": "Buồn bã", "🙁": "Buồn bã",
        "😰": "Buồn bã", "😪": "Buồn bã", "😕": "Buồn bã",

        # TỨC GIẬN
        "😡": "Tức giận", "😠": "Tức giận", "🤬": "Tức giận", "👿": "Tức giận", "💢": "Tức giận",
        "😤": "Tức giận", "😾": "Tức giận", "🚫": "Tức giận",

        # NGẠC NHIÊN
        "😲": "Ngạc nhiên", "😯": "Ngạc nhiên", "😮": "Ngạc nhiên", "😳": "Ngạc nhiên",
        "😱": "Ngạc nhiên", "🤯": "Ngạc nhiên", "😵": "Ngạc nhiên",

        # SỢ HÃI
        "😨": "Sợ hãi", "😰": "Sợ hãi", "😖": "Sợ hãi", "😬": "Sợ hãi", "😧": "Sợ hãi",
        "😷": "Sợ hãi", "👻": "Sợ hãi", "😱": "Sợ hãi",

        # KINH TỞM
        "🤢": "Kinh tởm", "🤮": "Kinh tởm", "💩": "Kinh tởm", "😒": "Kinh tởm",
        "😑": "Kinh tởm", "😣": "Kinh tởm", "😠": "Kinh tởm", "👎": "Kinh tởm",

        # KHÁC
        "🤔": "Khác", "😐": "Khác", "🤨": "Khác", "😶": "Khác",
        "🙃": "Khác", "😏": "Khác", "🧐": "Khác", "😌": "Khác",
        "💀": "Khác", "🔥": "Khác", "?": "Khác", "…": "Khác",


        #Chuẩn hóa 1 số sentiment words/English words
        ':))': '  positive ', ':)': ' positive ', 'ô kêi': ' ok ', 'okie': ' ok ', ' o kê ': ' ok ',
        'okey': ' ok ', 'ôkê': ' ok ', 'oki': ' ok ', ' oke ':  ' ok ',' okay':' ok ','okê':' ok ',
        ' tks ': u' cám ơn ', 'thks': u' cám ơn ', 'thanks': u' cám ơn ', 'ths': u' cám ơn ', 'thank': u' cám ơn ',
        '⭐': 'star ', '*': 'star ', '🌟': 'star ', '🎉': u' positive ',
        'kg ': u' không ','not': u' không ', u' kg ': u' không ', '"k ': u' không ',' kh ':u' không ','kô':u' không ','hok':u' không ',' kp ': u' không phải ',u' kô ': u' không ', '"ko ': u' không ', u' ko ': u' không ', u' k ': u' không ', 'khong': u' không ', u' hok ': u' không ',
        'he he': ' positive ','hehe': ' positive ','hihi': ' positive ', 'haha': ' positive ', 'hjhj': ' positive ',
        ' lol ': ' negative ',' cc ': ' negative ','cute': u' dễ thương ','huhu': ' negative ', ' vs ': u' với ', 'wa': ' quá ', 'wá': u' quá', 'j': u' gì ', '“': ' ',
        ' sz ': u' cỡ ', 'size': u' cỡ ', u' đx ': u' được ', 'dk': u' được ', 'dc': u' được ', 'đk': u' được ',
        'đc': u' được ','authentic': u' chuẩn chính hãng ',u' aut ': u' chuẩn chính hãng ', u' auth ': u' chuẩn chính hãng ', 'thick': u' positive ', 'store': u' cửa hàng ',
        'shop': u' cửa hàng ', 'sp': u' sản phẩm ', 'gud': u' tốt ','god': u' tốt ','wel done':' tốt ', 'good': u' tốt ', 'gút': u' tốt ',
        'sấu': u' xấu ','gut': u' tốt ', u' tot ': u' tốt ', u' nice ': u' tốt ', 'perfect': 'rất tốt', 'bt': u' bình thường ',
        'time': u' thời gian ', 'qá': u' quá ', u' ship ': u' giao hàng ', u' m ': u' mình ', u' mik ': u' mình ',
        'ể': 'ể', 'product': 'sản phẩm', 'quality': 'chất lượng','chat':' chất ', 'excelent': 'hoàn hảo', 'bad': 'tệ','fresh': ' tươi ','sad': ' tệ ',
        'date': u' hạn sử dụng ', 'hsd': u' hạn sử dụng ','quickly': u' nhanh ', 'quick': u' nhanh ','fast': u' nhanh ','delivery': u' giao hàng ',u' síp ': u' giao hàng ',
        'beautiful': u' đẹp tuyệt vời ', u' tl ': u' trả lời ', u' r ': u' rồi ', u' shopE ': u' cửa hàng ',u' order ': u' đặt hàng ',
        'chất lg': u' chất lượng ',u' sd ': u' sử dụng ',u' dt ': u' điện thoại ',u' nt ': u' nhắn tin ',u' tl ': u' trả lời ',u' sài ': u' xài ',u'bjo':u' bao giờ ',
        'thik': u' thích ',u' sop ': u' cửa hàng ', ' fb ': ' facebook ', ' face ': ' facebook ', ' very ': u' rất ',u'quả ng ':u' quảng  ',
        'dep': u' đẹp ',u' xau ': u' xấu ','delicious': u' ngon ', u'hàg': u' hàng ', u'qủa': u' quả ',
        'iu': u' yêu ','fake': u' giả mạo ', 'trl': 'trả lời', '><': u' positive ',
        ' por ': u' tệ ',' poor ': u' tệ ', 'ib':u' nhắn tin ', 'rep':u' trả lời ',u'fback':' feedback ','fedback':' feedback ',
        #dưới 3* quy về 1*, trên 3* quy về 5*
        '6 sao': ' 5star ','6 star': ' 5star ', '5star': ' 5star ','5 sao': ' 5star ','5sao': ' 5star ',
        'starstarstarstarstar': ' 5star ', '1 sao': ' 1star ', '1sao': ' 1star ','2 sao':' 1star ','2sao':' 1star ',
        '2 starstar':' 1star ','1star': ' 1star ', '0 sao': ' 1star ', '0star': ' 1star ',}
    sent = sent.lower()
    for k, v in replace_list.items():
        sent = sent.replace(k, v)


    sent = str(sent).replace('_',' ').replace('/',' trên ')
    sent = re.sub('-{2,}','',sent)
    sent = re.sub('\\s+',' ', sent)
    patPrice = r'([0-9]+k?(\s?-\s?)[0-9]+\s?(k|K))|([0-9]+(.|,)?[0-9]+\s?(triệu|ngàn|trăm|k|K|))|([0-9]+(.[0-9]+)?Ä‘)|([0-9]+k)'
    patHagTag = r'#\s?[aăâbcdđeêghiklmnoôơpqrstuưvxyàằầbcdđèềghìklmnòồờpqrstùừvxỳáắấbcdđéếghíklmnóốớpqrstúứvxýảẳẩbcdđẻểghỉklmnỏổởpqrstủửvxỷạặậbcdđẹệghịklmnọộợpqrstụựvxỵãẵẫbcdđẽễghĩklmnõỗỡpqrstũữvxỹAĂÂBCDĐEÊGHIKLMNOÔƠPQRSTUƯVXYÀẰẦBCDĐÈỀGHÌKLMNÒỒỜPQRSTÙỪVXỲÁẮẤBCDĐÉẾGHÍKLMNÓỐỚPQRSTÚỨVXÝẠẶẬBCDĐẸỆGHỊKLMNỌỘỢPQRSTỤỰVXỴẢẲẨBCDĐẺỂGHỈKLMNỎỔỞPQRSTỦỬVXỶÃẴẪBCDĐẼỄGHĨKLMNÕỖỠPQRSTŨỮVXỸ]+'
    patURL = r"(?:http://|www.)[^\"]+"
    sent = re.sub(patURL,'website',sent)
    sent = re.sub(patHagTag,' hagtag ',sent)
    sent = re.sub(patPrice, ' giá tiền ', sent)
    sent = re.sub(r'\.+','.',sent)
    sent = re.sub('(hagtag\\s+)+',' hagtag ',sent)
    sent = re.sub('\\s+',' ',sent)
    return sent

def deleteIcon(text):
    text = text.lower()
    s = ''
    pattern = r"[a-zA-ZaăâbcdđeêghiklmnoôơpqrstuưvxyàằầbcdđèềghìklmnòồờpqrstùừvxỳáắấbcdđéếghíklmnóốớpqrstúứvxýảẳẩbcdđẻểghỉklmnỏổởpqrstủửvxỷạặậbcdđẹệghịklmnọộợpqrstụựvxỵãẵẫbcdđẽễghĩklmnõỗỡpqrstũữvxỹAĂÂBCDĐEÊGHIKLMNOÔƠPQRSTUƯVXYÀẰẦBCDĐÈỀGHÌKLMNÒỒỜPQRSTÙỪVXỲÁẮẤBCDĐÉẾGHÍKLMNÓỐỚPQRSTÚỨVXÝẠẶẬBCDĐẸỆGHỊKLMNỌỘỢPQRSTỤỰVXỴẢẲẨBCDĐẺỂGHỈKLMNỎỔỞPQRSTỦỬVXỶÃẴẪBCDĐẼỄGHĨKLMNÕỖỠPQRSTŨỮVXỸ,._]"
    
    for char in text:
        if char !=' ':
            if len(re.findall(pattern, char)) != 0:
                s+=char
            elif char == '_':
                s+=char
        else:
            s+=char
    s = re.sub('\\s+',' ',s)
    return s.strip()

def normalize_elonge_word(sent):
    s_new = ''
    for word in sent.split(' '):
        word_new = ' '
        for char in word.strip():
            if char != word_new[-1]:
                word_new += char
        s_new += word_new.strip() + ' '
    return s_new.strip()

correct_mapping = {
    "ship": "vận chuyển",
    "shop": "cửa hàng",
    "m": "mình",
    "mik": "mình",
    "ko": "không",
    "k": " không ",
    "kh": "không",
    "khong": "không",
    "kg": "không",
    "khg": "không",
    "tl": "trả lời",
    "r": "rồi",
    "fb": "mạng xã hội", 
    "face": "mạng xã hội",
    "thanks": "cảm ơn",
    "thank": "cảm ơn",
    "tks": "cảm ơn",
    "tk": "cảm ơn",
    "ok": "tốt",
    "dc": "được",
    "vs": "với",
    "đt": "điện thoại",
    "thjk": "thích",
    "qá": "quá",
    "trể": "trễ",
    "bgjo": "bao giờ",
    "bùn": "buồn"
}
def tokmap(tok):
    if tok.lower() in correct_mapping:
        return correct_mapping[tok.lower()]
    else:
        return tok


def clean_doc(doc, lower_case=True, word_segment=True, max_length=256):
    if not doc:
        return ""
    #  Chuẩn hóa văn bản 
    doc = normalText(doc)

    # Chèn khoảng trắng quanh dấu "?" để tokenizer xử lý dễ hơn
    doc = re.sub(r"\?", r" ? ", doc)

    # Thay thế số bằng token "số"
    doc = re.sub(r"[0-9]+", " số ", doc)

    # Chuẩn hóa khoảng trắng
    doc = re.sub(r"\s+", " ", doc)

    # Chuẩn hóa các từ bị kéo dài (vd: "đẹppppp" → "đẹp")
    doc = normalize_elonge_word(doc)

    # Xử lý từ đặc biệt "giá tiền"
    if word_segment:
        doc = doc.replace("giá _ tiền", "giá_tiền").replace("giátiền", "giá_tiền")
    else:
        doc = doc.replace("giá _ tiền", "giá tiền").replace("giátiền", "giá tiền")

    # Chuẩn hóa khoảng trắng lần nữa
    doc = re.sub(r"\s+", " ", doc).strip()

    # Map từng token
    tokens = map(tokmap, doc.split())
    doc = " ".join(tokens)

    # Nếu chuỗi quá dài, giữ lại đầu + cuối (cắt giữa)
    array = doc.split()
    if len(array) > max_length:
        half = max_length // 2
        doc = " ".join(array[:half] + array[-half:])

    # 12. Thay thế một số mẫu đặc biệt
    doc = doc.replace(". . .", ".")

    return re.sub(r"\s+", " ", doc).strip()
