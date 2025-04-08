import numpy as np
from PIL import Image, ImageOps, ImageFilter

class Preprocessor:
    def __init__(self, final_width=256, 
                 trim=False, transform_mode="none", FM_halftone=False,
                 ):
        """
        width, height: 중간 단계의 목표 크기(패딩/크롭용)
        final_width: 최종 리사이즈할 크기
        trim: True 이면 크롭, False 이면 패딩
        transform_mode: "none" 또는 "ellipse"
        FM_halftone: True이면 FM halftone (Floyd-Steinberg 디더링) 적용
        """
        # self.width = width
        # self.height = height
        self.final_width = final_width
        self.trim = trim
        self.transform_mode = transform_mode
        self.FM_halftone = FM_halftone
        # self.vertical_paddings = tuple(vertical_paddings)

    def load_image_8bit_gray(self, config):
        """
        이미지를 8비트 그레이스케일로 로드하고, 크기가 크면 CenterCrop → White Padding → 최종 사이즈로 리사이즈.
        이후 (255 - arr)로 검정/흰색 반전.
        transform_mode 및 FM_halftone 옵션에 따라 후처리를 적용.
        """
        img = Image.open(config["img_path"]).convert('L')  # 8비트 그레이스케일
        w, h = img.size
        self.width = w
        self.height = h

        if self.trim:
            # 아직 구현 안함
            raise NotImplementedError("trim=True는 아직 구현되지 않았습니다.")

        # 1) CenterCrop (너무 큰 경우)
        if w > self.width or h > self.height:
            crop_w = min(w, self.width)
            crop_h = min(h, self.height)
            left = (w - crop_w) // 2
            top = (h - crop_h) // 2
            img = img.crop((left, top, left + crop_w, top + crop_h))
            w, h = img.size

        # 2) 패딩 (흰색=255)
        pad_w = self.width
        pad_h = self.height
        padded_arr = np.full((pad_h, pad_w), fill_value=255, dtype=np.uint8)

        img_arr = np.array(img, dtype=np.uint8)
        img_h, img_w = img_arr.shape
        if img_w > pad_w or img_h > pad_h:
            raise ValueError("Cropping failed: 여전히 이미지가 목표보다 큽니다.")

        left = (pad_w - img_w) // 2
        top = (pad_h - img_h) // 2
        padded_arr[top:top + img_h, left:left + img_w] = img_arr

        # 3) final_width x final_width로 리사이즈(가로세로 비율은 여기선 강제함)
        padded_img = Image.fromarray(padded_arr)
        w, h = padded_img.size
        # 세로 비율만 맞추려면, 아래와 같이:
        ratio = self.final_width / float(w)
        new_w = self.final_width
        new_h = int(h * ratio)
        self.final_width = new_w
        self.final_height = new_h
        resized_img = padded_img.resize((new_w, new_h), Image.LANCZOS)

        # (1) 히스토그램 평활화
        if config["do_equalize"]:
            resized_img = histogram_equalization_excluding_bg(resized_img, bg_threshold=config["bg_threshold"])

        # (2) 로컬 콘트라스트
        if config["do_local_contrast"]:
            resized_img = local_contrast_enhancement_excluding_bg(
                resized_img, radius=config["local_contrast"]["radius"], 
                amount=config["local_contrast"]["amount"], 
                bg_threshold=config["bg_threshold"]
            )

        # (3) 톤 커브
        if config["do_tone_curve"]:
            if config["tone_params"] is None:
                config["tone_params"] = {}
            resized_img = partial_tone_curve_excluding_bg(
                resized_img,
                bg_threshold=config["bg_threshold"],
                **config["tone_params"]
            )

        # (255 - arr) 반전: 원래 이미지의 밝기 반전
        arr = (255 - np.array(resized_img, dtype=np.uint8))

        # FM_halftone 옵션 적용 (Floyd-Steinberg 에러 확산 디더링)
        if self.FM_halftone:
            arr = self._fm_halftone_transform(arr)

        # transform_mode 적용
        if self.transform_mode == "ellipse":
            arr = self._ellipse_transform(arr)
        elif self.transform_mode == "none":
            pass
        else:
            raise ValueError(f"알 수 없는 transform_mode: {self.transform_mode}")

        arr = pad_array(arr, tuple(config["vertical_paddings"]))

        return arr

    def _fm_halftone_transform(self, image):
        """
        Floyd-Steinberg 에러 확산 디더링을 이용하여 이미지를 이진화(FM halftone 효과)합니다.
        """
        h, w = image.shape
        # 연산을 위해 float32 복사본 생성
        arr = image.astype(np.float32).copy()
        for y in range(h):
            for x in range(w):
                old_pixel = arr[y, x]
                new_pixel = 255 if old_pixel >= 128 else 0
                arr[y, x] = new_pixel
                error = old_pixel - new_pixel

                # 오른쪽 픽셀
                if x + 1 < w:
                    arr[y, x + 1] += error * 7 / 16
                # 아래쪽 픽셀
                if y + 1 < h:
                    arr[y + 1, x] += error * 5 / 16
                # 왼쪽 아래 픽셀
                if x - 1 >= 0 and y + 1 < h:
                    arr[y + 1, x - 1] += error * 3 / 16
                # 오른쪽 아래 픽셀
                if x + 1 < w and y + 1 < h:
                    arr[y + 1, x + 1] += error * 1 / 16

        # 값의 범위를 [0, 255]로 제한 후 정수형으로 변환
        arr = np.clip(arr, 0, 255)
        return arr.astype(np.uint8)

    def _ellipse_transform(self, image):
        """
        이미지 중앙부만 타원형으로 남기고, 바깥은 0으로 만듭니다.
        """
        H, W = image.shape
        center_x = W // 2
        center_y = H // 2
        radius_x = center_x
        radius_y = center_y

        x, y = np.meshgrid(np.arange(W), np.arange(H))
        distance = ((x - center_x) / radius_x) ** 2 + ((y - center_y) / radius_y) ** 2
        image[distance > 1] = 0
        return image
    
    def get_final_height(self):
        if self.final_height is None:
            raise ValueError("final_height가 설정되지 않았습니다.")
        return self.final_height
    
    def get_final_width(self):
        if self.final_width is None:
            raise ValueError("final_height가 설정되지 않았습니다.")
        return self.final_width


def histogram_equalization_excluding_bg(img: Image.Image, bg_threshold=250) -> Image.Image:
    """
    '배경(흰색)' 픽셀을 히스토그램 계산에서 제외하고, 
    그 외(전경) 픽셀만 가지고 히스토그램 평활화를 수행한 뒤,
    배경 픽셀은 원본 밝기를 유지하고, 전경 픽셀만 평활화 결과로 매핑하는 함수.

    img : 흑백(L 모드) PIL 이미지
    bg_threshold : 이 값 이상의 픽셀은 배경으로 간주
    """
    # 그레이스케일로 강제 변환
    gray = img.convert("L")
    arr = np.array(gray, dtype=np.uint8)

    # 1) 배경 마스크( boolean ) 만들기
    #    True면 '전경(변환 대상)', False면 '배경(변환 제외)'
    foreground_mask = arr < bg_threshold

    # 2) 전경 픽셀 값만 추출하여 히스토그램 계산
    fg_values = arr[foreground_mask]  # 배경 제외한 픽셀
    if len(fg_values) == 0:
        # 전경 픽셀이 아예 없으면 그대로 반환
        return gray

    hist, _ = np.histogram(fg_values, bins=256, range=(0, 256))
    cdf = hist.cumsum()  # 누적 분포
    cdf_normalized = cdf / cdf[-1]  # 0~1로 정규화

    # 3) LUT(look-up table) 생성: cdf 값 기반 0~255 매핑
    lut = (cdf_normalized * 255).astype(np.uint8)

    # 4) 전경 픽셀만 LUT 적용, 배경은 원본 유지
    #    arr[x, y] -> lut[arr[x, y]] (전경에 대해서만)
    result_arr = arr.copy()
    result_arr[foreground_mask] = lut[result_arr[foreground_mask]]

    # 5) 결과 이미지를 PIL로 변환
    result_img = Image.fromarray(result_arr, mode="L")
    return result_img


def local_contrast_enhancement_excluding_bg(
    img: Image.Image, radius=2.0, amount=1.0, bg_threshold=250
) -> Image.Image:
    """
    언샤프 마스킹을 통한 로컬 콘트라스트 강화 시, 흰색 배경(bg_threshold 이상)은 
    강화 대상에서 제외.
    - radius : 언샤프 블러 반경
    - amount : 결과 블렌딩(0~1)
    - bg_threshold : 이 값 이상의 픽셀은 배경으로 간주(원본 유지)
    """
    gray = img.convert("L")
    arr = np.array(gray, dtype=np.uint8)

    # 전경(변환 대상) 마스크
    foreground_mask = arr < bg_threshold

    # 1) 전체 이미지에 대해 언샤프 마스크 적용
    sharpened = gray.filter(ImageFilter.UnsharpMask(radius=radius, percent=150, threshold=3))

    # 2) original과 sharpened 블렌딩(amount)
    blended = Image.blend(gray, sharpened, alpha=amount)
    blended_arr = np.array(blended, dtype=np.uint8)

    # 3) 배경 픽셀은 원본( arr )으로 유지, 전경은 블렌딩 결과로 치환
    result_arr = arr.copy()
    result_arr[foreground_mask] = blended_arr[foreground_mask]

    # PIL 이미지로 변환
    return Image.fromarray(result_arr, mode="L")


def partial_tone_curve_excluding_bg(
    img: Image.Image,
    in_low=50, in_high=200,
    out_low=30, out_high=220,
    bg_threshold=250
) -> Image.Image:
    """
    부분 톤 커브(선형 매핑) 적용 시, 흰색 배경(bg_threshold 이상)은 제외하고 적용.
    """
    gray = img.convert("L")
    arr = np.array(gray, dtype=np.uint8)

    foreground_mask = arr < bg_threshold

    # LUT 생성
    def tone_map(x):
        if x < in_low:
            if in_low == 0:
                return 0
            return int((out_low / in_low) * x)
        elif x > in_high:
            if in_high == 255:
                return 255
            return int(out_high + (255 - out_high) * (x - in_high) / (255 - in_high))
        else:
            return int(out_low + (out_high - out_low) * (x - in_low) / (in_high - in_low))

    lut = [tone_map(i) for i in range(256)]
    lut = np.array(lut, dtype=np.uint8)

    result_arr = arr.copy()
    # 전경만 LUT 적용
    result_arr[foreground_mask] = lut[result_arr[foreground_mask]]

    return Image.fromarray(result_arr, mode="L")


def enhance_image_excluding_bg(
    img: Image.Image,
    bg_threshold=250,
    do_equalize=True,
    do_local_contrast=True,
    do_tone_curve=True,
    radius=2.0,
    amount=0.8,
    tone_params=None
) -> Image.Image:
    """
    1) 히스토그램 평활화(배경 제외)
    2) 로컬 콘트라스트 강화(배경 제외)
    3) 부분 톤 커브 조정(배경 제외)
    순차 적용.

    - bg_threshold: 이 값 이상이면 '흰색 배경'으로 간주하여 변환 제외
    - do_equalize / do_local_contrast / do_tone_curve: 각 처리 여부
    - radius, amount: 언샤프마스크 기반 로컬 콘트라스트 파라미터
    - tone_params: partial_tone_curve_excluding_bg에 전달할 파라미터 dict
    """
    out_img = img.convert("L")

    # (1) 히스토그램 평활화
    if do_equalize:
        out_img = histogram_equalization_excluding_bg(out_img, bg_threshold=bg_threshold)

    # (2) 로컬 콘트라스트
    if do_local_contrast:
        out_img = local_contrast_enhancement_excluding_bg(
            out_img, radius=radius, amount=amount, bg_threshold=bg_threshold
        )

    # (3) 톤 커브
    if do_tone_curve:
        if tone_params is None:
            tone_params = {}
        out_img = partial_tone_curve_excluding_bg(
            out_img,
            bg_threshold=bg_threshold,
            **tone_params
        )

    return out_img

def pad_array(arr, vertical_paddings=(0,0)):
    return np.pad(arr, pad_width=(vertical_paddings, (0, 0)), mode='constant', constant_values=0)
