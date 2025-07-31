import numpy as np
from PIL import Image, ImageOps, ImageFilter

class Preprocessor:
    def __init__(self, final_width=256, 
                 trim=False, transform_mode="none", FM_halftone=False,
                 ):
        """
        final_width: Final size to resize to
        trim: If True, crop; if False, pad
        transform_mode: "none" or "ellipse"
        FM_halftone: If True, apply FM halftone (Floyd-Steinberg dithering)
        """
        self.final_width = final_width
        self.trim = trim
        self.transform_mode = transform_mode
        self.FM_halftone = FM_halftone

    def load_image_8bit_gray(self, config):
        """
        Load image as 8-bit grayscale, and if large, apply CenterCrop → White Padding → resize to final size.
        Then invert with (255 - arr).
        Apply post-processing based on transform_mode and FM_halftone options.
        """
        img = Image.open(config["img_path"]).convert('L')  # 8-bit grayscale
        w, h = img.size
        self.width = w
        self.height = h

        if self.trim:
            # Not yet implemented
            raise NotImplementedError("trim=True is not yet implemented.")

        # 1) CenterCrop (if too large)
        if w > self.width or h > self.height:
            crop_w = min(w, self.width)
            crop_h = min(h, self.height)
            left = (w - crop_w) // 2
            top = (h - crop_h) // 2
            img = img.crop((left, top, left + crop_w, top + crop_h))
            w, h = img.size

        # 2) Padding (white=255)
        pad_w = self.width
        pad_h = self.height
        padded_arr = np.full((pad_h, pad_w), fill_value=255, dtype=np.uint8)

        img_arr = np.array(img, dtype=np.uint8)
        img_h, img_w = img_arr.shape
        if img_w > pad_w or img_h > pad_h:
            raise ValueError("Cropping failed: image is still larger than target.")

        left = (pad_w - img_w) // 2
        top = (pad_h - img_h) // 2
        padded_arr[top:top + img_h, left:left + img_w] = img_arr

        # 3) Resize to final_width x final_width (aspect ratio is enforced here)
        padded_img = Image.fromarray(padded_arr)
        w, h = padded_img.size
        # To match vertical ratio only, do:
        ratio = self.final_width / float(w)
        new_w = self.final_width
        new_h = int(h * ratio)
        self.final_width = new_w
        self.final_height = new_h
        resized_img = padded_img.resize((new_w, new_h), Image.LANCZOS)

        # (1) Histogram equalization
        if config["do_equalize"]:
            resized_img = histogram_equalization_excluding_bg(resized_img, bg_threshold=config["bg_threshold"])

        # (2) Local contrast
        if config["do_local_contrast"]:
            resized_img = local_contrast_enhancement_excluding_bg(
                resized_img, radius=config["local_contrast"]["radius"], 
                amount=config["local_contrast"]["amount"], 
                bg_threshold=config["bg_threshold"]
            )

        # (3) Tone curve
        if config["do_tone_curve"]:
            if config["tone_params"] is None:
                config["tone_params"] = {}
            resized_img = partial_tone_curve_excluding_bg(
                resized_img,
                bg_threshold=config["bg_threshold"],
                **config["tone_params"]
            )

        # Invert (255 - arr): invert the brightness of original image
        arr = (255 - np.array(resized_img, dtype=np.uint8))

        # Apply FM_halftone option (Floyd-Steinberg error diffusion dithering)
        if self.FM_halftone:
            arr = self._fm_halftone_transform(arr)

        # Apply transform_mode
        if self.transform_mode == "ellipse":
            arr = self._ellipse_transform(arr)
        elif self.transform_mode == "none":
            pass
        else:
            raise ValueError(f"Unknown transform_mode: {self.transform_mode}")

        arr = pad_array(arr, tuple(config["vertical_paddings"]))

        return arr

    def load_image_8bit_color(self, config):
        """
        Load image as 8-bit color (RGB), and if large, apply CenterCrop → White Padding → resize to final size.
        Then invert with (255 - arr).
        Apply post-processing based on transform_mode and FM_halftone options.
        """
        img = Image.open(config["img_path"]).convert('RGB')  # 8-bit color
        w, h = img.size
        
        # Store original image dimensions for reference
        original_w, original_h = w, h
        
        # Use target dimensions for cropping/padding (don't overwrite with actual dimensions)
        target_w = getattr(self, 'width', w)  # Use existing width or default to image width
        target_h = getattr(self, 'height', h)  # Use existing height or default to image height
        
        if self.trim:
            # Not yet implemented
            raise NotImplementedError("trim=True is not yet implemented.")

        # 1) CenterCrop (if too large)
        if w > target_w or h > target_h:
            crop_w = min(w, target_w)
            crop_h = min(h, target_h)
            left = (w - crop_w) // 2
            top = (h - crop_h) // 2
            img = img.crop((left, top, left + crop_w, top + crop_h))
            w, h = img.size

        # 2) Padding (white=255)
        pad_w = target_w
        pad_h = target_h
        
        # Update self.width and self.height to the target dimensions for consistency
        self.width = target_w
        self.height = target_h
        padded_arr = np.full((pad_h, pad_w, 3), fill_value=255, dtype=np.uint8)  # 3-channel white padding

        img_arr = np.array(img, dtype=np.uint8)
        img_h, img_w, _ = img_arr.shape
        if img_w > pad_w or img_h > pad_h:
            raise ValueError("Cropping failed: image is still larger than target.")

        left = (pad_w - img_w) // 2
        top = (pad_h - img_h) // 2
        padded_arr[top:top + img_h, left:left + img_w, :] = img_arr

        # 3) Resize to final_width x final_width (aspect ratio is enforced here)
        padded_img = Image.fromarray(padded_arr)
        w, h = padded_img.size
        ratio = self.final_width / float(w)
        new_w = self.final_width
        new_h = int(h * ratio)
        self.final_width = new_w
        self.final_height = new_h
        resized_img = padded_img.resize((new_w, new_h), Image.LANCZOS)

        # (1) Histogram equalization
        if config["do_equalize"]:
            resized_img = histogram_equalization_excluding_bg(resized_img, bg_threshold=config["bg_threshold"])

        # (2) Local contrast
        if config["do_local_contrast"]:
            resized_img = local_contrast_enhancement_excluding_bg(
                resized_img, radius=config["local_contrast"]["radius"], 
                amount=config["local_contrast"]["amount"], 
                bg_threshold=config["bg_threshold"]
            )

        # (3) Tone curve
        if config["do_tone_curve"]:
            if config["tone_params"] is None:
                config["tone_params"] = {}
            resized_img = partial_tone_curve_excluding_bg(
                resized_img,
                bg_threshold=config["bg_threshold"],
                **config["tone_params"]
            )

        # (255 - arr) inversion: invert brightness of color image
        arr = (np.array(resized_img, dtype=np.uint8))

        # Apply FM_halftone option (Floyd-Steinberg error diffusion dithering)
        if self.FM_halftone:
            arr = self._fm_halftone_transform(arr)

        # Apply transform_mode
        if self.transform_mode == "ellipse":
            arr = self._ellipse_transform(arr)
        elif self.transform_mode == "none":
            pass
        else:
            raise ValueError(f"Unknown transform_mode: {self.transform_mode}")

        arr = pad_array_color(arr, tuple(config["vertical_paddings"]))

        return arr

    def _fm_halftone_transform(self, image):
        """
        Binarize the image using Floyd-Steinberg error diffusion dithering (FM halftone effect).
        """
        h, w = image.shape
        # Create a float32 copy for calculations
        arr = image.astype(np.float32).copy()
        for y in range(h):
            for x in range(w):
                old_pixel = arr[y, x]
                new_pixel = 255 if old_pixel >= 128 else 0
                arr[y, x] = new_pixel
                error = old_pixel - new_pixel

                # Right pixel
                if x + 1 < w:
                    arr[y, x + 1] += error * 7 / 16
                # Bottom pixel
                if y + 1 < h:
                    arr[y + 1, x] += error * 5 / 16
                # Bottom-left pixel
                if x - 1 >= 0 and y + 1 < h:
                    arr[y + 1, x - 1] += error * 3 / 16
                # Bottom-right pixel
                if x + 1 < w and y + 1 < h:
                    arr[y + 1, x + 1] += error * 1 / 16

        # Limit values to range [0, 255] and convert to integer
        arr = np.clip(arr, 0, 255)
        return arr.astype(np.uint8)

    def _ellipse_transform(self, image):
        """
        Keep only the elliptical center part of the image, setting the outside to 0.
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
            raise ValueError("final_height has not been set.")
        return self.final_height
    
    def get_final_width(self):
        if self.final_width is None:
            raise ValueError("final_width has not been set.")
        return self.final_width
    
    def load_dual_images_for_xy_dynamics(self, config):
        """
        Load two images for XY dynamics mode, cropping both to minimum dimensions.
        
        Args:
            config: Configuration dictionary containing 'img_paths' with two image paths
            
        Returns:
            tuple: (image1_tensor, image2_tensor) both as torch tensors with identical dimensions
        """
        if 'img_paths' not in config or len(config['img_paths']) != 2:
            raise ValueError("XY dynamics mode requires exactly 2 image paths in 'img_paths'")
        
        # Load both images first to determine minimum dimensions
        img1 = Image.open(config['img_paths'][0]).convert('RGB')
        img2 = Image.open(config['img_paths'][1]).convert('RGB')
        
        w1, h1 = img1.size
        w2, h2 = img2.size
        
        # Find minimum dimensions
        min_w = min(w1, w2)
        min_h = min(h1, h2)
        
        print(f"Image 1 size: {w1}x{h1}, Image 2 size: {w2}x{h2}")
        print(f"Cropping both to minimum size: {min_w}x{min_h}")
        
        # Process both images with the same minimum dimensions
        def process_image(img, target_w, target_h):
            w, h = img.size
            
            # Center crop to target dimensions
            if w > target_w or h > target_h:
                crop_w = min(w, target_w)
                crop_h = min(h, target_h)
                left = (w - crop_w) // 2
                top = (h - crop_h) // 2
                img = img.crop((left, top, left + crop_w, top + crop_h))
                w, h = img.size
            
            # White padding if needed
            if w < target_w or h < target_h:
                padded_arr = np.full((target_h, target_w, 3), fill_value=255, dtype=np.uint8)
                img_arr = np.array(img, dtype=np.uint8)
                img_h, img_w, _ = img_arr.shape
                
                left = (target_w - img_w) // 2
                top = (target_h - img_h) // 2
                padded_arr[top:top + img_h, left:left + img_w, :] = img_arr
                img = Image.fromarray(padded_arr)
            
            # Resize to final dimensions
            ratio = self.final_width / float(target_w)
            new_w = self.final_width
            new_h = int(target_h * ratio)
            self.final_height = new_h
            img = img.resize((new_w, new_h), Image.LANCZOS)
            
            """
            # Apply preprocessing options
            if config["do_equalize"]:
                img = histogram_equalization_excluding_bg(img, bg_threshold=config["bg_threshold"])
            
            if config["do_local_contrast"]:
                img = local_contrast_enhancement_excluding_bg(
                    img, radius=config["local_contrast"]["radius"], 
                    amount=config["local_contrast"]["amount"], 
                    bg_threshold=config["bg_threshold"]
                )
            
            if config["do_tone_curve"]:
                if config["tone_params"] is None:
                    config["tone_params"] = {}
                img = partial_tone_curve_excluding_bg(
                    img,
                    bg_threshold=config["bg_threshold"],
                    **config["tone_params"]
                )
            
            """

            # Convert to array
            arr = np.array(img, dtype=np.uint8)
            
            # Apply vertical padding
            arr = pad_array_color(arr, tuple(config["vertical_paddings"]))
            
            return arr
        
        # Process both images
        image1_array = process_image(img1, min_w, min_h)
        image2_array = process_image(img2, min_w, min_h)
        
        

        # Verify dimensions match
        if image1_array.shape != image2_array.shape:
            raise RuntimeError(f"Dimension mismatch after preprocessing: {image1_array.shape} vs {image2_array.shape}")
        
        print(f"Loaded dual images for XY dynamics: {config['img_paths'][0]} and {config['img_paths'][1]}")
        print(f"Final array dimensions: {image1_array.shape}")
        
        return image1_array, image2_array


def histogram_equalization_excluding_bg(img: Image.Image, bg_threshold=250) -> Image.Image:
    """
    Performs histogram equalization only on foreground pixels,
    excluding 'background (white)' pixels from the histogram calculation,
    keeping the original brightness of background pixels and
    mapping only the foreground pixels to the equalized result.

    img : Grayscale (L mode) PIL image
    bg_threshold : Pixels with values greater than or equal to this are considered background
    """
    # Force conversion to grayscale
    gray = img.convert("L")
    arr = np.array(gray, dtype=np.uint8)

    # 1) Create background mask (boolean)
    #    True is 'foreground (target for transformation)', False is 'background (excluded from transformation)'
    foreground_mask = arr < bg_threshold

    # 2) Extract only foreground pixel values for histogram calculation
    fg_values = arr[foreground_mask]  # Pixels excluding background
    if len(fg_values) == 0:
        # If there are no foreground pixels, return as is
        return gray

    hist, _ = np.histogram(fg_values, bins=256, range=(0, 256))
    cdf = hist.cumsum()  # Cumulative distribution
    cdf_normalized = cdf / cdf[-1]  # Normalize to 0~1

    # 3) Create LUT (look-up table): 0~255 mapping based on cdf values
    lut = (cdf_normalized * 255).astype(np.uint8)

    # 4) Apply LUT only to foreground pixels, keep background unchanged
    #    arr[x, y] -> lut[arr[x, y]] (only for foreground)
    result_arr = arr.copy()
    result_arr[foreground_mask] = lut[result_arr[foreground_mask]]

    # 5) Convert result image to PIL
    result_img = Image.fromarray(result_arr, mode="L")
    return result_img


def local_contrast_enhancement_excluding_bg(
    img: Image.Image, radius=2.0, amount=1.0, bg_threshold=250
) -> Image.Image:
    """
    For local contrast enhancement through unsharp masking, exclude white background 
    (values >= bg_threshold) from enhancement targets.
    - radius : Unsharp blur radius
    - amount : Result blending (0~1)
    - bg_threshold : Pixels with values greater than or equal to this are considered background (preserve original)
    """
    gray = img.convert("L")
    arr = np.array(gray, dtype=np.uint8)

    # Foreground (transformation target) mask
    foreground_mask = arr < bg_threshold

    # 1) Apply unsharp mask to the entire image
    sharpened = gray.filter(ImageFilter.UnsharpMask(radius=radius, percent=150, threshold=3))

    # 2) Blend original and sharpened (amount)
    blended = Image.blend(gray, sharpened, alpha=amount)
    blended_arr = np.array(blended, dtype=np.uint8)

    # 3) Keep background pixels as original (arr), replace foreground with blended result
    result_arr = arr.copy()
    result_arr[foreground_mask] = blended_arr[foreground_mask]

    # Convert to PIL image
    return Image.fromarray(result_arr, mode="L")


def partial_tone_curve_excluding_bg(
    img: Image.Image,
    in_low=50, in_high=200,
    out_low=30, out_high=220,
    bg_threshold=250
) -> Image.Image:
    """
    When applying partial tone curve (linear mapping), exclude white background (values >= bg_threshold).
    """
    gray = img.convert("L")
    arr = np.array(gray, dtype=np.uint8)

    foreground_mask = arr < bg_threshold

    # Create LUT
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
    # Apply LUT only to foreground
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
    Apply sequentially:
    1) Histogram equalization (excluding background)
    2) Local contrast enhancement (excluding background)
    3) Partial tone curve adjustment (excluding background)

    - bg_threshold: Values >= this are considered 'white background' and excluded from transformations
    - do_equalize / do_local_contrast / do_tone_curve: Whether to apply each process
    - radius, amount: Parameters for unsharp mask-based local contrast
    - tone_params: Dictionary of parameters to pass to partial_tone_curve_excluding_bg
    """
    out_img = img.convert("L")

    # (1) Histogram equalization
    if do_equalize:
        out_img = histogram_equalization_excluding_bg(out_img, bg_threshold=bg_threshold)

    # (2) Local contrast
    if do_local_contrast:
        out_img = local_contrast_enhancement_excluding_bg(
            out_img, radius=radius, amount=amount, bg_threshold=bg_threshold
        )

    # (3) Tone curve
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

def pad_array_color(arr, vertical_paddings=(0, 0)):
    """
    Padding function for color (RGB) images.
    Adds white (255) padding to top and bottom with specified size.
    """
    return np.pad(
        arr,
        pad_width=(vertical_paddings, (0, 0), (0, 0)),  # Don't pad the last axis (channels)
        mode='constant',
        constant_values=255  # White padding
    )