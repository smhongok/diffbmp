import numpy as np
from PIL import Image, ImageOps, ImageFilter
from imbrush.util.target_masks import binary_mask
from imbrush.util.constants import get_resampling_method
import cv2
import os
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
        resampling = get_resampling_method(config.get("resampling", "LANCZOS"))
        resized_img = padded_img.resize((new_w, new_h), resampling)

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
        Load image as 8-bit color (RGB) and resize.
        If square_crop is enabled: produces exact final_width x final_width output.
        Otherwise: maintains aspect ratio with variable height.
        Load image as 8-bit color (RGB) and resize.
        """
        img = Image.open(config["img_path"]).convert('RGB')
        w, h = img.size
        
        if self.trim:
            raise NotImplementedError("trim=True is not yet implemented.")

        # Check if square cropping is enabled
        square_crop = config.get("square_crop", False)
        
        if square_crop:
            # Smart resize + center crop to produce exact square output
            target_size = self.final_width
            
            # Resize to cover the target dimensions (maintain aspect ratio)
            img_aspect = w / h
            if img_aspect > 1:
                # Image is wider - match height and crop width
                new_h = target_size
                new_w = int(target_size * img_aspect)
            else:
                # Image is taller - match width and crop height
                new_w = target_size
                new_h = int(target_size / img_aspect)
            
            resampling = get_resampling_method(config.get("resampling", "LANCZOS"))
            resized_img = img.resize((new_w, new_h), resampling)
            
            # Center crop to exact target_size x target_size
            left = (new_w - target_size) // 2
            top = (new_h - target_size) // 2
            resized_img = resized_img.crop((left, top, left + target_size, top + target_size))
            
            self.final_width = target_size
            self.final_height = target_size
        else:
            # Original behavior: resize to final_width, maintaining aspect ratio
            ratio = self.final_width / float(w)
            new_w = self.final_width
            new_h = int(h * ratio)
            self.final_width = new_w
            self.final_height = new_h
            resampling = get_resampling_method(config.get("resampling", "LANCZOS"))
            resized_img = img.resize((new_w, new_h), resampling)

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

    def load_image_8bit_color_opacity(self, config, make_bg_white=True):
        """
        Load image as 8-bit color and opacity(RGBA), and if large, apply CenterCrop → White Padding → resize to final size.
        Args:
            config (dict): Configuration dictionary containing 'img_path'.
            make_bg_white (bool): If True, convert transparent background to white. This is for some cases that the input image has wierd RGB values in transparent area.
        Returns:
            arr (np.ndarray): Processed RGBA image array.
            binary_image (np.ndarray): Binary mask of the alpha channel.
        """
        img = Image.open(config["img_path"]).convert('RGBA')  # 8-bit color with alpha
        binary_image = (1-(np.array(img)[:, :, 3] > 0).astype(np.uint8)) * 255
        img_array = np.array(img)
        if config.get("mask_path") is not None:
            # Use provided mask path(s) to create binary mask
            mask_paths = config["mask_path"]
            
            # Handle both single path (string) and multiple paths (list)
            if isinstance(mask_paths, str):
                mask_paths = [mask_paths]
            
            img_w, img_h = img.size
            combined_mask = None
            
            # Load and combine all masks using OR operation
            for mask_path in mask_paths:
                mask_img = Image.open(mask_path).convert('L')
                
                # Resize mask to match image size if they don't match
                mask_w, mask_h = mask_img.size
                if (mask_w, mask_h) != (img_w, img_h):
                    target_w = min(img_w, mask_w)
                    target_h = min(img_h, mask_h)
                    mask_img = mask_img.resize((target_w, target_h), Image.NEAREST)
                    
                    # Only resize img once (on first iteration if needed)
                    if combined_mask is None and (img_w, img_h) != (target_w, target_h):
                        img = img.resize((target_w, target_h), Image.NEAREST)
                        img_w, img_h = target_w, target_h
                
                mask_array = np.array(mask_img)
                
                # Combine masks using OR operation (any non-zero pixel is considered filled)
                if combined_mask is None:
                    combined_mask = mask_array
                else:
                    combined_mask = np.maximum(combined_mask, mask_array)
            
            binary_image = 255 - combined_mask

        if make_bg_white:
            # Update img_array after potential resize
            img_array = np.array(img)

            # Convert transparent background (alpha=0) to white
            img_array[binary_image!=0, :3] = 255  # Set RGB to white
            img_array[binary_image!=0, 3] = 0  # Set alpha to 0
            img = Image.fromarray(img_array, mode='RGBA')


        w, h = img.size
        
        # Extract alpha channel as binary mask before resizing
        binary_image = (1 - (np.array(img)[:, :, 3] > 0).astype(np.uint8)) * 255

        if self.trim:
            raise NotImplementedError("trim=True is not yet implemented.")

        # Check if square cropping is enabled
        square_crop = config.get("square_crop", False)
        
        if square_crop:
            # Smart resize + center crop to produce exact square output
            target_size = self.final_width
            
            # Resize to cover the target dimensions (maintain aspect ratio)
            img_aspect = w / h
            if img_aspect > 1:
                # Image is wider - match height and crop width
                new_h = target_size
                new_w = int(target_size * img_aspect)
            else:
                # Image is taller - match width and crop height
                new_w = target_size
                new_h = int(target_size / img_aspect)
            
            resampling = get_resampling_method(config.get("resampling", "NEAREST"))
            resized_img = img.resize((new_w, new_h), resampling)
            
            # Center crop to exact target_size x target_size
            left = (new_w - target_size) // 2
            top = (new_h - target_size) // 2
            resized_img = resized_img.crop((left, top, left + target_size, top + target_size))
            
            # Resize and crop binary mask to match
            binary_image = Image.fromarray(binary_image, mode='L')
            binary_image = binary_image.resize((new_w, new_h), Image.NEAREST)
            binary_image = binary_image.crop((left, top, left + target_size, top + target_size))
            
            self.final_width = target_size
            self.final_height = target_size
        else:
            # Original behavior: resize to final_width, maintaining aspect ratio
            ratio = self.final_width / float(w)
            new_w = self.final_width
            new_h = int(h * ratio)
            self.final_width = new_w
            self.final_height = new_h
            resampling = get_resampling_method(config.get("resampling", "NEAREST"))
            resized_img = img.resize((new_w, new_h), resampling)

            # Resize binary mask to match
            binary_image = Image.fromarray(binary_image, mode='L')
            binary_image = binary_image.resize((new_w, new_h), Image.NEAREST)

        arr = np.array(resized_img, dtype=np.uint8)
        binary_image = np.array(binary_image, dtype=np.uint8)
        binary_image = (binary_image > 0) * 255
        binary_image = binary_image.astype(np.uint8)

        return arr, binary_image


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
    

    def load_gif_frames(self, gif_path, config):
        """
        Load GIF frames and preprocess each frame using the same pipeline as single images.
        Returns a list of preprocessed frame arrays.
        """
        frames = []
        
        # Open GIF and extract frames
        with Image.open(gif_path) as gif:
            for frame_idx in range(gif.n_frames):
                gif.seek(frame_idx)
                frame = gif.copy().convert('RGB')
                
                # Apply the same preprocessing pipeline as load_image_8bit_color
                frame_array = self._preprocess_single_frame(frame, config)
                frames.append(frame_array)
        
        return frames
    
    def load_video_frames(self, video_path, config, max_frames=None):
        """
        Load video frames and preprocess each frame using the same pipeline as single images.
        Returns a list of preprocessed frame arrays.
        """
        frames = []
        
        # Open video with OpenCV
        cap = cv2.VideoCapture(video_path)
        
        # If max_frames is None, set it to the total number of frames in the video
        if max_frames is None:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # If automatic frame count detection fails (returns 0), manually count frames
            if total_frames == 0:
                print("Automatic frame count detection failed, manually counting frames...")
                temp_cap = cv2.VideoCapture(video_path)
                total_frames = 0
                while True:
                    ret, _ = temp_cap.read()
                    if not ret:
                        break
                    total_frames += 1
                temp_cap.release()
                print(f"Manually counted {total_frames} frames in video")
            else:
                print(f"Auto-detected {total_frames} frames in video")
            
            max_frames = total_frames
            print(f"Processing all {max_frames} frames")
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if max_frames and frame_count >= max_frames:
                break
                
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            
            # Apply the same preprocessing pipeline as load_image_8bit_color
            frame_array = self._preprocess_single_frame(frame_pil, config)
            frames.append(frame_array)
            frame_count += 1
        
        cap.release()
        return frames
    
    def load_image_sequence(self, sequence_dir, config, extensions=('.png', '.jpg', '.jpeg')):
        """
        Load image sequence from directory and preprocess each frame.
        Returns a list of preprocessed frame arrays.
        """
        frames = []
        
        # Get all image files in directory
        image_files = []
        for ext in extensions:
            image_files.extend([f for f in os.listdir(sequence_dir) if f.lower().endswith(ext)])
        
        # Sort files naturally
        image_files.sort()
        
        for img_file in image_files:
            img_path = os.path.join(sequence_dir, img_file)
            frame = Image.open(img_path).convert('RGB')
            
            # Apply the same preprocessing pipeline as load_image_8bit_color
            frame_array = self._preprocess_single_frame(frame, config)
            frames.append(frame_array)
        
        return frames
    
    def _preprocess_single_frame(self, frame_pil, config):
        """
        Apply the same preprocessing pipeline as load_image_8bit_color to a single frame.
        This ensures consistent preprocessing across all frames.
        """
        img = frame_pil
        w, h = img.size
        
        # Store original dimensions for first frame to ensure consistency
        if not hasattr(self, 'width') or not hasattr(self, 'height'):
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
        padded_arr = np.full((pad_h, pad_w, 3), fill_value=255, dtype=np.uint8)

        img_arr = np.array(img, dtype=np.uint8)
        img_h, img_w = img_arr.shape[:2]
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
        
        # Store final dimensions for consistency
        if not hasattr(self, 'final_height'):
            self.final_height = new_h
        
        resampling = get_resampling_method(config.get("resampling", "LANCZOS"))
        resized_img = padded_img.resize((new_w, new_h), resampling)

        # Apply post-processing (same as load_image_8bit_color)
        if config["do_equalize"]:
            # Convert to grayscale for equalization, then back to RGB
            gray_img = resized_img.convert('L')
            eq_gray = histogram_equalization_excluding_bg(gray_img, bg_threshold=config["bg_threshold"])
            resized_img = Image.merge('RGB', [eq_gray, eq_gray, eq_gray])

        if config["do_local_contrast"]:
            # Convert to grayscale for local contrast, then back to RGB
            gray_img = resized_img.convert('L')
            lc_gray = local_contrast_enhancement_excluding_bg(
                gray_img, radius=config["local_contrast"]["radius"], 
                amount=config["local_contrast"]["amount"], 
                bg_threshold=config["bg_threshold"]
            )
            resized_img = Image.merge('RGB', [lc_gray, lc_gray, lc_gray])

        if config["do_tone_curve"]:
            if config["tone_params"] is None:
                config["tone_params"] = {}
            # Convert to grayscale for tone curve, then back to RGB
            gray_img = resized_img.convert('L')
            tc_gray = partial_tone_curve_excluding_bg(
                gray_img,
                bg_threshold=config["bg_threshold"],
                **config["tone_params"]
            )
            resized_img = Image.merge('RGB', [tc_gray, tc_gray, tc_gray])

        # Invert (255 - arr): invert the brightness of original image
        arr = (np.array(resized_img, dtype=np.uint8))

        # Apply FM_halftone option (Floyd-Steinberg error diffusion dithering)
        if self.FM_halftone:
            # Apply to each channel separately
            for c in range(3):
                arr[:, :, c] = self._fm_halftone_transform(arr[:, :, c])

        # Apply transform_mode
        if self.transform_mode == "ellipse":
            # Apply to each channel separately
            for c in range(3):
                arr[:, :, c] = self._ellipse_transform(arr[:, :, c])
        elif self.transform_mode == "none":
            pass
        else:
            raise ValueError(f"Unknown transform_mode: {self.transform_mode}")

        # Apply padding (handle both 2D and 3D arrays)
        if arr.ndim == 3:  # RGB image
            arr = pad_array_color(arr, tuple(config["vertical_paddings"]))
        else:  # Grayscale image
            arr = pad_array(arr, tuple(config["vertical_paddings"]))

        return arr


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