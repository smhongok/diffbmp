import tempfile
import xml.etree.ElementTree as ET
from cairosvg import svg2pdf, svg2png
import numpy as np
import os
import torch
import glob
import cv2
from copy import deepcopy

# Register default namespace so it exports without prefixes
ET.register_namespace('', 'http://www.w3.org/2000/svg')
SVG_NS = 'http://www.w3.org/2000/svg'

class PDFExporter:
    """
    Exports optimized vector elements into a combined SVG and renders to PDF.
    Supports cycling through multiple SVG shape templates (e.g., per-letter primitives).
    """
    def __init__(self,
                 svg_paths,
                 canvas_size: tuple,
                 viewbox_size: tuple,
                 alpha_upper_bound: float = 1.0,
                 stroke_width: float = 3.0):
        # Accept single string or list/tuple of strings
        if isinstance(svg_paths, (list, tuple)):
            self.svg_paths = list(svg_paths)
        else:
            self.svg_paths = [svg_paths]

        self.canvas_w, self.canvas_h = canvas_size
        self.view_w, self.view_h = viewbox_size
        self.norm_scale = 2 / max(self.view_w, self.view_h)
        self.alpha_upper_bound = alpha_upper_bound
        self.stroke_width = stroke_width

    def _remove_styles(self, root):
        # Remove style-related attributes and <style> tags
        for elem in root.iter():
            for attr in ['style', 'stroke', 'fill', 'stroke-opacity', 'fill-opacity']:
                if attr in elem.attrib:
                    del elem.attrib[attr]
        to_remove = [e for e in root.iter() if e.tag.endswith('style')]
        for child in to_remove:
            parent = self._find_parent(root, child)
            if parent is not None:
                parent.remove(child)

    def _find_parent(self, root, child):
        for elem in root.iter():
            for sub in elem:
                if sub is child:
                    return elem
        return None

    def export(self,
               x: torch.Tensor,
               y: torch.Tensor,
               r: torch.Tensor,
               theta: torch.Tensor,
               v: torch.Tensor,
               c: torch.Tensor,
               output_path: str,
               svg_hollow: bool = False):
        # Convert tensors to numpy arrays
        x_np = x.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        r_np = r.detach().cpu().numpy()
        theta_np = theta.detach().cpu().numpy()
        v_np = v.detach().cpu().numpy()
        c_np = torch.sigmoid(c).detach().cpu().numpy()
        alpha_vals = self.alpha_upper_bound * (1 / (1 + np.exp(-v_np)))

        # Create new SVG root element
        root = ET.Element(f'{{{SVG_NS}}}svg', {
            'width': str(self.canvas_w),
            'height': str(self.canvas_h),
            'viewBox': f"{-self.canvas_w/2} {-self.canvas_h/2} {self.canvas_w} {self.canvas_h}"
        })

        N = len(x_np)
        p = len(self.svg_paths)

        for i in reversed(range(N)):
            idx = i % p
            # Re-parse template to get fresh children
            tree = ET.parse(self.svg_paths[idx])
            template_root = tree.getroot()
            self._remove_styles(template_root)
            children = list(template_root)

            # Compute transform string
            theta_deg = np.degrees(theta_np[i])
            transform = (
                f"translate({x_np[i]-self.canvas_w/2},{y_np[i]-self.canvas_h/2}) "
                f"rotate({theta_deg}) "
                f"scale({r_np[i]}) "
                f"scale({self.norm_scale}) "
                f"translate({-self.view_w/2},{-self.view_h/2})"
            )
            g = ET.Element(f'{{{SVG_NS}}}g', {'transform': transform})

            # Style attributes
            r_color, g_color, b_color = c_np[i]
            r_int = int(np.clip(r_color * 255, 0, 255))
            g_int = int(np.clip(g_color * 255, 0, 255))
            b_int = int(np.clip(b_color * 255, 0, 255))
            if svg_hollow:
                g.attrib.update({
                    'stroke': f'rgb({r_int},{g_int},{b_int})',
                    'stroke-opacity': str(alpha_vals[i]),
                    'stroke-width': str(self.stroke_width),
                    'fill': f'rgb({r_int},{g_int},{b_int})',
                    'fill-opacity': '0'
                })
            else:
                g.attrib.update({
                    'fill': f'rgb({r_int},{g_int},{b_int})',
                    'fill-opacity': str(alpha_vals[i])
                })

            # Append fresh children
            for child in children:
                g.append(deepcopy(child))
            root.append(g)

        # Write combined SVG to temp and convert to PDF
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.svg')
        tree = ET.ElementTree(root)
        tree.write(tmp.name, encoding='utf-8', xml_declaration=True)
        svg2pdf(url=tmp.name, write_to=output_path)
        tmp.close()
        os.remove(tmp.name)

    def export_dropout_right_third(self,
                                   x: torch.Tensor,
                                   y: torch.Tensor,
                                   r: torch.Tensor,
                                   theta: torch.Tensor,
                                   v: torch.Tensor,
                                   c: torch.Tensor,
                                   output_path: str,
                                   svg_hollow: bool = False):
        W = self.canvas_w
        with torch.no_grad():
            keep_left = x < (W / 2)
            prob = torch.where(keep_left,
                                torch.ones_like(x),
                                (W - x) / (W / 2))
            keep_idx = keep_left | (torch.rand_like(prob) < prob)

            x_d, y_d, r_d = [t[keep_idx] for t in (x, y, r)]
            theta_d, v_d, c_d = [t[keep_idx] for t in (theta, v, c)]

        self.export(x_d, y_d, r_d, theta_d, v_d, c_d,
                    output_path=output_path,
                    svg_hollow=svg_hollow)
        print(f"drop-out PDF saved to {output_path}")

    def export_with_pngs(self,
                         x: torch.Tensor,
                         y: torch.Tensor,
                         r: torch.Tensor,
                         theta: torch.Tensor,
                         v: torch.Tensor,
                         c: torch.Tensor,
                         output_folder: str,
                         svg_hollow: bool = False,
                         border_color: str = 'red',
                         border_width: float = 6.0):
        """
        For each primitive added (bottom to top), export a PNG frame
        highlighting that primitive with a colored border.
        """
        # Prepare arrays
        x_np = x.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        r_np = r.detach().cpu().numpy()
        theta_np = theta.detach().cpu().numpy()
        v_np = v.detach().cpu().numpy()
        c_np = torch.sigmoid(c).detach().cpu().numpy()
        alpha_vals = self.alpha_upper_bound * (1 / (1 + np.exp(-v_np)))

        N = len(x_np)
        p = len(self.svg_paths)
        rev_indices = list(reversed(range(N)))

        os.makedirs(output_folder, exist_ok=True)

        # Step through adding one primitive at a time
        for frame_idx, idx in enumerate(rev_indices):
            # Build SVG root
            root = ET.Element(f'{{{SVG_NS}}}svg', {
                'width': str(self.canvas_w),
                'height': str(self.canvas_h),
                'viewBox': f"{-self.canvas_w/2} {-self.canvas_h/2} {self.canvas_w} {self.canvas_h}"
            })
            # Draw all primitives up to this step
            for j in rev_indices[:frame_idx+1]:
                i_mod = j % p
                tree = ET.parse(self.svg_paths[i_mod])
                tpl_root = tree.getroot()
                self._remove_styles(tpl_root)
                children = list(tpl_root)

                theta_deg = np.degrees(theta_np[j])
                transform = (
                    f"translate({x_np[j]-self.canvas_w/2},{y_np[j]-self.canvas_h/2}) "
                    f"rotate({theta_deg}) "
                    f"scale({r_np[j]}) "
                    f"scale({self.norm_scale}) "
                    f"translate({-self.view_w/2},{-self.view_h/2})"
                )

                # Main group
                g = ET.Element(f'{{{SVG_NS}}}g', {'transform': transform})
                # Style fill or hollow
                r_col, g_col, b_col = c_np[j]
                style = ({
                    'fill': f'rgb({int(r_col*255)},{int(g_col*255)},{int(b_col*255)})',
                    'fill-opacity': str(alpha_vals[j])
                } if not svg_hollow else {
                    'stroke': f'rgb({int(r_col*255)},{int(g_col*255)},{int(b_col*255)})',
                    'stroke-opacity': str(alpha_vals[j]),
                    'stroke-width': str(border_width),
                    'fill': f'rgb({int(r_col*255)},{int(g_col*255)},{int(b_col*255)})',
                    'fill-opacity': '0'
                })
                g.attrib.update(style)
                for ch in children:
                    g.append(deepcopy(ch))
                root.append(g)

                # If this is the newly added primitive, draw colored border
                if j == idx:
                    border_g = ET.Element(f'{{{SVG_NS}}}g', {'transform': transform})
                    border_g.attrib.update({
                        'fill': 'none',
                        'stroke': border_color,
                        'stroke-width': str(border_width)
                    })
                    for ch in children:
                        border_g.append(deepcopy(ch))
                    root.append(border_g)

            # Write temp SVG and export PNG with white background
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.svg')
            ET.ElementTree(root).write(tmp.name, encoding='utf-8', xml_declaration=True)
            frame_path = os.path.join(output_folder, f'frame_{frame_idx:04d}.png')
            svg2png(
                url=tmp.name,
                write_to=frame_path,
                output_width=self.canvas_w,
                background_color='white'
            )
            tmp.close()
            os.remove(tmp.name)
            print(f"Wrote frame {frame_idx} -> {frame_path}")

        self.make_mp4(frames_folder=output_folder,
                        video_path=output_folder[:-1]+".mp4",
                        )


    def make_mp4(self,
                frames_folder: str,
                video_path: str,
                fps: int = 60):
        """
        Assemble all PNG frames in `frames_folder` named 'frame_*.png' into an MP4.
        """
        frame_paths = sorted(glob.glob(os.path.join(frames_folder, 'frame_*.png')))
        if not frame_paths:
            raise ValueError(f"No frames found in {frames_folder}")
        # Read first to get size
        first = cv2.imread(frame_paths[0])
        if first is None:
            raise RuntimeError(f"Failed to read {frame_paths[0]}")
        h, w = first.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
        for fp in frame_paths:
            img = cv2.imread(fp)
            if img is None:
                continue
            writer.write(img)
        writer.release()
        print(f"MP4 video saved to {video_path}")


    def export_to_mp4_incremental(self,
                                x: torch.Tensor,
                                y: torch.Tensor,
                                r: torch.Tensor,
                                theta: torch.Tensor,
                                v: torch.Tensor,
                                c: torch.Tensor,
                                video_path: str,
                                fps: int = 60,
                                svg_hollow: bool = False,
                                border_color: str = 'red',
                                border_width: float = 6.0):
        """
        - primitive를 하나씩 누적해가는 단일 루프
        - 루프 안에서 바로 PNG 프레임으로 변환 → VideoWriter에 기록
        """
        import io
        import numpy as np
        import cv2
        import xml.etree.ElementTree as ET
        from cairosvg import svg2png

        # 1) 텐서 → NumPy
        x_np     = x.detach().cpu().numpy()
        y_np     = y.detach().cpu().numpy()
        r_np     = r.detach().cpu().numpy()
        theta_np = theta.detach().cpu().numpy()
        v_np     = v.detach().cpu().numpy()
        c_np     = torch.sigmoid(c).detach().cpu().numpy()
        alpha_vals = self.alpha_upper_bound * (1 / (1 + np.exp(-v_np)))

        N = len(x_np)
        p = len(self.svg_paths)
        rev_indices = list(reversed(range(N)))

        # 2) 빈 SVG 루트 하나만 만든 뒤, 프리미티브와 강조 테두리를 여기에 누적
        root = ET.Element(f'{{{SVG_NS}}}svg', {
            'width': str(self.canvas_w),
            'height': str(self.canvas_h),
            'viewBox': f"{-self.canvas_w/2} {-self.canvas_h/2} {self.canvas_w} {self.canvas_h}"
        })

        # 3) 비디오 크기 알아내기 — 첫 프레임 생성
        # (여기서는 첫 primitive 하나만 추가한 임시 root 생성)
        first_idx = rev_indices[0]
        self._append_primitive_to_root(root, first_idx,
                                    x_np, y_np, r_np, theta_np, c_np, alpha_vals,
                                    svg_hollow, border_color, border_width)
        svg_bytes = ET.tostring(root, encoding='utf-8', xml_declaration=True)
        png_bytes = svg2png(bytestring=svg_bytes,
                            output_width=self.canvas_w,
                            background_color='white')
        arr = np.frombuffer(png_bytes, dtype=np.uint8)
        first = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        h, w = first.shape[:2]

        # 4) VideoWriter 초기화
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
        import time
        # 5) 단일 루프: primitive 하나씩 누적 → 매 사이클마다 한 프레임 기록
        #    첫 번째는 이미 root에 추가되어 있으므로 enumerate 시작점 0
        for frame_idx, idx in enumerate(rev_indices):
            t_start = time.time()

            if frame_idx > 0:
                t_primitive_start = time.time()
                self._append_primitive_to_root(
                    root, idx,
                    x_np, y_np, r_np, theta_np, c_np, alpha_vals,
                    svg_hollow, border_color, border_width
                )
                t_primitive_end = time.time()
                print(f"[{frame_idx}] Primitive 추가 시간: {t_primitive_end - t_primitive_start:.4f}초")
            else:
                print(f"[{frame_idx}] 첫 프레임: Primitive 추가 생략")

            t_svg_start = time.time()
            svg_bytes = ET.tostring(root, encoding='utf-8', xml_declaration=True)
            t_svg_end = time.time()

            t_png_start = time.time()
            png_bytes = svg2png(
                bytestring=svg_bytes,
                output_width=self.canvas_w,
                background_color='white'
            )
            t_png_end = time.time()

            t_decode_start = time.time()
            arr = np.frombuffer(png_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
            t_decode_end = time.time()

            t_write_start = time.time()
            writer.write(frame)
            t_write_end = time.time()

            print(f"[{frame_idx}] SVG to string: {t_svg_end - t_svg_start:.4f}초")
            print(f"[{frame_idx}] SVG to PNG:   {t_png_end - t_png_start:.4f}초")
            print(f"[{frame_idx}] PNG decode:   {t_decode_end - t_decode_start:.4f}초")
            print(f"[{frame_idx}] Frame write:  {t_write_end - t_write_start:.4f}초")
            print(f"[{frame_idx}] 전체 루프 시간: {t_write_end - t_start:.4f}초\n")
            print("asdf")

        writer.release()
        print(f"Saved MP4 video to {video_path}")

    # -----------------------------------------------------
    def _append_primitive_to_root(self, root, j,
                                x_np, y_np, r_np, theta_np, c_np, alpha_vals,
                                svg_hollow, border_color, border_width):
        """
        root(<svg> Element)에
        - j번째 프리미티브와
        - 강조용 테두리(이 프리미티브 전용)를 append 해 주는 헬퍼
        """
        tree     = ET.parse(self.svg_paths[j % len(self.svg_paths)])
        tpl_root = tree.getroot()
        self._remove_styles(tpl_root)
        children = list(tpl_root)

        theta_deg = np.degrees(theta_np[j])
        transform = (
            f"translate({x_np[j]-self.canvas_w/2},{y_np[j]-self.canvas_h/2}) "
            f"rotate({theta_deg}) "
            f"scale({r_np[j]}) "
            f"scale({self.norm_scale}) "
            f"translate({-self.view_w/2},{-self.view_h/2})"
        )

        # 1) 메인 도형
        g = ET.Element(f'{{{SVG_NS}}}g', {'transform': transform})
        if not svg_hollow:
            g.attrib.update({
                'fill':   f'rgb({int(c_np[j][0]*255)},{int(c_np[j][1]*255)},{int(c_np[j][2]*255)})',
                'fill-opacity': str(alpha_vals[j])
            })
        else:
            g.attrib.update({
                'stroke':       f'rgb({int(c_np[j][0]*255)},{int(c_np[j][1]*255)},{int(c_np[j][2]*255)})',
                'stroke-opacity': str(alpha_vals[j]),
                'stroke-width': str(border_width),
                'fill':         f'rgb({int(c_np[j][0]*255)},{int(c_np[j][1]*255)},{int(c_np[j][2]*255)})',
                'fill-opacity': '0'
            })
        for ch in children:
            g.append(deepcopy(ch))
        root.append(g)

        # 2) 강조 테두리
        border_g = ET.Element(f'{{{SVG_NS}}}g', {'transform': transform})
        border_g.attrib.update({
            'fill': 'none',
            'stroke': border_color,
            'stroke-width': str(border_width)
        })
        for ch in children:
            border_g.append(deepcopy(ch))
        root.append(border_g)
