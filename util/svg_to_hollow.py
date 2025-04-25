#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import re
from xml.dom import minidom
from xml.etree import ElementTree as ET

def convert_svg_to_hollow(input_file, output_file, stroke_width=None, stroke_color='#000000'):
    """
    SVG 파일을 hollow 형태로 변환하는 함수
    
    Args:
        input_file (str): 입력 SVG 파일 경로
        output_file (str): 출력 SVG 파일 경로
        stroke_width (float): 테두리 두께 (None이면 자동 계산)
        stroke_color (str): 테두리 색상
    """
    # XML 네임스페이스 등록
    ET.register_namespace('', 'http://www.w3.org/2000/svg')
    ET.register_namespace('xlink', 'http://www.w3.org/1999/xlink')
    
    try:
        # SVG 파일 파싱
        tree = ET.parse(input_file)
        root = tree.getroot()
        
        # SVG 네임스페이스 정의
        svg_ns = {'svg': 'http://www.w3.org/2000/svg'}
        
        # 원본 뷰박스와 크기 정보 얻기
        viewBox = root.get('viewBox')
        width = root.get('width')
        height = root.get('height')
        
        # 이미지 크기에 따라 테두리 두께 조정
        adjusted_width = None
        
        # viewBox가 없는 경우 width와 height로부터 생성
        if not viewBox and width and height:
            try:
                w = float(width.rstrip('px').rstrip('em').rstrip('pt').rstrip('%'))
                h = float(height.rstrip('px').rstrip('em').rstrip('pt').rstrip('%'))
                viewBox = f"0 0 {w} {h}"
                root.set('viewBox', viewBox)
                print(f"viewBox가 없어 생성했습니다: {viewBox}")
            except (ValueError, TypeError):
                pass
        
        # 테두리 두께 자동 계산
        if viewBox:
            # viewBox의 크기에 비례하여 적절한 테두리 두께 계산
            vb_parts = viewBox.split()
            if len(vb_parts) == 4:
                vb_min_x = float(vb_parts[0])
                vb_min_y = float(vb_parts[1])
                vb_width = float(vb_parts[2])
                vb_height = float(vb_parts[3])
                
                # 이미지 크기에 따른 자동 테두리 두께 계산
                if stroke_width is None:
                    # 작은 이미지
                    if min(vb_width, vb_height) <= 50:
                        adjusted_width = min(vb_width, vb_height) * 0.02  # 2%
                    # 중간 크기 이미지
                    elif min(vb_width, vb_height) <= 200:
                        adjusted_width = min(vb_width, vb_height) * 0.01  # 1%
                    # 큰 이미지
                    else:
                        adjusted_width = min(vb_width, vb_height) * 0.005  # 0.5%
                    
                    # 최소/최대 값 범위 설정
                    adjusted_width = max(adjusted_width, 0.5)  # 최소 0.5
                    print(f"테두리 두께가 자동 계산되었습니다: {adjusted_width:.1f}")
                else:
                    # 사용자 지정 값 사용
                    adjusted_width = stroke_width
                
                # 테두리 두께를 고려하여 viewBox 확장
                # 테두리 두께의 절반만큼 각 방향으로 확장 (테두리가 중앙에서 양쪽으로 늘어나므로)
                padding = adjusted_width / 2
                
                # 새로운 viewBox 설정 (테두리가 잘리지 않도록 padding 추가)
                new_viewBox = f"{vb_min_x - padding} {vb_min_y - padding} {vb_width + padding*2} {vb_height + padding*2}"
                root.set('viewBox', new_viewBox)
        
        # viewBox가 없는 경우 기본 테두리 두께 설정
        if adjusted_width is None:
            if stroke_width is not None:
                adjusted_width = stroke_width
            else:
                adjusted_width = 1.0  # 기본값
        
        # 다른 속성도 테두리 두께를 고려하여 조정
        if width and height:
            try:
                # 단위 추출
                w_str = width
                h_str = height
                
                w_unit = ""
                h_unit = ""
                
                # 숫자 부분과 단위 부분 분리
                w_match = re.match(r'([0-9.]+)([a-z%]*)', w_str)
                h_match = re.match(r'([0-9.]+)([a-z%]*)', h_str)
                
                if w_match:
                    w = float(w_match.group(1))
                    w_unit = w_match.group(2)
                else:
                    w = float(w_str)
                
                if h_match:
                    h = float(h_match.group(1))
                    h_unit = h_match.group(2)
                else:
                    h = float(h_str)
                
                # 확장된 크기 계산 (padding은 테두리 두께)
                padding = adjusted_width
                w_new = w + padding * 2
                h_new = h + padding * 2
                
                # 새 크기 설정
                root.set('width', f"{w_new}{w_unit}")
                root.set('height', f"{h_new}{h_unit}")
            except (ValueError, TypeError) as e:
                print(f"크기 조정 중 오류 발생: {e}")
            
        # <style> 요소 찾기 또는 생성
        style_elem = root.find('.//svg:style', svg_ns)
        
        if style_elem is None:
            # style 요소가 없으면 생성
            style_elem = ET.SubElement(root, '{http://www.w3.org/2000/svg}style')
            style_elem.set('type', 'text/css')
            style_elem.text = '\n\t.st0{fill:none;stroke:' + stroke_color + ';stroke-width:' + str(adjusted_width) + ';stroke-linejoin:round;stroke-linecap:round;}\n'
        else:
            # 기존 style 요소가 있으면 hollow 스타일 대체 또는 추가
            style_text = style_elem.text or ""
            if '.st0' in style_text:
                # 기존 st0 클래스를 업데이트
                style_text = re.sub(r'\.st0\{[^}]*\}', '.st0{fill:none;stroke:' + stroke_color + ';stroke-width:' + 
                                   str(adjusted_width) + ';stroke-linejoin:round;stroke-linecap:round;}', style_text)
                style_elem.text = style_text
            else:
                # 새로운 st0 클래스 추가
                style_elem.text = style_text + '\n\t.st0{fill:none;stroke:' + stroke_color + ';stroke-width:' + str(adjusted_width) + ';stroke-linejoin:round;stroke-linecap:round;}\n'
        
        # 모든 path, rect, circle, ellipse 등의 요소에 hollow 스타일 적용
        for elem in root.findall('.//{http://www.w3.org/2000/svg}path') + \
                    root.findall('.//{http://www.w3.org/2000/svg}rect') + \
                    root.findall('.//{http://www.w3.org/2000/svg}circle') + \
                    root.findall('.//{http://www.w3.org/2000/svg}ellipse') + \
                    root.findall('.//{http://www.w3.org/2000/svg}polygon'):
            
            # 원본 스타일 정보 저장
            original_style = {}
            if 'style' in elem.attrib:
                style_text = elem.attrib['style']
                # 중요한 원본 스타일 속성 보존
                stroke_match = re.search(r'stroke-width:([^;]+);', style_text)
                if stroke_match:
                    original_style['stroke-width'] = stroke_match.group(1)
                
                # 기존 스타일에서 fill 속성만 제거
                style_text = re.sub(r'fill:[^;]+;', '', style_text)
                elem.attrib['style'] = style_text
            
            # fill 속성 제거
            if 'fill' in elem.attrib:
                del elem.attrib['fill']
                
            # 원본 테두리 속성 유지하면서 class 속성 설정
            elem.set('class', 'st0')
            
            # 사용자 지정 테두리 두께 설정
            if 'stroke-width' in original_style:
                elem.set('stroke-width', original_style['stroke-width'])
            else:
                elem.set('stroke-width', str(adjusted_width))
            
            # 테두리 색상 설정
            elem.set('stroke', stroke_color)
            elem.set('fill', 'none')
            
        # 결과를 문자열로 변환
        rough_string = ET.tostring(root, 'utf-8')
        
        # minidom을 사용하여 더 깔끔한 XML 형식으로 변환
        reparsed = minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent='\t')
        
        # 빈 줄 제거
        pretty_xml = os.linesep.join([s for s in pretty_xml.splitlines() if s.strip()])
        
        # 결과 저장
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(pretty_xml)
            
        print(f"Hollow SVG가 성공적으로 생성되었습니다: {output_file}")
        print(f"적용된 테두리 두께: {adjusted_width}")
        if viewBox:
            print(f"원본 viewBox: {viewBox}")
            print(f"수정된 viewBox: {new_viewBox if 'new_viewBox' in locals() else '변경 없음'}")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()

def main():
    """
    메인 함수: 명령줄 인자를 처리하고 SVG 변환 함수 호출
    """
    parser = argparse.ArgumentParser(description='SVG 파일을 hollow 형태로 변환합니다.')
    parser.add_argument('input_file', help='입력 SVG 파일 경로')
    parser.add_argument('--output', '-o', help='출력 SVG 파일 경로 (기본값: input_file에 _hollow 접미사 추가)')
    parser.add_argument('--stroke-width', '-w', type=float, 
                        help='테두리 두께 (기본값: 이미지 크기에 따라 자동 계산)')
    parser.add_argument('--stroke-color', '-c', default='#000000', help='테두리 색상 (기본값: #000000)')
    
    args = parser.parse_args()
    
    # 출력 파일명이 지정되지 않았으면 입력 파일명에 _hollow 접미사 추가
    if args.output is None:
        base_name, ext = os.path.splitext(args.input_file)
        args.output = f"{base_name}_hollow{ext}"
    
    convert_svg_to_hollow(args.input_file, args.output, args.stroke_width, args.stroke_color)

if __name__ == '__main__':
    main() 